"""RAG xizmati: data/ papkasidagi JSON bilim bazasini yuklaydi va qidiradi.

Vazifasi:
  1. doctors.json / services.json / rules.json / faqs.json fayllarini o'qish.
  2. Har bir yozuvni qidiriladigan "chunk" ga aylantirish (BM25 indeksi).
  3. Foydalanuvchi savolidan kelib chiqib eng kerakli kontekstni topish.
  4. Grounding (tekshirish) uchun "ruxsat etilgan faktlar" lug'atini berish:
     shifokor ismlari, xizmat nomlari, narxlar. LLM o'ylab topgan faktni
     shu lug'at orqali ushlaymiz.

Tashqi og'ir kutubxonalar ishlatilmaydi (torch/sentence-transformers/chroma yo'q):
BM25 + fuzzy moslashtirish + ixtisoslik sinonimlari yetarli aniqlik beradi va
o'rnatishni oson saqlaydi.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Matn normalizatsiyasi
# --------------------------------------------------------------------------
STOPWORDS = {
    # o'zbek
    "va", "bilan", "uchun", "bu", "shu", "men", "siz", "biz", "u", "ular", "bor", "yoq", "yo'q",
    "kerak", "qanday", "qancha", "nima", "kim", "qaysi", "mumkin", "boladimi", "bo'ladimi",
    "iltimos", "rahmat", "salom", "assalomu", "alaykum", "yaxshi", "menga", "menda", "sizda",
    "klinikada", "klinikangizda", "haqida", "haqida?", "ayting", "aytib", "bering", "kerakmi",
    # rus
    "и", "в", "во", "на", "с", "со", "для", "это", "я", "вы", "мы", "он", "она", "есть", "нет",
    "надо", "нужно", "как", "сколько", "что", "кто", "какой", "можно", "ли", "пожалуйста",
    "спасибо", "здравствуйте", "у", "вас", "меня", "мне", "о", "об", "про",
    # ingliz
    "the", "a", "an", "is", "are", "do", "does", "i", "you", "we", "he", "she", "they", "have",
    "has", "how", "much", "what", "who", "which", "can", "could", "please", "thanks", "hello",
    "about", "for", "with", "and", "of", "to", "in", "on", "there", "any",
}

# O'zbek tilida ko'p uchraydigan qo'shimchalar (juda oddiy "stemming")
SUFFIXES = ("laringiz", "laring", "larimiz", "larni", "larga", "larda", "lardan", "lari",
            "ning", "dan", "ga", "da", "ni", "si", "lik", "cha", "mi", "chi")

# Rus tilidagi kelishik/ko'plik qo'shimchalari. Bilim bazasi o'zbek lotin
# alifbosida yozilgani uchun ruscha so'rovlar mos kelmay qoladi —
# shuning uchun ruscha so'zlar ham "o'zak"ka keltiriladi (консультация -> консультаци).
RU_SUFFIXES = ("ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ая", "яя", "ое", "ее",
               "ые", "ие", "ой", "ей", "ом", "ем", "ах", "ях", "ов", "ев", "ам", "ям",
               "а", "я", "у", "ю", "е", "ы", "и", "о")

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁʻʼ'’\u0400-\u04FF]+", re.UNICODE)


def normalize(text: str) -> str:
    """Matnni kichik harflarga o'tkazib, tinish belgilarini tozalaydi."""
    text = text.lower()
    text = text.replace("ʻ", "'").replace("ʼ", "'").replace("’", "'").replace("`", "'")
    text = text.replace("ё", "е")   # rus tilida ё/е farqi qidiruvga xalaqit beradi
    text = re.sub(r"[^\w\s'\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    """So'z o'zagini ajratadi (o'zbek va rus qo'shimchalari uchun)."""
    for suffix in SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    if _CYRILLIC_RE.search(token):
        # Ruscha so'zlar qisqaroq bo'ladi — qoldiq 3 belgidan boshlab yetarli
        for suffix in RU_SUFFIXES:
            if len(token) - len(suffix) >= 3 and token.endswith(suffix):
                return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        if len(raw) < 2:
            continue
        tokens.append(stem(raw))
    return tokens


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


# --------------------------------------------------------------------------
# Sinonimlar: bemor so'zlashi -> klinika atamalari
# --------------------------------------------------------------------------
SYNONYMS: dict[str, list[str]] = {
    "yurak": ["kardiolog", "kardiologiya", "ekg", "exokg", "yurak"],
    "yurakdan": ["kardiolog"],
    "bosim": ["kardiolog", "gipertenziya", "konsultatsiya"],
    "asab": ["nevrolog"],
    "nevroz": ["nevrolog"],
    "migren": ["nevrolog"],
    "bosh": ["nevrolog"],
    "uyqu": ["nevrolog"],
    "bel": ["nevrolog", "travmatolog", "massaj"],
    "tish": ["stomatolog", "karies"],
    "stomatolog": ["stomatolog", "karies", "tish"],
    "karies": ["stomatolog", "karies"],
    "kanal": ["stomatolog", "kanal"],
    "bola": ["pediatr", "bolam"],
    "bolam": ["pediatr"],
    "chaqaloq": ["pediatr"],
    "emlash": ["pediatr", "emlash"],
    "suyak": ["travmatolog"],
    "sinish": ["travmatolog"],
    "sinik": ["travmatolog"],
    "tizza": ["travmatolog"],
    "gips": ["travmatolog", "gips"],
    "chokish": ["travmatolog"],
    "cho'kish": ["travmatolog"],
    "teri": ["dermatolog"],
    "toshma": ["dermatolog"],
    "akne": ["dermatolog"],
    "zamburug": ["dermatolog"],
    "allergiya": ["dermatolog"],
    "buyrak": ["urolog", "buyrak"],
    "prostata": ["urolog"],
    "siydik": ["urolog", "siydik"],
    "ayol": ["ginekolog"],
    "homilador": ["ginekolog"],
    "homiladorlik": ["ginekolog"],
    "ginekolog": ["ginekolog"],
    "quloq": ["lor"],
    "tomoq": ["lor"],
    "burun": ["lor"],
    "gaymorit": ["lor"],
    "otit": ["lor"],
    "angina": ["lor"],
    "qalqonsimon": ["endokrinolog", "qalqonsimon"],
    "diabet": ["endokrinolog"],
    "qand": ["endokrinolog", "glyukoza"],
    "gormon": ["endokrinolog", "gormon"],
    "vazn": ["endokrinolog"],
    "oshqozon": ["gastroenterolog"],
    "ichak": ["gastroenterolog"],
    "jigar": ["gastroenterolog"],
    "gastrit": ["gastroenterolog"],
    "fgds": ["gastroenterolog", "fgds"],
    "koz": ["oftalmolog"],
    "ko'z": ["oftalmolog"],
    "ko'zoynak": ["oftalmolog"],
    "korish": ["oftalmolog"],
    "ko'rish": ["oftalmolog"],
    "tahlil": ["laboratoriya", "qon", "siydik"],
    "qon": ["qon", "laboratoriya"],
    "narx": ["narx", "price"],
    "narxi": ["narx"],
    "qancha": ["narx"],
    "necha": ["narx"],
    "pul": ["narx"],
    "arzon": ["narx"],
    "chegirma": ["chegirma", "qoida"],
    "ish": ["ish", "vaqt", "schedule"],
    "vaqt": ["vaqt", "ish"],
    "manzil": ["manzil", "address"],
    "qayerda": ["manzil"],
    "telefon": ["telefon", "raqam"],
    "raqam": ["telefon", "raqam"],
    "yozilish": ["yozilish", "qabul", "booking"],
    "qabul": ["qabul", "yozilish"],
    "navbat": ["qabul", "yozilish"],
    "checkup": ["check-up", "paket"],
    "check-up": ["check-up", "paket"],
    "paket": ["paket"],
    "sugurta": ["sug'urta", "insurance"],
    "sug'urta": ["sug'urta"],
    "karta": ["to'lov", "karta"],
    "tolov": ["to'lov"],
    "to'lov": ["to'lov"],
    "uy": ["uyga", "chaqirish"],
    "uyga": ["uyga"],
    "onlayn": ["onlayn", "video"],
    "video": ["onlayn", "video"],
    "travmatolog": ["travmatolog"],
    "ortoped": ["travmatolog"],
    "nevrolog": ["nevrolog"],
    "kardiolog": ["kardiolog"],
    "urolog": ["urolog"],
    "lor": ["lor"],
    "endokrinolog": ["endokrinolog"],
    "gastroenterolog": ["gastroenterolog"],
    "oftalmolog": ["oftalmolog"],
    "dermatolog": ["dermatolog"],
    "pediatr": ["pediatr"],
    "ginekolog": ["ginekolog"],
    "stomatologiya": ["stomatolog"],
}

# --------------------------------------------------------------------------
# Rus tili: bemor kirill alifbosida yozsa, bilim bazasi o'zbek lotin
# alifbosida yozilgan. Shuning uchun ruscha atamalarni bazadagi atamalarga
# bog'laymiz. Kalitlar `stem()` orqali o'zakka keltiriladi.
# --------------------------------------------------------------------------
_RU_SYNONYMS_RAW: dict[str, list[str]] = {
    # shifokorlar
    "кардиолог": ["kardiolog"], "кардиология": ["kardiolog"],
    "невролог": ["nevrolog"], "стоматолог": ["stomatolog"], "зуб": ["stomatolog", "tish", "karies"],
    "зубы": ["stomatolog"], "педиатр": ["pediatr"], "детский": ["pediatr"],
    "травматолог": ["travmatolog"], "дерматолог": ["dermatolog"], "уролог": ["urolog"],
    "гинеколог": ["ginekolog"], "лор": ["lor"], "эндокринолог": ["endokrinolog"],
    "гастроэнтеролог": ["gastroenterolog"], "окулист": ["oftalmolog"], "офтальмолог": ["oftalmolog"],
    "терапевт": ["terapevt"], "врач": ["shifokor"], "доктор": ["shifokor"],
    # xizmatlar
    "консультация": ["konsultatsiya"], "прием": ["qabul", "konsultatsiya"],
    "анализ": ["tahlil", "laboratoriya"], "кровь": ["qon", "tahlil"], "моча": ["siydik", "tahlil"],
    "узи": ["uzi"], "рентген": ["rentgen"], "экг": ["ekg"], "эхокг": ["exokg"],
    "фгдс": ["fgds"], "гастроскопия": ["fgds"], "массаж": ["massaj"], "прививка": ["emlash"],
    "вакцина": ["emlash"], "капельница": ["kapelnitsa"], "укол": ["emlash"],
    "гипс": ["gips"], "чистка": ["tozalash"], "удаление": ["olish"],
    "пакет": ["paket"], "чек-ап": ["check-up", "paket"], "чекап": ["check-up", "paket"],
    "госпитализация": ["statsionar"],
    # umumiy
    "цена": ["narx"], "стоимость": ["narx"], "сколько": ["narx", "qancha"], "стоит": ["narx"],
    "запись": ["yozilish", "qabul"], "записаться": ["yozilish", "qabul"],
    "работа": ["ish", "vaqt"], "работаете": ["ish", "vaqt"], "часы": ["vaqt"], "режим": ["vaqt"],
    "адрес": ["manzil"], "где": ["manzil"], "находится": ["manzil"], "телефон": ["telefon"],
    "номер": ["telefon"], "скидка": ["chegirma"], "правила": ["qoida"], "страховка": ["sug'urta"],
    "оплата": ["to'lov"], "карта": ["karta"], "очередь": ["qabul", "yozilish"],
    "сердце": ["kardiolog", "yurak"], "давление": ["kardiolog"], "голова": ["nevrolog", "bosh"],
    "боль": ["og'riq"], "спина": ["nevrolog", "bel"], "почки": ["urolog", "buyrak"],
    "желудок": ["gastroenterolog", "oshqozon"], "щитовидная": ["endokrinolog", "qalqonsimon"],
    "сахар": ["endokrinolog", "glyukoza"], "диабет": ["endokrinolog"], "беременность": ["ginekolog"],
    "горло": ["lor"], "нос": ["lor"], "ухо": ["lor"], "глаз": ["oftalmolog"],
    "кожа": ["dermatolog"], "ребенок": ["pediatr"], "дети": ["pediatr"],
}
SYNONYMS.update({stem(key): value for key, value in _RU_SYNONYMS_RAW.items()})


def synonyms_for(token: str) -> list[str]:
    """Token uchun sinonimlarni topadi (o'zbek qo'shimchalari hisobga olinadi).

    "tishim" -> "tish" -> ["stomatolog", "karies"]
    "kardiologga" -> "kardiolog" -> [...]
    """
    if token in SYNONYMS:
        return SYNONYMS[token]
    for cut in range(len(token) - 1, 3, -1):
        if token[:cut] in SYNONYMS:
            return SYNONYMS[token[:cut]]
    return []


# --------------------------------------------------------------------------
# Chunk / natija modellari
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    type: str          # doctor | service | rule | faq | clinic
    title: str
    text: str
    score: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_source(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "title": self.title, "score": round(self.score, 3)}


def _money(value: int) -> str:
    return f"{value:,}".replace(",", " ") + " so'm"


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.n = len(docs) or 1
        self.avgdl = sum(len(d) for d in docs) / self.n
        self.freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in docs:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.freqs.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        result = [0.0] * len(self.docs)
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                # Qisman mos kelish (prefiks) — o'zbek qo'shimchalari uchun foydali
                for known, known_idf in self.idf.items():
                    if len(token) >= 4 and (known.startswith(token[:5]) or token.startswith(known[:5])):
                        idf = known_idf * 0.4
                        token = known
                        break
                if idf is None:
                    continue
            for i, counts in enumerate(self.freqs):
                tf = counts.get(token, 0)
                if not tf:
                    continue
                dl = len(self.docs[i]) or 1
                result[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return result


# --------------------------------------------------------------------------
# RAG xizmati
# --------------------------------------------------------------------------
class RAGService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.clinic: dict[str, Any] = {}
        self.doctors: list[dict[str, Any]] = []
        self.services: list[dict[str, Any]] = []
        self.rules: list[dict[str, Any]] = []
        self.faqs: list[dict[str, Any]] = []
        self.chunks: list[Chunk] = []
        self._index: BM25 | None = None

        # grounding uchun "ruxsat etilgan faktlar"
        self.allowed_prices: set[int] = set()
        self.allowed_doctor_tokens: set[str] = set()
        self.allowed_service_tokens: set[str] = set()
        self.allowed_specialty_tokens: set[str] = set()
        self.allowed_terms: set[str] = set()

        self.reload()

    # ---------------- yuklash ----------------
    def _read(self, filename: str) -> dict[str, Any]:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Bilim bazasi topilmadi: {path}. Avval `python scripts/generate_data.py` ishga tushiring."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def reload(self) -> None:
        doctors_payload = self._read("doctors.json")
        services_payload = self._read("services.json")
        rules_payload = self._read("rules.json")
        faqs_payload = self._read("faqs.json")

        self.clinic = doctors_payload.get("clinic", {})
        self.doctors = doctors_payload.get("doctors", [])
        self.services = services_payload.get("services", [])
        self.rules = rules_payload.get("rules", [])
        self.faqs = faqs_payload.get("faqs", [])

        self._build_chunks()
        self._build_vocabulary()
        self._build_index()

    def _build_chunks(self) -> None:
        chunks: list[Chunk] = []

        clinic_text = (
            f"Klinika: {self.clinic.get('name')}. Manzil: {self.clinic.get('address')}. "
            f"Mo'ljal: {self.clinic.get('landmark')}. Telefon: {self.clinic.get('phone')}, "
            f"qisqa raqam {self.clinic.get('short_number')}. Email: {self.clinic.get('email')}. "
            f"Ish vaqti: {self.clinic.get('work_hours', {}).get('weekdays')}; "
            f"Yakshanba: {self.clinic.get('work_hours', {}).get('sunday')}. "
            f"Laboratoriya: {self.clinic.get('work_hours', {}).get('lab')}. "
            f"Avtoturargoh: {self.clinic.get('parking')}. "
            f"To'lov usullari: {', '.join(self.clinic.get('payment', []))}. "
            f"Sug'urta hamkorlari: {', '.join(self.clinic.get('insurance_partners', []))}. "
            f"Xizmat tillari: {', '.join(self.clinic.get('languages', []))}."
        )
        chunks.append(Chunk(id="CLINIC", type="clinic", title="Klinika umumiy ma'lumoti",
                            text=clinic_text, data=self.clinic))

        for d in self.doctors:
            schedule = d.get("schedule", {})
            children = "ha" if d.get("accepts_children") else "yo'q"
            text = (
                f"Shifokor {d['full_name']} — {d['specialty']}. {d.get('degree', '')}. "
                f"Ish tajribasi: {d.get('experience_years')} yil. Xona: {d.get('room')}. "
                f"Qabul kunlari: {', '.join(schedule.get('days', []))}, vaqt: {schedule.get('time')}. "
                f"Konsultatsiya narxi: {_money(d.get('consultation_price_uzs', 0))}. "
                f"Bolalar qabul qilinadi: {children}. "
                f"Tillar: {', '.join(d.get('languages', []))}. "
                f"Ma'lumoti: {d.get('about', '')} "
                f"Xizmatlari: {', '.join(d.get('procedures', []))}. "
                f"Ta'limi: {d.get('education', '')}."
            )
            chunks.append(Chunk(id=d["id"].upper(), type="doctor",
                                title=f"{d['full_name']} ({d['specialty']})", text=text, data=d))

        for s in self.services:
            prep = f" Tayyorgarlik: {s['preparation']}." if s.get("preparation") else ""
            note = f" Izoh: {s['note']}." if s.get("note") else ""
            text = (
                f"Xizmat: {s['name']}. Kategoriya: {s.get('category')}. "
                f"Narxi: {_money(s.get('price_uzs', 0))}. Davomiyligi: {s.get('duration_min')} daqiqa."
                f"{prep}{note}"
            )
            chunks.append(Chunk(id=s["id"].upper(), type="service",
                                title=s["name"], text=text, data=s))

        for r in self.rules:
            chunks.append(Chunk(id=r["id"].upper(), type="rule",
                                title=f"Qoida: {r['title']}",
                                text=f"Qoida {r['id']} ({r.get('category')}): {r['title']}. {r['text']}",
                                data=r))

        for f in self.faqs:
            chunks.append(Chunk(id=f["id"].upper(), type="faq",
                                title=f"FAQ: {f['question']}",
                                text=f"Savol: {f['question']} Javob: {f['answer']}",
                                data=f))

        self.chunks = chunks

    def _build_vocabulary(self) -> None:
        self.allowed_prices = {int(d.get("consultation_price_uzs", 0)) for d in self.doctors}
        self.allowed_prices |= {int(s.get("price_uzs", 0)) for s in self.services}
        self.allowed_prices.discard(0)

        for d in self.doctors:
            for token in normalize(d["full_name"]).split():
                if len(token) >= 3:
                    self.allowed_doctor_tokens.add(token)
            self.allowed_doctor_tokens.add(normalize(d["specialty"]).split()[0])

        for s in self.services:
            for token in content_tokens(s["name"]):
                if len(token) >= 4:
                    self.allowed_service_tokens.add(token)

        # Mutaxassislik nomlari ("-log" bilan tugaydigan so'zlar uchun tekshiruv):
        # model "allergolog" kabi bazada yo'q mutaxassisni o'ylab topmasligi kerak.
        for d in self.doctors:
            for token in content_tokens(d["specialty"]):
                if len(token) >= 4:
                    self.allowed_specialty_tokens.add(token)
                stemmed = stem(token)
                if len(stemmed) >= 4:
                    self.allowed_specialty_tokens.add(stemmed)

        for chunk in self.chunks:
            for token in content_tokens(chunk.text):
                self.allowed_terms.add(token)

    def _build_index(self) -> None:
        docs = [content_tokens(f"{c.title} {c.text}") for c in self.chunks]
        self._index = BM25(docs)

    # ---------------- qidiruv ----------------
    def search(self, query: str, top_k: int = 6) -> list[Chunk]:
        """BM25 + sinonim kengaytirish orqali eng mos chunklarni qaytaradi."""
        if self._index is None:
            return []

        query_tokens = content_tokens(query)
        expanded = list(query_tokens)
        for token in query_tokens:
            expanded.extend(synonyms_for(token))
        # savol turini aniqlash uchun umumiy kalitlar
        expanded.extend(self._topic_keys(query))

        scores = self._index.scores(expanded)
        for chunk, score in zip(self.chunks, scores):
            chunk.score = score

        ranked = sorted(self.chunks, key=lambda c: c.score, reverse=True)
        return [c for c in ranked[:top_k] if c.score > 0]

    @staticmethod
    def _topic_keys(query: str) -> list[str]:
        q = normalize(query)
        keys: list[str] = []
        if any(w in q for w in ("narx", "qancha", "necha", "so'm", "pul", "tolov", "to'lov", "стоит", "цена", "price")):
            keys.append("narx")
        if any(w in q for w in ("yozil", "qabul", "navbat", "bron", "запис", "записаться", "appointment", "book")):
            keys.append("yozilish")
        if any(w in q for w in ("ish vaqti", "qachon", "ochiq", "vaqt", "работа", "часы", "hours", "open")):
            keys.append("vaqt")
        if any(w in q for w in ("manzil", "qayerda", "address", "адрес", "где")):
            keys.append("manzil")
        if any(w in q for w in ("qoida", "chegirma", "sug'urta", "sugurta", "rule", "policy")):
            keys.append("qoida")
        return keys

    # ---------------- entity aniqlash ----------------
    def find_doctors(self, query: str) -> list[dict[str, Any]]:
        """Savolda tilga olingan shifokor(lar)ni topadi.

        Ism bo'yicha moslik FAQAT familiya (yoki to'liq ism) bo'yicha tekshiriladi:
        bemorning ismi shifokorning ismi bilan bir xil bo'lishi mumkin
        (masalan bemor "Bekzod", shifokor "Bekzod Yusupov") — bunda shifokor
        qabuliga yozib qo'yish xatosi bo'lmasligi kerak.
        """
        q = normalize(query)
        q_tokens = set(content_tokens(query))
        found: list[dict[str, Any]] = []

        for d in self.doctors:
            name_tokens = normalize(d["full_name"]).split()
            surname = name_tokens[-1]
            full_name = normalize(d["full_name"])

            # 1) familiya yoki to'liq ism bo'yicha
            hit = full_name in q or surname in q
            if not hit:
                hit = any(
                    SequenceMatcher(None, t, surname).ratio() >= 0.88
                    for t in q_tokens if len(t) >= 5 and abs(len(t) - len(surname)) <= 2
                )
            # 2) ixtisoslik bo'yicha
            if not hit:
                spec_tokens = [t for t in content_tokens(d["specialty"]) if len(t) >= 4]
                hit = any(t in q for t in spec_tokens)
            # 3) sinonimlar bo'yicha ("tishim og'riyapti" -> stomatolog)
            if not hit:
                specialty = d["specialty"].lower()
                for token in q_tokens:
                    if any(syn in specialty for syn in synonyms_for(token)):
                        hit = True
                        break
            if hit:
                found.append(d)
        return found

    def find_services(self, query: str) -> list[dict[str, Any]]:
        """Savolda tilga olingan xizmat(lar)ni topadi.

        O'zbek qo'shimchalari ("konsultatsiyaga", "tahlilni") sababli qat'iy
        token tengligi ishlamaydi — shuning uchun asosiy so'zlar bo'yicha
        qisman (substring) moslik ham tekshiriladi.
        """
        q = normalize(query)
        q_tokens = set(content_tokens(query))
        found: list[dict[str, Any]] = []
        for s in self.services:
            name_tokens = content_tokens(s["name"])
            strong_words = [t for t in name_tokens if len(t) >= 5]
            if any(word in q for word in strong_words):
                found.append(s)
                continue
            overlap = q_tokens & set(name_tokens)
            if sum(1 for t in overlap if len(t) >= 4) >= 2:
                found.append(s)
        return found

    def find_services_by_category(self, category: str) -> list[dict[str, Any]]:
        return [s for s in self.services if s.get("category", "").lower() == category.lower()]

    def find_and_chunk(self, query: str, top_k: int = 4) -> list[Chunk]:
        """Savolda tilga olingan shifokor/xizmatlarga tegishli chunklarni qaytaradi.

        Tibbiy maslahat rad etilganda "mos mutaxassis" ma'lumotini ko'rsatish uchun.
        Moslik topilmasa oddiy BM25 qidiruviga qaytadi.
        """
        target_ids = {d["id"].upper() for d in self.find_doctors(query)}
        if not target_ids:
            target_ids = {s["id"].upper() for s in self.find_services(query)}
        if not target_ids:
            return self.search(query, top_k=top_k)

        matched = [c for c in self.chunks if c.id in target_ids]
        for chunk in matched:
            chunk.score = 1.0
        return matched[:top_k]

    # ---------------- grounding ----------------
    def known_price(self, value: int) -> bool:
        return value in self.allowed_prices

    def is_known_doctor_token(self, token: str) -> bool:
        token = normalize(token)
        if token in self.allowed_doctor_tokens:
            return True
        return any(SequenceMatcher(None, token, known).ratio() >= 0.9
                   for known in self.allowed_doctor_tokens)

    def is_known_specialty(self, token: str) -> bool:
        """Mutaxassislik nomi bilim bazasida bormi ("-log" so'zlari uchun)."""
        token = normalize(token)
        for candidate in (token, stem(token)):
            if candidate in self.allowed_specialty_tokens:
                return True
        return any(SequenceMatcher(None, token, known).ratio() >= 0.9
                   for known in self.allowed_specialty_tokens)

    def stats(self) -> dict[str, int]:
        return {
            "doctors": len(self.doctors),
            "services": len(self.services),
            "rules": len(self.rules),
            "faqs": len(self.faqs),
            "chunks": len(self.chunks),
            "prices": len(self.allowed_prices),
        }
