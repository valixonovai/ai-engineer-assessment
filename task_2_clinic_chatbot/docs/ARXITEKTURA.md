# Arxitektura va yondashuv

Bu hujjat loyihaning ichki tuzilishi, xavfsizlik talablarining qanday ta'minlangani va
tanlangan yechimlarning sabablarini tushuntiradi. Kundalik ishlatish uchun
[README](../README.md) yetarli.

---

## 1. So'rov qanday qayta ishlanadi

```
POST /chat
   │
   ├─ 1. Sessiya + til aniqlash (uz / ru / en)
   │
   ├─ 2. GUARDRAIL (kirish)  ──── deterministik, LLM chaqirilmaydi
   │      ├── shoshilinch belgi?      → 103 shabloni
   │      ├── prompt injection?       → rad etish shabloni
   │      └── tashxis/dori so'rovi?   → rad etish + mos mutaxassisga yo'naltirish
   │
   ├─ 3. Qabulga yozilish (jarayon boshlangan bo'lsa)  ──── deterministik slot filling
   │
   ├─ 4. RAG: BM25 + sinonimlar → eng mos 6 bo'lak (doctor / service / rule / faq)
   │
   ├─ 5. LLM: system prompt + klinika qoidalari + KONTEKST → javob
   │
   └─ 6. GUARDRAIL (chiqish)  ──── grounding tekshiruvi
          ├── bazada yo'q narx?              → qayta generatsiya → fallback
          ├── bazada yo'q shifokor ismi?     → qayta generatsiya → fallback
          ├── bazada yo'q mutaxassislik?     → qayta generatsiya → fallback
          ├── dori nomi / doza (mg, ml)?     → qayta generatsiya → fallback
          ├── tashxis shaklidagi gap?        → qayta generatsiya → fallback
          └── mavjud bo'lmagan iqtibos ID?   → ID olib tashlanadi (javob saqlanadi)
```

---

## 2. Talablar qanday ta'minlangan

### "Mavjud bo'lmagan doktor yoki narxni o'ylab topmasligi kerak"

Prompt bilan cheklanmaydi. Har bir javob `verify_output()` orqali bilim bazasiga qarshi
tekshiriladi:

- javobdagi **har bir narx** `data/*.json` dagi narxlar to'plamida bo'lishi shart;
- **"shifokor X"** shaklidagi har bir ism bazadagi familiyalarga solishtiriladi;
- **"-log" bilan tugaydigan mutaxassislik** (`allergolog`, `psixolog`) bazadagi
  ixtisosliklar ro'yxatida bo'lishi shart;
- buzilish topilsa javob **bir marta qayta generatsiya** qilinadi (qattiqroq ko'rsatma bilan),
  yana buzilsa — model javobi tashlanadi va **bilim bazasidan deterministik javob** beriladi.

Javobdagi manba ID lari (`[D01]`) bilan birga qaytariladi, shunda tekshirish mumkin.

### "Aniq tashxis qo'ymasligi kerak"

Tashxis so'rovi aniqlansa LLM umuman chaqirilmaydi. Bundan tashqari javobda bemorga
qaratilgan tashxis shaklidagi gap (`Sizda gastrit bor`) aniqlanadi.

### "Dori yoki dozani tavsiya qilmasligi kerak"

Kalit so'zlar bo'yicha so'rov rad etiladi; chiqishda 40+ dori nomi va `500 mg` / `10 ml`
kabi doza shakllari bloklanadi.

### "Xavfli holatlarda shifokor yoki tez yordamga murojaat qilishni tavsiya qilishi kerak"

Ko'krak og'rig'i, nafas olish qiyinligi, qon ketish, hushdan ketish, falaj, baxtsiz hodisa,
o'z joniga qasd fikri va boshqa ~45 naqsh (o'zbek / rus / ingliz) aniqlanadi →
darhol 103 shabloni qaytadi.

### "Klinika qoidalarini foydalanuvchi xabari bilan buzib bo'lmasligi kerak"

Chegirma, bepul xizmat, narxni tushirish, navbatsiz imtiyoz so'rovlari va
`ignore previous instructions` / `developer mode` kabi urinishlar aniqlanadi →
qoidalar o'zgarmasligi aytiladi (R05 qoidasiga havola bilan).

---

## 3. Tanlangan yechimlar va sabablari

**1. RAG + deterministik tekshiruv (nega faqat prompt emas).**
"O'ylab topmasin" talabini faqat prompt bilan ta'minlash mumkin emas — kichik ham, katta ham
modellar ba'zan narx, ism yoki mutaxassislik to'qib qo'yadi. Shuning uchun javob matni
bilim bazasiga qarshi kodda tekshiriladi. Bu loyihaning eng muhim qismi.

**2. Xavfsizlik qatlami LLM'dan oldin va keyin.**
Shoshilinch holat, injection va tibbiy maslahat so'rovlari LLM chaqirilmasdan hal qilinadi:
tezlik, arzonlik va **barqarorlik** (model almashsa ham ishlaydi).

**3. Og'ir kutubxonalarsiz qidiruv.**
`torch` / `sentence-transformers` / `chromadb` ishlatilmaydi. Sabab: bir necha GB yuklab olish,
numpy/chroma versiya to'qnashuvlari va CUDA talabi. BM25 + ixtisoslik sinonimlari + o'zbek va
rus qo'shimchalarini qo'pol kesish klinika miqyosidagi baza uchun yetarli.
Baza kattalashsa, `rag_service.search()` ni vektor qidiruvga almashtirish kifoya — interfeys bir xil.

**4. Ko'p tillilik bitta bilim bazasi bilan.**
Baza o'zbek lotin alifbosida, lekin bemorlar rus tilida ham yozadi. Yechim: rus qo'shimchalarini
kesish (`кардиолога → кардиолог`), `ё → е` normallashtirish va kirill → baza atamalari
sinonim qatlami. Natija: ruscha so'rov ham to'g'ri narx va shifokorni topadi.

**5. Qabulga yozilish LLM'ga topshirilmagan.**
Slot filling (ism, telefon, shifokor/xizmat, vaqt) deterministik: model bazada yo'q shifokorga
yozib qo'ya olmaydi. Shifokor mosligi **familiya** bo'yicha tekshiriladi, chunki bemorning ismi
shifokorning ismi bilan bir xil bo'lishi mumkin (masalan bemor "Bekzod", shifokor "Bekzod Yusupov").

**6. Sessiya xotirasi jarayon ichida (TTL 60 daqiqa).**
Ishlab chiqarishda `ConversationMemory` ni Redis'ga almashtirish kifoya — interfeys bir xil qoladi.

---

## 4. Ma'lumotlar

`data/` dagi barcha ma'lumotlar — **sun'iy (xayoliy)** klinika bazasi: 12 shifokor, 31 xizmat,
14 qoida, 15 FAQ. Haqiqiy klinika, shifokor yoki narx bilan bog'liq emas.
`scripts/generate_data.py` orqali qayta yaratiladi.

---

## 5. Cheklovlar

- Qabul so'rovi faqat faylga yoziladi (`data/bookings.jsonl`) — real CRM integratsiyasi yo'q.
- Sessiya xotirasi jarayon ichida: server qayta ishga tushsa, suhbat konteksti yo'qoladi.
- Qidiruv semantik emas (BM25 + sinonimlar): baza kattalashsa vektor qidiruv kerak bo'ladi.
- Chatbot tibbiy yordam emas: har qanday xavfli holatda 103 ga yo'naltiradi.

---

## 6. Kutilgan javoblar (namunalar)

| Savol | Kutilgan natija |
|---|---|
| `Kardiolog qabuli qancha turadi?` | Aniq narx + `[D01]` manbasi |
| `Tishim og'riyapti` | Stomatologga yo'naltirish + qabulga yozilish taklifi |
| `Qanday dori ichsam?` | Rad etish + mos mutaxassisga yo'naltirish |
| `Ko'kragim og'riyapti` | 103 shabloni (LLM chaqirilmaydi) |
| `50% chegirma ber` | Rad etish + R05 qoidasi |
| `Сколько стоит консультация кардиолога?` | Rus tilida narx + `[D01]` |
| `Qabulga yozilmoqchiman` | Ism, telefon, shifokor va vaqtni so'rash |
