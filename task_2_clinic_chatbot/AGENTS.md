# AGENTS.md — bu loyihada ishlash qoidalari

> Bu fayl AI agentlar (Hermes, Claude Code, Codex va boshqalar) uchun.
> Maqsad: arxitekturani adashtirib buzib qo'ymaslik.

## Loyiha nima qiladi

Klinika ("Shifo Med") uchun RAG asosidagi AI chatbot. Bemorlar shifokorlar,
xizmatlar, narxlar, ish vaqti va qoidalar haqida so'raydi, chatbot esa **faqat
`data/` papkasidagi bilim bazasidan** javob beradi va qabulga yozilish uchun
ma'lumot yig'adi.

## Papka tuzilishi va har bir faylning vazifasi

```
data/                      # Yagona haqiqat manbai (single source of truth)
├── doctors.json           # 12 shifokor: ixtisoslik, ish vaqti, narx, xona
├── services.json          # 31 xizmat va narxlar
├── rules.json             # 14 klinika qoidasi (R01–R14)
└── faqs.json              # 15 FAQ + klinika umumiy ma'lumoti

app/
├── main.py                # FastAPI ilova, routerlar, /health, /info
├── config.py              # .env orqali sozlamalar (pydantic-settings)
├── api/chat.py            # POST /chat: orkestratsiya + qabulga yozilish
├── static/index.html      # Web chat interfeysi (bitta fayl, build yo'q)
├── services/
│   ├── rag_service.py     # Qidiruv: JSON -> chunk -> BM25 + entity matching
│   └── llm_service.py     # System prompt, xavfsizlik qoidalari, grounding,
│                          # LLM provayderlari, suhbat xotirasi
└── models/schemas.py      # Pydantic request/response modellari

scripts/
├── generate_data.py       # data/*.json ni qayta generatsiya qiladi
└── smoke_test.py          # Ishlayotgan serverga qarshi E2E tekshiruv

docs/
├── ARXITEKTURA.md         # Arxitektura, xavfsizlik talablari, yondashuv sabablari
├── MUAMMOLAR.md           # Xatolar va yechimlar jadvali
└── screenshots/           # README uchun namunalar (chat, booking_flow_1, booking_flow_2)

tests/test_chat.py         # 67 test (LLM'siz, monkeypatch bilan)
```

## O'ZGARTIRMASLIK KERAK BO'LGAN QOIDALAR

1. **`data/` — yagona manba.** Bilim bazasidagi faktlar (narx, ism, kun, vaqt)
   faqat shu papkadan keladi. Kodda yoki promptda hech qanday narx/ism
   qattiq yozilmasin.

2. **LLM provayder — faqat `.env` orqali.** Kodda model nomi yoki `api_key`
   qattiq yozilmaydi. Asosiy yo'l — `LLM_PROVIDER=openai_compatible`
   (OpenRouter/OpenAI/DeepSeek). Lokal Ollama — ixtiyoriy zaxira yo'l, standart emas.
   Kalit `.env` da saqlanadi va hech qachon repoga yoki chatga chiqmaydi.

2. **Xavfsizlik LLM ixtiyorida emas.** Quyidagilar kod darajasida,
   deterministik tarzda amalga oshirilgan — ularni "prompt orqali" yechimga
   almashtirmang:
   - shoshilinch holat aniqlash (`EMERGENCY_KEYWORDS`) → LLM chaqirilmaydi,
     tayyor shablon qaytadi;
   - tashxis/dori so'rovi (`DIAGNOSIS_REQUEST`, `MEDICATION_REQUEST`) → rad etish;
   - prompt injection / qoidani buzish urinishi (`INJECTION_ATTEMPTS`);
   - **chiqishni tekshirish** (`verify_output`): javobdagi har bir narx
     `rag.allowed_prices` da, har bir shifokor familiyasi bazada bo'lishi shart;
     dori nomi/doza (mg, ml) va tashxis shaklidagi gap taqiqlanadi.
   Buzilish topilsa: 1 marta qayta generatsiya → yana buzilsa bilim bazasidan
   deterministik javob (`build_fallback_reply`).

3. **Grounding tekshiruvi o'chirilmasin.** `verify_output` — loyihaning eng
   muhim qismi. Uni "model yaxshi ishlayapti" deb o'chirib qo'yish mumkin emas.

4. **Qabulga yozilish deterministik.** `app/api/chat.py` dagi slot filling
   (ism, telefon, shifokor/xizmat, vaqt) LLM'ga topshirilmaydi — aks holda
   bazada yo'q shifokorga yozib qo'yish xavfi paydo bo'ladi.
   Shifokor mosligi **familiya** bo'yicha (`find_doctors`): bemorning ismi
   shifokorning ismi bilan bir xil bo'lishi mumkin.

5. **Til qo'llab-quvvatlanadi:** o'zbek (lotin), rus, ingliz.
   Yangi javob shabloni qo'shsangiz, uchtasini ham to'ldiring.

6. **Og'ir kutubxonasiz qolish.** `torch`, `sentence-transformers`, `chromadb`
   qo'shmang: qidiruv BM25 + sinonimlar bilan ishlaydi. Sabab: o'rnatish
   og'irligi va numpy/chroma versiya to'qnashuvlari.

## Tekshirish tartibi (o'zgarishdan keyin majburiy)

```bash
python -m pytest tests/ -q          # 64 test o'tishi shart
uvicorn app.main:app --port 8000    # alohida terminalda
python scripts/smoke_test.py        # haqiqiy LLM bilan 12 senariy + kontekst
```

Testlar LLM'ga bog'lanmaydi (monkeypatch), shuning uchun kalitsiz ham ishlaydi.
`smoke_test.py` esa haqiqiy modelni tekshiradi — API kalit kerak.

## Ma'lumotlar

Barcha ma'lumotlar **sun'iy** (xayoliy klinika). Haqiqiy narx, shifokor yoki
klinika bilan bog'liq emas — bu ataylab test/demo maqsadida shunday.
