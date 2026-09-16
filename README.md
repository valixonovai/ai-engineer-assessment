# AI Engineer Assessment

Ushbu repoda ikkita mustaqil vazifa bajarilgan: **LLM fine-tuning** va **RAG asosidagi
AI chatbot**. Har bir vazifa alohida papkada, o'z README si, kodi va hujjatlari bilan.

> Muallif: valixonovai · Har bir vazifa mustaqil ishga tushiriladi va tekshiriladi.

---

## Tuzilma

```
.
├── task_1_finetuning/          # Vazifa 1: kiberxavfsizlik LLM (QLoRA/LoRA)
│   ├── README.md               # batafsil hujjat
│   ├── dataset/                # ChatML formatidagi SFT dataset
│   ├── notebooks/              # training notebook (Unsloth, 4-bit)
│   └── scripts/                # dataset yig'ish skriptlari
│
└── task_2_clinic_chatbot/      # Vazifa 2: klinika uchun RAG chatbot
    ├── README.md               # batafsil hujjat (o'rnatish, ishlatish)
    ├── app/                    # FastAPI ilovasi + web chat interfeysi
    ├── data/                   # bilim bazasi (JSON)
    ├── docs/                   # arxitektura, muammolar, skrinshotlar
    ├── scripts/                # ma'lumot generatori, E2E test
    └── tests/                  # 67 test
```

---

## Vazifa 1 — Kiberxavfsizlik LLM Fine-Tuning

Kichik open-source model (**Qwen-2.5-7B-Base**) kiberxavfsizlik yo'nalishida
**QLoRA (4-bit)** usulida fine-tune qilingan. Dataset RAG va Agent/Tool Use
formatida (ChatML) tayyorlangan: `<thought>` va `<call:tool_name>` teglari bilan.

| Bo'lim | Fayl |
|---|---|
| Dataset | `task_1_finetuning/dataset/kiber_agent_dataset.json` |
| Training | `task_1_finetuning/notebooks/kiber-sft-train-kaggle.ipynb` |
| Skriptlar | `task_1_finetuning/scripts/` (eski versiyalar `scripts/legacy/`) |
| Model | Hugging Face: `valixonov04/qwen-7b-kiberagent-full` |

Batafsil: **[task_1_finetuning/README.md](task_1_finetuning/README.md)**

---

## Vazifa 2 — Klinika uchun AI Chatbot (RAG)

"Shifo Med" klinikasi uchun chatbot: shifokorlar, xizmatlar, narxlar va qoidalar
haqida javob beradi, mos mutaxassisga yo'naltiradi va qabulga yozilish uchun
ma'lumot yig'adi. Javoblar faqat bilim bazasidan olinadi — model o'ylab topgan
narx, shifokor, dori yoki mutaxassislik **kod darajasida** ushlanadi.

**Stack:** Python · FastAPI · RAG (BM25 + sinonimlar) · LLM API (OpenRouter / OpenAI / DeepSeek / Ollama)

| Bo'lim | Joyi |
|---|---|
| API endpoint | `POST /chat` — `task_2_clinic_chatbot/app/api/chat.py` |
| Web chat interfeysi | `app/static/index.html` |
| Bilim bazasi | `data/` — 12 shifokor, 31 xizmat, 14 qoida, 15 FAQ |
| Xavfsizlik qatlami | `app/services/llm_service.py` |
| Hujjatlar | `docs/ARXITEKTURA.md`, `docs/MUAMMOLAR.md` |

**Tez ishga tushirish:**

```bash
cd task_2_clinic_chatbot
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                              # va .env ga API kalitni yozing
uvicorn app.main:app --reload --port 8000
```

Brauzer: `http://127.0.0.1:8000/` (chat) · `http://127.0.0.1:8000/docs` (Swagger)

Testlar: `python -m pytest tests/ -q` → 67 test

Batafsil: **[task_2_clinic_chatbot/README.md](task_2_clinic_chatbot/README.md)**

---

## Xavfsizlik va konfiguratsiya

- API kalit faqat `.env` faylida saqlanadi (`.gitignore` da) — repoda kalit yo'q.
- Har bir vazifa o'z `.env.example` fayliga ega: kalitni o'sha namunadan ko'chirib to'ldirasiz.
- `task_2_clinic_chatbot` da model nomi ham `.env` orqali almashtiriladi.
