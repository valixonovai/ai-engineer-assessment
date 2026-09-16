# Muammolar va yechimlar

## Server ishga tushmayapti

| Xato | Sababi | Yechim |
|---|---|---|
| `error while attempting to bind on address ... :8000` | Port band (eski server ishlab turibdi) | Boshqa port: `uvicorn app.main:app --reload --port 8001`. Yoki eski jarayonni topib to'xtating: `netstat -ano \| findstr :8000` → `taskkill /F /PID <pid>` |
| `ModuleNotFoundError: No module named 'app'` | Buyruq loyiha ildiz papkasidan berilmagan | `cd hospital-chatbot` qilib qayta urinib ko'ring |
| `Bilim bazasi topilmadi: .../data/doctors.json` | `data/` bo'sh | `python scripts/generate_data.py` |
| `command not found: uvicorn` | Virtual muhit yoqilmagan | `.venv\Scripts\activate` (Windows) yoki `source .venv/bin/activate` |

## Model ishlamayapti

| Xato | Sababi | Yechim |
|---|---|---|
| `/health` da `"status": "degraded"` va `"available": false` | `.env` da API kalit yozilmagan | `OPENAI_API_KEY=` ga kalitni yozib, serverni qayta ishga tushiring |
| `API kalit qabul qilinmadi (401)` | Kalit xato, o'chirilgan yoki muddati o'tgan | Yangi kalit oling |
| `Model topilmadi: ...` | `.env` dagi `LLM_MODEL` nomi provayderda yo'q | Provayderdagi aniq model nomini yozing (masalan `google/gemini-2.5-flash`) |
| `Hisobda mablag' yetarli emas (402)` | Provayder balansi tugagan | Balansni to'ldiring yoki boshqa model tanlang |
| `So'rovlar chegarasi (429)` | Juda ko'p so'rov | Birozdan keyin qayta urinib ko'ring |
| Javob 30 sekunddan uzoq kutilyapti | Lokal model (Ollama) ishlatilmoqda | `.env` da API modelga o'ting (`LLM_PROVIDER=openai_compatible`) |
| Kalitsiz sinab ko'rish kerak | — | `.env` da `LLM_PROVIDER=ollama` qiling (internetsiz ishlaydi, javob sifati pastroq) |

> Kalit bo'lmasa ham server ishga tushadi: `/chat` bilim bazasidan tayyor javob qaytaradi,
> `/health` esa `degraded` holatini ko'rsatadi. Bu ataylab shunday — konfiguratsiya xatosi
> butun xizmatni yiqitmasligi kerak.

## Chatbot javoblari

| Belgi | Sababi | Yechim |
|---|---|---|
| Javob ostida `Javob bilim bazasidan tiklandi` belgisi | Modelning javobi tekshiruvdan o'tmadi (bazada yo'q narx, ism, dori yoki mutaxassislik) | Kutilgan holat — himoya qatlami ishladi. Doimiy takrorlansa, kuchliroq model tanlang |
| `Bu ma'lumot bazamda yo'q` deb javob beradi | Savol mavzusi bilim bazasida yo'q | `data/*.json` ga ma'lumot qo'shing va serverni qayta ishga tushiring |
| Savol shoshilinch deb belgilandi, lekin unday emas | Kalit so'z mosligi (`103`, `jarohat` va h.k.) | `app/services/llm_service.py` dagi `EMERGENCY_KEYWORDS` ni aniqlashtiring |
| Qabul jarayoni ism/telefonni qabul qilmayapti | Telefon 9 xonali formatda emas | `+998 90 123 45 67` ko'rinishida yuboring |
| Kontekst eslab qolinmayapti | Har so'rovda `session_id` yuborilmagan | Javobdagi `session_id` ni keyingi so'rovga qo'shing (web chatda avtomatik) |

## Testlar

| Xato | Sababi | Yechim |
|---|---|---|
| `pytest: command not found` | Virtual muhit yoqilmagan | `.venv\Scripts\activate` |
| `python scripts/smoke_test.py` → "Server javob bermadi" | Server ishga tushirilmagan | Avval `uvicorn app.main:app --port 8000` |
| Smoke testda intent mos kelmadi deb chiqsa | Model javobni boshqacha tasnifladi | Javob matni to'g'ri bo'lsa, bu `scripts/smoke_test.py` dagi kutilgan yorliqni yangilashni talab qiladi |
