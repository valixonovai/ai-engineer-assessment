"""FastAPI ilovasi: hospital-chatbot.

Ishga tushirish:
    uvicorn app.main:app --reload --port 8000

Manzillar:
    /            — oddiy web chat interfeysi
    /docs        — Swagger UI (API hujjati va sinov)
    /redoc       — ReDoc
    /health      — ilova va LLM holati
    /api         — xizmat meta-ma'lumotlari
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat as chat_api
from app.config import BASE_DIR, settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("hospital-chatbot")

STATIC_DIR = BASE_DIR / "app" / "static"

TAGS_METADATA = [
    {
        "name": "chat",
        "description": (
            "Asosiy suhbat endpointi. Javob bilan birga `sources` (ishlatilgan bilim bazasi "
            "bo'laklari), `safety` (qaysi himoya qatlami ishga tushgani) va `booking` "
            "(qabulga yozilish holati) qaytariladi."
        ),
    },
    {
        "name": "meta",
        "description": "Xizmat holati va bilim bazasi statistikasi.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    stats = chat_api.rag.stats()
    logger.info("Bilim bazasi yuklandi: %s", stats)
    logger.info("LLM: %s | model: %s", settings.LLM_PROVIDER, settings.LLM_MODEL)
    if not settings.llm_configured:
        logger.error(
            "OPENAI_API_KEY bo'sh — chatbot bilim bazasidan deterministik javob beradi. "
            "Kalitni .env fayliga qo'ying va serverni qayta ishga tushiring.")
    elif settings.LLM_PROVIDER == "ollama":
        available = await chat_api.llm.list_local_models()
        if settings.LLM_MODEL not in available:
            logger.warning(
                "Ollama modeli '%s' topilmadi. Mavjud modellar: %s. "
                "`ollama pull %s` buyrug'ini bajaring yoki .env dagi LLM_MODEL ni o'zgartiring.",
                settings.LLM_MODEL, available or "yo'q", settings.LLM_MODEL)
    yield
    await chat_api.llm.close()


app = FastAPI(
    title="Shifo Med — Hospital Chatbot API",
    description=(
        "Klinika uchun RAG asosidagi AI chatbot.\n\n"
        "**Qat'iy cheklovlar (kod darajasida ta'minlangan):**\n"
        "- bilim bazasida yo'q narx, shifokor yoki xizmat o'ylab topilmaydi;\n"
        "- tibbiy tashxis qo'yilmaydi, dori yoki doza tavsiya qilinmaydi;\n"
        "- hayot uchun xavfli holatda 103 raqamiga yo'naltiriladi;\n"
        "- klinika qoidalarini foydalanuvchi xabari bilan o'zgartirib bo'lmaydi.\n\n"
        "**Web interfeys:** [`/`](/) &nbsp;·&nbsp; **ReDoc:** [`/redoc`](/redoc)"
    ),
    version=__version__,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_api.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    """Oddiy web chat interfeysi (bitta HTML fayl, build talab qilmaydi)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api", tags=["meta"], summary="Xizmat haqida qisqa ma'lumot")
async def api_root() -> dict:
    return {
        "service": "Shifo Med — Hospital Chatbot API",
        "version": __version__,
        "docs": "/docs",
        "web_ui": "/",
        "endpoints": ["POST /chat", "POST /chat/reset", "GET /health", "GET /info"],
    }


@app.get("/health", tags=["meta"], summary="Ilova va LLM holati")
async def health() -> dict:
    llm_state = await chat_api.llm.health()
    return {
        "status": "ok" if llm_state["available"] else "degraded",
        "knowledge_base": chat_api.rag.stats(),
        "llm": llm_state,
        "active_sessions": chat_api.memory.count(),
    }


@app.get("/info", tags=["meta"], summary="Bilim bazasi statistikasi va konfiguratsiya")
async def info() -> dict:
    return {
        "clinic": chat_api.rag.clinic.get("name"),
        "knowledge_base": chat_api.rag.stats(),
        "llm": {
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL,
            "base_url": (settings.OPENAI_BASE_URL if settings.is_openai_compatible
                         else settings.OLLAMA_BASE_URL),
            "configured": settings.llm_configured,
        },
        "safety": {
            "no_hallucinated_facts": True,
            "no_diagnosis": True,
            "no_medication_or_dosage": True,
            "emergency_redirect": "103",
            "rules_immutable": True,
            "output_grounding_check": True,
        },
    }
