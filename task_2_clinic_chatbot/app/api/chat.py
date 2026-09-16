"""POST /chat — asosiy endpoint.

Oqim (har bir xabar uchun):
    1. Sessiyani olish/yaratish, tilni aniqlash, xabarni tarixga yozish
    2. Guardrail: shoshilinch holat / prompt injection / tibbiy maslahat so'rovi
    3. Qabulga yozilish jarayoni (agar boshlangan yoki niyat aniqlansa)
    4. RAG: bilim bazasidan kontekst yig'ish
    5. LLM javob beradi -> grounding tekshiruvi -> buzilsa qayta urinish -> fallback
    6. Javobni qaytarish (manbalar, xavfsizlik bayroqlari, booking holati bilan)
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from app.config import settings
from app.models.schemas import (
    BookingDetails,
    ChatRequest,
    ChatResponse,
    SafetyFlags,
    SessionResetResponse,
    SourceChunk,
)
from app.services.llm_service import (
    EMERGENCY_REPLY,
    INJECTION_REPLY,
    MEDICAL_REFUSAL_REPLY,
    ConversationMemory,
    LLMError,
    LLMService,
    build_fallback_reply,
    build_strict_retry_prompt,
    build_system_prompt,
    check_input,
    detect_language,
    sanitize_citations,
    verify_output,
)
from app.services.rag_service import RAGService, normalize

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# --------------------------------------------------------------------------
# Singletonlar
# --------------------------------------------------------------------------
rag = RAGService(settings.DATA_DIR)
llm = LLMService()
memory = ConversationMemory(settings.SESSION_TTL_MINUTES, settings.MAX_HISTORY_MESSAGES)

# --------------------------------------------------------------------------
# Qabulga yozilish: deterministik slot filling (LLM o'ylab topmasligi uchun)
# --------------------------------------------------------------------------
BOOKING_INTENT = re.compile(
    r"(yozil|yozmoqchi|qabul\w*\s*(ga|iga)?\s*(yoz|bormoqchi|kerak|olmoqchi)|bron|"
    r"navbat\w*\s*(ol|kerak)|appointment|book\s+(a|an)\s+appointment|запис|записаться)",
    re.IGNORECASE | re.UNICODE)

PHONE_PATTERN = re.compile(r"(?:\+?998)?[\s\-()]*(\d{2})[\s\-()]*(\d{3})[\s\-()]*(\d{2})[\s\-()]*(\d{2})")

DAY_WORDS = ["dushanba", "seshanba", "chorshanba", "payshanba", "juma", "shanba", "yakshanba",
             "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
             "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
             "bugun", "ertaga", "today", "tomorrow", "сегодня", "завтра"]
TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3])[:.](\d{2})\b|\b(ertalab|tushdan keyin|kechqurun|"
                          r"утром|вечером|morning|afternoon|evening)\b", re.IGNORECASE)

CONFIRM_WORDS = ("ha", "ha,", "xa", "tasdiq", "to'g'ri", "togri", "ok", "mayli", "bo'ladi",
                 "да", "верно", "подтверждаю", "yes", "confirm", "correct")
CANCEL_WORDS = ("yo'q", "yoq", "bekor", "kerak emas", "нет", "отмена", "отменить", "no", "cancel")

BOOKING_TEMPLATES: dict[str, dict[str, str]] = {
    "ask_all": {
        "uz": "Qabulga yozilish uchun bir necha ma'lumot kerak.\n\n1) To'liq ismingiz?\n"
              "2) Telefon raqamingiz?\n3) Qaysi shifokor yoki xizmat kerak?\n4) Qulay kun va vaqt?\n\n"
              "Barchasini bir xabarda yozsangiz ham bo'ladi.",
        "ru": "Для записи на приём нужны несколько данных.\n\n1) Ваше полное имя?\n"
              "2) Номер телефона?\n3) Какой врач или услуга нужны?\n4) Удобный день и время?\n\n"
              "Можно отправить всё одним сообщением.",
        "en": "To book an appointment I need a few details.\n\n1) Your full name?\n"
              "2) Your phone number?\n3) Which doctor or service?\n4) Preferred day and time?\n\n"
              "You can send everything in one message.",
    },
    "ask_name": {"uz": "To'liq ismingizni ayting.", "ru": "Назовите ваше полное имя.",
                 "en": "Please tell me your full name."},
    "ask_phone": {"uz": "Telefon raqamingizni yozing (masalan: +998 90 123 45 67).",
                  "ru": "Напишите ваш номер телефона (например: +998 90 123 45 67).",
                  "en": "Please provide your phone number (e.g. +998 90 123 45 67)."},
    "ask_doctor": {"uz": "Qaysi shifokorga yoki qaysi xizmatga yozilmoqchisiz?",
                   "ru": "К какому врачу или на какую услугу хотите записаться?",
                   "en": "Which doctor or service do you need?"},
    "ask_time": {"uz": "Qaysi kun va vaqt sizga qulay?",
                 "ru": "Какой день и время вам удобны?",
                 "en": "Which day and time suits you?"},
    "ask_confirm": {
        "uz": "Ma'lumotlarni tekshiring:\n\n- Ism: {full_name}\n- Telefon: {phone}\n"
              "- Shifokor/xizmat: {target}\n- Vaqt: {preferred_time}\n\nTasdiqlaysizmi? (ha / yo'q)",
        "ru": "Проверьте данные:\n\n- Имя: {full_name}\n- Телефон: {phone}\n"
              "- Врач/услуга: {target}\n- Время: {preferred_time}\n\nПодтверждаете? (да / нет)",
        "en": "Please check the details:\n\n- Name: {full_name}\n- Phone: {phone}\n"
              "- Doctor/service: {target}\n- Time: {preferred_time}\n\nConfirm? (yes / no)",
    },
    "confirmed": {
        "uz": "So'rovingiz qabul qilindi.\n\n- Yozuv raqami: {booking_id}\n- Ism: {full_name}\n"
              "- Telefon: {phone}\n- Shifokor/xizmat: {target}\n- Vaqt: {preferred_time}\n\n"
              "Administrator tasdiqlash uchun sizga qo'ng'iroq qiladi (+998 71 200 45 45). "
              "Agar vaqtni o'zgartirmoqchi bo'lsangiz yoki qabulga kelolmasangiz, "
              "iltimos kamida 3 soat oldin xabar bering (R03 qoidasi).",
        "ru": "Заявка принята.\n\n- Номер записи: {booking_id}\n- Имя: {full_name}\n"
              "- Телефон: {phone}\n- Врач/услуга: {target}\n- Время: {preferred_time}\n\n"
              "Администратор позвонит для подтверждения (+998 71 200 45 45). "
              "Если нужно изменить время или вы не сможете прийти, "
              "сообщите минимум за 3 часа (правило R03).",
        "en": "Your request has been received.\n\n- Booking ID: {booking_id}\n- Name: {full_name}\n"
              "- Phone: {phone}\n- Doctor/service: {target}\n- Time: {preferred_time}\n\n"
              "The administrator will call to confirm (+998 71 200 45 45). "
              "To reschedule or cancel, please notify at least 3 hours in advance (rule R03).",
    },
    "cancelled": {
        "uz": "Yozuv bekor qilindi. Boshqa savolingiz bo'lsa, yordam berishga tayyorman.",
        "ru": "Запись отменена. Если есть другие вопросы, готов помочь.",
        "en": "The booking has been cancelled. Happy to help with anything else.",
    },
}


NAME_NOISE = {
    "ismim", "ismim:", "mening", "meni", "ism", "telefon", "telefonim", "raqam", "raqamim",
    "nomer", "номер", "телефон", "имя", "phone", "name", "full", "my", "is", "va", "и",
    "qabul", "qabulga", "yozilmoqchiman", "yozilaman", "yozilish", "navbat", "bron",
    "kerak", "kerakmi", "iltimos", "salom", "assalomu", "alaykum", "marhamat",
}
DAY_TIME_WORDS = {d for d in DAY_WORDS} | {"ertalab", "kechqurun", "tushdan", "keyin", "kuni", "soat"}


def _looks_like_name(text: str) -> bool:
    """Matn ism-familiyaga o'xshaydimi (savol emas, raqam yo'q)?"""
    cleaned = text.strip().strip(".,!")
    if not cleaned or "?" in text or any(ch.isdigit() for ch in cleaned):
        return False
    words = cleaned.split()
    if not 1 <= len(words) <= 4:
        return False
    return all(len(w) >= 3 and w.replace("'", "").isalpha() for w in words)


def _extract_name(text: str) -> str | None:
    """Aralash xabardan ism-familiyani ajratib oladi.

    Masalan: "Ismim Bekzod Aliyev, telefon +998 90 123 45 67" -> "Bekzod Aliyev"
    """
    cleaned = PHONE_PATTERN.sub(" ", text)
    cleaned = re.sub(
        r"(?i)\b(mening\s+ismim|ismim|ism|raqamim|raqam|telefonim|telefon|nomer|"
        r"имя|телефон|номер|phone|name|full\s+name|my\s+name\s+is)\b", " ", cleaned)
    cleaned = re.sub(r"[,\-:;.!?]", " ", cleaned)
    words = [w for w in cleaned.split()
             if w and len(w) >= 3 and w.lower() not in NAME_NOISE
             and w.lower() not in DAY_TIME_WORDS
             and not BOOKING_INTENT.search(w)]
    if not 1 <= len(words) <= 4:
        return None
    if not all(w.replace("'", "").isalpha() for w in words):
        return None
    return " ".join(w.capitalize() for w in words)


def _extract_phone(text: str) -> str | None:
    match = PHONE_PATTERN.search(text)
    if not match:
        return None
    digits = "".join(match.groups())
    if len(digits) != 9:
        return None
    return f"+998 {digits[:2]} {digits[2:5]} {digits[5:7]} {digits[7:9]}"


def _extract_time(text: str) -> str | None:
    """Kun va vaqtni ajratib oladi.

    Kun nomlari normalizatsiya qilingan matndan, soat esa xom matndan olinadi
    (normalizatsiya "09:30" dagi ikki nuqtani olib tashlaydi).
    """
    lowered = normalize(text)
    # \b chegara bilan: "dushanba" ichidan "shanba" ajratilib olinmasligi uchun
    found_days = [d for d in DAY_WORDS if re.search(rf"\b{re.escape(d)}\b", lowered)]
    time_match = TIME_PATTERN.search(text.lower())
    parts: list[str] = []
    if found_days:
        parts.append(", ".join(dict.fromkeys(found_days)))
    if time_match:
        parts.append(time_match.group(0).strip())
    return " ".join(parts) if parts else None


def _update_booking(booking: dict[str, Any], message: str) -> None:
    """Xabardan mavjud slotlarni ajratib oladi (faqat bo'sh slotlarni to'ldiradi)."""
    if not booking.get("phone"):
        phone = _extract_phone(message)
        if phone:
            booking["phone"] = phone

    if not booking.get("doctor") and not booking.get("service"):
        doctors = rag.find_doctors(message)
        if doctors:
            booking["doctor"] = doctors[0]["full_name"] + f" ({doctors[0]['specialty']})"
        else:
            services = rag.find_services(message)
            if services:
                booking["service"] = services[0]["name"]

    if not booking.get("preferred_time"):
        slot = _extract_time(message)
        if slot:
            booking["preferred_time"] = slot

    if not booking.get("full_name"):
        name = _extract_name(message)
        # shifokor/xizmat nomi ism sifatida qabul qilinmasin
        if name and _looks_like_name(name) and not rag.find_doctors(name):
            booking["full_name"] = name


def _missing_fields(booking: dict[str, Any]) -> list[str]:
    missing = []
    if not booking.get("full_name"):
        missing.append("full_name")
    if not booking.get("phone"):
        missing.append("phone")
    if not booking.get("doctor") and not booking.get("service"):
        missing.append("doctor")
    if not booking.get("preferred_time"):
        missing.append("preferred_time")
    return missing


def _booking_reply(booking: dict[str, Any], language: str) -> str:
    missing = _missing_fields(booking)
    if not missing:
        return BOOKING_TEMPLATES["ask_confirm"][language].format(
            full_name=booking["full_name"], phone=booking["phone"],
            target=booking.get("doctor") or booking.get("service"),
            preferred_time=booking["preferred_time"])
    key = {"full_name": "ask_name", "phone": "ask_phone",
           "doctor": "ask_doctor", "preferred_time": "ask_time"}[missing[0]]
    prefix = {
        "uz": "Rahmat. ",
        "ru": "Спасибо. ",
        "en": "Thank you. ",
    }[language]
    return prefix + BOOKING_TEMPLATES[key][language]


def _save_booking(booking: dict[str, Any], session_id: str) -> str:
    booking_id = "BM-" + uuid.uuid4().hex[:6].upper()
    record = {
        "booking_id": booking_id,
        "session_id": session_id,
        "full_name": booking.get("full_name"),
        "phone": booking.get("phone"),
        "doctor": booking.get("doctor"),
        "service": booking.get("service"),
        "preferred_time": booking.get("preferred_time"),
        "status": "pending_admin_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    settings.BOOKINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.BOOKINGS_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return booking_id


def _booking_details(booking: dict[str, Any], status: str) -> BookingDetails:
    return BookingDetails(
        booking_id=booking.get("booking_id"),
        full_name=booking.get("full_name"),
        phone=booking.get("phone"),
        doctor=booking.get("doctor"),
        service=booking.get("service"),
        preferred_time=booking.get("preferred_time"),
        status=status,
        missing_fields=_missing_fields(booking),
    )


def _redirect_text(message: str, language: str) -> str:
    """Tibbiy maslahat so'rovida mos mutaxassisga yo'naltirish matni."""
    doctors = rag.find_doctors(message)
    if doctors:
        d = doctors[0]
        schedule = d.get("schedule", {})
        if language == "ru":
            return (f"По вашему вопросу подойдёт {d['full_name']} — {d['specialty']} "
                    f"({', '.join(schedule.get('days', []))}, {schedule.get('time')}). "
                    f"Консультация: {d['consultation_price_uzs']:,} сум.".replace(",", " "))
        if language == "en":
            return (f"For your concern the right specialist is {d['full_name']} — {d['specialty']} "
                    f"({', '.join(schedule.get('days', []))}, {schedule.get('time')}). "
                    f"Consultation: {d['consultation_price_uzs']:,} UZS.".replace(",", " "))
        return (f"Sizning savolingiz bo'yicha mos mutaxassis — {d['full_name']} ({d['specialty']}). "
                f"Qabul kunlari: {', '.join(schedule.get('days', []))}, {schedule.get('time')}. "
                f"Konsultatsiya narxi: {d['consultation_price_uzs']:,} so'm.".replace(",", " "))

    if language == "ru":
        return ("Уточнить, к какому специалисту обратиться, можно у администратора: "
                "+998 71 200 45 45. Если хотите, я запишу вас на приём.")
    if language == "en":
        return ("The administrator can help you choose the right specialist: "
                "+998 71 200 45 45. I can also book an appointment for you.")
    return ("Qaysi mutaxassisga murojaat qilishni administrator aniqlab beradi: "
            "+998 71 200 45 45. Xohlasangiz, qabulga yozib qo'yaman.")


# --------------------------------------------------------------------------
# Endpointlar
# --------------------------------------------------------------------------
CHAT_EXAMPLES = {
    "narx": {
        "summary": "Narx savoli",
        "value": {"message": "Kardiolog qabuli qancha turadi?"},
    },
    "shifokor": {
        "summary": "Shifokor haqida",
        "value": {"message": "Aziz Karimov qabul kunlari qanday?", "session_id": None},
    },
    "qabul": {
        "summary": "Qabulga yozilish",
        "value": {"message": "Qabulga yozilmoqchiman, ismim Bekzod Aliyev, "
                             "telefon +998 90 123 45 67, kardiologga dushanba 09:30"},
    },
    "shoshilinch": {
        "summary": "Shoshilinch holat (103 shabloni qaytadi)",
        "value": {"message": "Ko'kragim og'riyapti va nafas ololmayapman"},
    },
    "dori": {
        "summary": "Dori so'rovi (rad etiladi)",
        "value": {"message": "Qanday dori ichsam bosh og'rig'im o'tadi?"},
    },
    "rus": {
        "summary": "Rus tilida",
        "value": {"message": "Сколько стоит консультация кардиолога?", "language": "ru"},
    },
}

CHAT_DESCRIPTION = """\
Foydalanuvchi xabarini qabul qiladi va klinika bilim bazasiga asoslangan javob qaytaradi.

**Ishlash tartibi**

1. `session_id` bo'lmasa yangi sessiya ochiladi (kontekst uchun javobda qaytariladi).
2. Shoshilinch holat / prompt injection / tashxis-dori so'rovi aniqlansa — LLM chaqirilmaydi,
   tayyor xavfsiz javob beriladi (`safety` bayroqlariga qarang).
3. Aks holda RAG qidiruv + LLM orqali javob yaratiladi va **javob bilim bazasiga qarshi
   tekshiriladi** (`verify_output`). Bazada yo'q narx, shifokor ismi, dori nomi yoki doza
   topilsa — javob qayta generatsiya qilinadi yoki bilim bazasidan tiklanadi
   (`safety.grounding_violation = true`).

**Kontekst:** javobdagi `session_id` ni keyingi so'rovda yuboring — oldingi suhbat hisobga olinadi.
Kontekstni tozalash: `POST /chat/reset?session_id=...`.
"""

CHAT_RESPONSES = {
    200: {
        "description": "Javob (shifokor narxi savoli misolida)",
        "content": {
            "application/json": {
                "example": {
                    "session_id": "9f2c1a7b4e10",
                    "reply": "Kardiolog Aziz Karimov konsultatsiyasi 250 000 so'm turadi [D01].",
                    "intent": "doctors",
                    "sources": [
                        {"id": "D01", "type": "doctor",
                         "title": "Aziz Karimov (Kardiolog)", "score": 8.42}
                    ],
                    "booking": None,
                    "safety": {
                        "emergency": False, "refused_medical_advice": False,
                        "injection_attempt": False, "grounding_violation": False,
                        "regenerated": False,
                    },
                    "model": "openai_compatible:google/gemini-2.5-flash",
                    "latency_ms": 1180,
                }
            }
        },
    },
    422: {"description": "Xabar bo'sh yoki 2000 belgidan uzun"},
}


@router.post("/chat", response_model=ChatResponse, summary="Klinika chatboti bilan suhbat",
             description=CHAT_DESCRIPTION, responses=CHAT_RESPONSES)
async def chat(
    request: ChatRequest = Body(openapi_examples=CHAT_EXAMPLES),
) -> ChatResponse:
    started = time.perf_counter()
    session = memory.get_or_create(request.session_id)
    language = request.language or detect_language(request.message, session.language or "uz")
    session.language = language
    session.add("user", request.message)

    safety = SafetyFlags()
    chunks: list = []
    booking_payload: BookingDetails | None = None
    intent = "other"

    guard = check_input(request.message)

    # ---------- 1. Shoshilinch holat ----------
    if guard.action == "emergency":
        reply = EMERGENCY_REPLY[language]
        intent = "emergency"
        safety.emergency = True

    # ---------- 2. Prompt injection / qoidani buzish urinishi ----------
    elif guard.action == "injection":
        reply = INJECTION_REPLY[language]
        intent = "injection_warning"
        safety.injection_attempt = True
        rules = [c for c in rag.search("chegirmalar qoidalari", top_k=4) if c.type == "rule"]
        chunks = rules[:1]

    # ---------- 3. Tashxis / dori so'rovi ----------
    elif guard.action == "medical_refusal":
        reply = MEDICAL_REFUSAL_REPLY[language].format(redirect=_redirect_text(request.message, language))
        intent = "medical_refusal"
        safety.refused_medical_advice = True
        chunks = rag.find_and_chunk(request.message)

    # ---------- 4. Qabulga yozilish jarayoni ----------
    elif session.booking and session.booking.get("status") == "active":
        reply, booking_payload, chunks, intent = _handle_booking_step(session, request.message, language)

    elif BOOKING_INTENT.search(request.message):
        session.booking = {"status": "active"}
        _update_booking(session.booking, request.message)
        missing = _missing_fields(session.booking)
        if missing:
            reply = BOOKING_TEMPLATES["ask_all"][language] if len(missing) >= 3 else _booking_reply(
                session.booking, language)
        else:
            reply = _booking_reply(session.booking, language)
        booking_payload = _booking_details(session.booking, "collecting")
        intent = "booking"
        chunks = [c for c in rag.search(request.message, top_k=3) if c.type in ("doctor", "service")]

    # ---------- 5. Oddiy ma'lumot so'rovi (RAG + LLM) ----------
    else:
        chunks = rag.search(request.message, top_k=settings.RETRIEVAL_TOP_K)
        intent = _infer_intent(request.message, chunks)
        try:
            reply, safety = await _generate_grounded(session, request.message, chunks, language, safety)
        except LLMError as exc:
            logger.error("LLM xatosi: %s", exc)
            reply = build_fallback_reply(chunks, language)
            safety.grounding_violation = False

    session.add("assistant", reply)

    return ChatResponse(
        session_id=session.id,
        reply=reply,
        intent=intent,  # type: ignore[arg-type]
        sources=[SourceChunk(**c.to_source()) for c in chunks[:4]],
        booking=booking_payload,
        safety=safety,
        model=llm.label,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _handle_booking_step(session, message: str, language: str):
    """Faol booking jarayonida keyingi qadam."""
    booking = session.booking
    chunks: list = []
    lowered = normalize(message)

    # tasdiqlash bosqichi
    if not _missing_fields(booking):
        if any(word in lowered.split() or lowered.startswith(word) for word in CANCEL_WORDS):
            booking["status"] = "cancelled"
            session.booking = {}
            return BOOKING_TEMPLATES["cancelled"][language], None, chunks, "booking"
        if any(word in lowered.split() or lowered.startswith(word) for word in CONFIRM_WORDS):
            booking_id = _save_booking(booking, session.id)
            booking["booking_id"] = booking_id
            reply = BOOKING_TEMPLATES["confirmed"][language].format(
                booking_id=booking_id, full_name=booking["full_name"], phone=booking["phone"],
                target=booking.get("doctor") or booking.get("service"),
                preferred_time=booking["preferred_time"])
            details = _booking_details(booking, "confirmed")
            session.booking = {}
            return reply, details, chunks, "booking"
        return BOOKING_TEMPLATES["ask_confirm"][language].format(
            full_name=booking["full_name"], phone=booking["phone"],
            target=booking.get("doctor") or booking.get("service"),
            preferred_time=booking["preferred_time"]), _booking_details(booking, "collecting"), chunks, "booking"

    _update_booking(booking, message)
    return _booking_reply(booking, language), _booking_details(booking, "collecting"), chunks, "booking"


async def _generate_grounded(session, message: str, chunks: list, language: str,
                             safety: SafetyFlags) -> tuple[str, SafetyFlags]:
    """LLM javobini oladi, grounding tekshiradi, kerak bo'lsa qayta urinadi."""
    history = memory.history_for_prompt(session)[:-1]  # oxirgi xabar alohida beriladi
    messages = [*history, {"role": "user", "content": message}]

    answer = await llm.generate(messages, build_system_prompt(chunks, language))
    report = verify_output(answer, chunks, rag)

    if not report.ok:
        logger.warning("Grounding buzilishi: %s", report.violations)
        safety.grounding_violation = True
        safety.regenerated = True
        retry_system = build_strict_retry_prompt(chunks, language, report.violations)
        strict_messages = [
            {"role": "user", "content": f"{message}\n\n(Qoidaga rioya qilib, faqat kontekstdagi "
                                        f"ma'lumot bilan qayta javob bering.)"}]
        answer = await llm.generate(strict_messages, retry_system)
        report = verify_output(answer, chunks, rag)
        if not report.ok:
            logger.warning("Qayta urinishdan keyin ham buzilish: %s", report.violations)
            answer = build_fallback_reply(chunks, language)
            return answer, safety

    if report.invalid_citations:
        logger.info("Mavjud bo'lmagan iqtiboslar olib tashlandi: %s", report.invalid_citations)
        answer = sanitize_citations(answer, chunks)

    return answer, safety


def _infer_intent(message: str, chunks: list) -> str:
    """Javob turini belgilaydi (faqat yorliq — oqimni o'zgartirmaydi).

    Tartib muhim: "Kardiolog qabuli qancha turadi?" — narx savoli, "qabul"
    so'zi borligi uchun booking deb belgilanmasligi kerak.
    """
    lowered = normalize(message)
    types = {c.type for c in chunks}

    if any(word in lowered for word in ("narx", "qancha", "necha", "so'm", "цена", "стоит", "price")):
        return "doctors" if rag.find_doctors(message) else "services"
    if any(word in lowered for word in ("xizmat", "paket", "check-up", "checkup", "tahlil",
                                        "analiz", "услуг", "service", "package", "test")):
        return "services"
    if "doctor" in types and (rag.find_doctors(message) or "shifokor" in lowered or "doktor" in lowered):
        return "doctors"
    if any(word in lowered for word in ("yozilmoqchi", "yozilish", "yozilaman", "bron",
                                        "записаться", "запись", "appointment")):
        return "booking"
    return "info"


@router.post("/chat/reset", response_model=SessionResetResponse,
             summary="Suhbat kontekstini tozalash",
             description="Sessiyadagi suhbat tarixi va qabul jarayoni holatini o'chiradi. "
                         "`session_id` o'zgarmaydi, ya'ni shu ID bilan davom etish mumkin.")
async def reset_session(session_id: str) -> SessionResetResponse:
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id kerak")
    session = memory.reset(session_id)
    return SessionResetResponse(session_id=session.id)
