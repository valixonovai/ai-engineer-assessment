"""Pydantic request/response modellari."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["uz", "ru", "en"]
Intent = Literal[
    "info",            # umumiy ma'lumot / FAQ
    "doctors",         # shifokor haqida
    "services",        # xizmat va narx
    "booking",         # qabulga yozilish
    "emergency",       # shoshilinch holat
    "medical_refusal",  # tashxis/dori/doza so'rovi rad etildi
    "injection_warning",  # prompt injection urinishi
    "other",
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "message": "Kardiolog qabuli qancha turadi?",
            "session_id": None,
            "language": "uz",
        }
    })

    message: str = Field(..., min_length=1, max_length=2000,
                         description="Foydalanuvchi xabari")
    session_id: str | None = Field(
        default=None, description="Suhbat konteksti uchun ID (bo'sh bo'lsa yangi sessiya ochiladi)")
    language: Language | None = Field(
        default=None, description="Majburiy til; berilmasa avtomatik aniqlanadi")


class SourceChunk(BaseModel):
    """Javobda ishlatilgan bilim bazasi bo'lagi."""
    id: str
    type: str
    title: str
    score: float


class BookingDetails(BaseModel):
    """Qabulga yozilish uchun yig'ilgan ma'lumotlar."""
    booking_id: str | None = None
    full_name: str | None = None
    phone: str | None = None
    doctor: str | None = None
    service: str | None = None
    preferred_time: str | None = None
    status: Literal["collecting", "confirmed", "cancelled"] = "collecting"
    missing_fields: list[str] = Field(default_factory=list)


class SafetyFlags(BaseModel):
    emergency: bool = False
    refused_medical_advice: bool = False
    injection_attempt: bool = False
    grounding_violation: bool = False
    regenerated: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Intent
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: BookingDetails | None = None
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    model: str = ""
    latency_ms: int = 0


class SessionResetResponse(BaseModel):
    session_id: str
    status: str = "reset"
