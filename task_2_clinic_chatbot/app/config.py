"""Application configuration (env-based)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    LLM_PROVIDER: str = "openai_compatible"   # openai_compatible | ollama
    LLM_MODEL: str = "google/gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 800
    LLM_TIMEOUT: float = 90.0
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_THINK: bool = False
    OLLAMA_NUM_CTX: int = 4096
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    # OpenRouter statistikasi uchun (majburiy emas)
    OPENROUTER_APP_NAME: str = "Shifo Med Clinic Chatbot"
    OPENROUTER_APP_URL: str = ""

    # --- app ---
    SESSION_TTL_MINUTES: int = 60
    MAX_HISTORY_MESSAGES: int = 12
    RETRIEVAL_TOP_K: int = 6
    LOG_LEVEL: str = "INFO"

    # --- paths ---
    DATA_DIR: Path = DATA_DIR
    BOOKINGS_FILE: Path = DATA_DIR / "bookings.jsonl"

    @property
    def clinic_files(self) -> dict[str, Path]:
        return {
            "doctors": self.DATA_DIR / "doctors.json",
            "services": self.DATA_DIR / "services.json",
            "rules": self.DATA_DIR / "rules.json",
            "faqs": self.DATA_DIR / "faqs.json",
        }

    @property
    def is_openai_compatible(self) -> bool:
        return self.LLM_PROVIDER == "openai_compatible"

    @property
    def llm_configured(self) -> bool:
        """LLM chaqiruvga tayyormi?"""
        if self.is_openai_compatible:
            return bool(self.OPENAI_API_KEY.strip())
        return True  # ollama uchun kalit kerak emas


settings = Settings()

# Diqqat: kalit bo'lmasa ilova ishga tushaveradi, lekin /health "degraded"
# holatini ko'rsatadi va /chat bilim bazasidan deterministik javob qaytaradi.
# Sabab: konfiguratsiya xatosi butun xizmatni yiqitmasligi kerak.
