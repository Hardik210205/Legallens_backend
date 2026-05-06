from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str | None = None
    SECRET_KEY: str | None = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    HF_TOKEN: str | None = None
    GROQ_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @model_validator(mode="after")
    def validate_required_fields(self):
        required_fields = [
            "DATABASE_URL",
            "SECRET_KEY",
            "HF_TOKEN",
            "GROQ_API_KEY",
        ]
        missing = [name for name in required_fields if not getattr(self, name)]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Missing required environment settings: {missing_list}"
            )
        return self


settings = Settings()
