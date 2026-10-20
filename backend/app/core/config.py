from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Вычисляем корневую директорию монорепозитория (recommender-system/)
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Глобальная конфигурация приложения с валидацией типов."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),  # Убирает конфликт с полем model_name
    )

    # Общие настройки приложения
    environment: Literal["development", "production", "testing"] = "development"
    debug: bool = True
    app_name: str = "PNRPU Educational Recommender API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_v1_prefix: str = "/api/v1"

    # Настройки PostgreSQL
    postgres_user: str = "postgres"
    postgres_password: str = "postgres_dev_password"
    postgres_db: str = "recommender_db"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433

    # Настройки модели эмбеддингов FRIDA
    model_name: str = "ai-forever/FRIDA"
    local_model_path: str = "./models/frida"
    embedding_dimension: int = 1536

    # Настройки планировщика (ETL Scheduler)
    scheduler_enabled_by_default: bool = False
    scheduler_default_interval_hours: int = 168
    pipeline_lock_timeout_minutes: int = 60

    # Настройки CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_database_url(self) -> str:
        """Формирует строку подключения для асинхронного драйвера asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек (Singleton)."""
    return Settings()


settings = get_settings()