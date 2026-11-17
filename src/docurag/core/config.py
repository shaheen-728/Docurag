from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocuRAG"

    postgres_db: str = "docurag"
    postgres_user: str = "docurag"
    postgres_password: SecretStr

    db_host: str = "postgres"
    db_port: int = Field(default=5432, ge=1, le=65535)

    redis_url: str = "redis://redis:6379/0"

    llm_provider: Literal["gemini", "openai"] = "gemini"
    embedding_dimensions: Literal[768] = 768

    gemini_api_key: SecretStr = SecretStr("")
    gemini_generation_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    openai_api_key: SecretStr = SecretStr("")
    openai_generation_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
