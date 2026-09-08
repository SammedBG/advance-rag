from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "advanced-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()