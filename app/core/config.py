from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "advanced-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"
    postgres_url: str = ""
    sqlite_fallback_url: str = "sqlite+aiosqlite:///./data/rag.db"

    # Caching
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    embedding_cache_ttl_seconds: int = 3600

    # Security & Authentication
    security_enabled: bool = False
    api_keys: list[str] = ["dev-api-key-12345"]
    jwt_secret_key: str = "insecure-dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60

    mcp_server_url: str = "http://127.0.0.1:8001/mcp"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def effective_database_url(self) -> str:
        if self.postgres_url:
            return self.postgres_url
        if self.database_url:
            return self.database_url
        return self.sqlite_fallback_url


settings = Settings()