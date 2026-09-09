class RAGError(Exception):
    """Base exception for the RAG platform."""


class IngestionError(RAGError):
    """Raised when document ingestion fails."""


class DocumentNotFoundError(RAGError):
    """Raised when a requested document is not found."""


class ChunkNotFoundError(RAGError):
    """Raised when a requested chunk is not found."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class RetrievalError(RAGError):
    """Raised when retrieval fails."""


class GenerationError(RAGError):
    """Raised when LLM generation fails."""


class ConfigurationError(RAGError):
    """Raised when configuration is invalid or missing."""


class ExternalServiceError(RAGError):
    """Raised when an external service is unavailable."""


class MCPError(ExternalServiceError):
    """Raised when MCP tool invocation fails."""


class QdrantError(ExternalServiceError):
    """Raised when Qdrant operations fail."""
