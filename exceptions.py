class OceanBotError(Exception):
    """Base class for all Ocean-Bot exceptions."""
    pass

class DatabaseExecutionError(OceanBotError):
    """Raised when a DuckDB SQL query fails to execute."""
    pass

class LLMGenerationError(OceanBotError):
    """Raised when the LLM service fails to generate a response."""
    pass

class VectorSearchError(OceanBotError):
    """Raised when the FAISS vector search fails."""
    pass
