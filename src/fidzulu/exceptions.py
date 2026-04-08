class RepositoryError(RuntimeError):
    """Generic repository-level error with a safe message for callers."""

class EngineCreationError(RuntimeError):
    """Raised when creating a database engine fails."""
