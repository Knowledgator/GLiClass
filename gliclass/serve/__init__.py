"""GLiClass serving module."""

from .config import GLiClassServeConfig
from .memory import GLiClassMemoryEstimator
from .server import GLiClassServer, GLiClassFactory, shutdown, serve_gliclass

__all__ = [
    "GLiClassClient",
    "GLiClassFactory",
    "GLiClassMemoryEstimator",
    "GLiClassServeConfig",
    "GLiClassServer",
    "serve_gliclass",
    "shutdown",
]


def __getattr__(name):
    # The HTTP client needs `requests` (the "serve" extra); import it only when used.
    if name == "GLiClassClient":
        from .client import GLiClassClient

        return GLiClassClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
