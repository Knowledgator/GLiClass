"""GLiClass serving module."""

from .api import APIError
from .config import GLiClassServeConfig
from .memory import GLiClassMemoryEstimator
from .server import GLiClassServer, GLiClassFactory, shutdown, serve_gliclass

__all__ = [
    "APIError",
    "GLiClassAPIError",
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
    if name in ("GLiClassClient", "GLiClassAPIError"):
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
