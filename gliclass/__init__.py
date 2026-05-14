from .model import GLiClassModel, GLiClassBiEncoder, GLiClassUniEncoder, GLiClassEncoderDecoderCLS
from .config import GLiClassModelConfig
from .pipeline import (
    ZeroShotClassificationPipeline,
    BiEncoderZeroShotClassificationPipeline,
    ZeroShotClassificationWithChunkingPipeline,
)

__version__ = "0.1.18"

# Serve module (optional import)
try:
    from . import serve
except ImportError:
    serve = None
