from .model import GLiClassModel, GLiClassBiEncoder, GLiClassDecoderKV, GLiClassUniEncoder, GLiClassEncoderDecoderCLS
from .config import GLiClassModelConfig
from .pipeline import (
    ZeroShotClassificationPipeline,
    BiEncoderZeroShotClassificationPipeline,
    ZeroShotClassificationWithChunkingPipeline,
)
from .streaming import StreamingZeroShotClassificationPipeline

__version__ = "0.1.20"

# Serve module (optional import)
try:
    from . import serve
except ImportError:
    serve = None
