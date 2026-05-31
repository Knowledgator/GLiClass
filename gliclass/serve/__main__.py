"""CLI entry point for GLiClass serving."""

import sys
import signal
import logging
import argparse

import ray
from ray import serve

from .config import GLiClassServeConfig
from .server import serve_gliclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for GLiClass serving."""
    parser = argparse.ArgumentParser(
        description="GLiClass Ray Serve deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Maximum number of labels (-1 for unlimited)",
    )
    parser.add_argument(
        "--max-labels-alloc",
        type=str,
        default=None,
        help='Label memory allocation: "dynamic", "fixed", or integer (e.g., "50")',
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )
    parser.add_argument(
        "--route-prefix",
        type=str,
        default=None,
        help="HTTP route prefix",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Number of model replicas",
    )

    # Performance configuration
    parser.add_argument(
        "--calibrate-on-startup",
        action="store_true",
        default=None,
        help="Run memory calibration on startup",
    )
    parser.add_argument(
        "--precompile-on-startup",
        action="store_true",
        default=None,
        help="Precompile model on startup",
    )
    parser.add_argument(
        "--use-memory-aware-batching",
        action="store_true",
        default=None,
        help="Use memory-aware dynamic batching",
    )
    parser.add_argument(
        "--enable-compilation",
        action="store_true",
        default=None,
        help="Enable torch.compile",
    )
    parser.add_argument(
        "--tokenizer-threads",
        type=int,
        default=None,
        help="Number of tokenizer threads",
    )

    # Calibration configuration
    parser.add_argument(
        "--calibration-min-batch-size",
        type=int,
        default=None,
        help="Minimum batch size for calibration",
    )
    parser.add_argument(
        "--calibration-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for calibration",
    )
    parser.add_argument(
        "--calibration-min-seq-len",
        type=int,
        default=None,
        help="Minimum sequence length for calibration",
    )

    args = parser.parse_args()

    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = GLiClassServeConfig.from_yaml(args.config)
    else:
        config = GLiClassServeConfig(model=args.model or "knowledgator/gliclass-edge-v3.0")

    # Convert max_labels_alloc to int if it's a digit string
    max_labels_alloc_value = args.max_labels_alloc
    if max_labels_alloc_value and max_labels_alloc_value.isdigit():
        max_labels_alloc_value = int(max_labels_alloc_value)

    cli_overrides = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_batch_size": args.max_batch_size,
        "max_labels": args.max_labels,
        "max_labels_alloc": max_labels_alloc_value,
        "http_port": args.port,
        "route_prefix": args.route_prefix,
        "num_replicas": args.num_replicas,
        "calibrate_on_startup": args.calibrate_on_startup,
        "precompile_on_startup": args.precompile_on_startup,
        "use_memory_aware_batching": args.use_memory_aware_batching,
        "enable_compilation": args.enable_compilation,
        "tokenizer_threads": args.tokenizer_threads,
        "calibration_min_batch_size": args.calibration_min_batch_size,
        "calibration_max_batch_size": args.calibration_max_batch_size,
        "calibration_min_seq_len": args.calibration_min_seq_len,
    }
    config.update(**cli_overrides)

    logger.info("=" * 60)
    logger.info("GLiClass Serve Configuration:")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Dtype: {config.dtype}")
    logger.info(f"  Max model length: {config.max_model_len}")
    logger.info(f"  Max batch size: {config.max_batch_size}")
    logger.info(f"  Max labels: {config.max_labels}")
    logger.info(f"  Max labels alloc: {config.max_labels_alloc}")
    logger.info(f"  HTTP port: {config.http_port}")
    logger.info(f"  Route prefix: {config.route_prefix}")
    logger.info(f"  Num replicas: {config.num_replicas}")
    logger.info(f"  Calibrate on startup: {config.calibrate_on_startup}")
    logger.info(f"  Precompile on startup: {config.precompile_on_startup}")
    logger.info(f"  Memory-aware batching: {config.use_memory_aware_batching}")
    logger.info("=" * 60)

    logger.info("Initializing Ray...")
    ray.init(ignore_reinit_error=True)

    logger.info(f"Deploying GLiClass with model: {config.model}")
    _app = serve_gliclass(config, blocking=False, host=args.host)  # Keep reference to prevent GC

    logger.info(f"GLiClass server running at http://{args.host}:{config.http_port}{config.route_prefix}")
    logger.info("Press Ctrl+C to stop the server")

    def signal_handler(_sig, _frame):
        logger.info("Shutting down server...")
        serve.shutdown()
        ray.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    import time

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
