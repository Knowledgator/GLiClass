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


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GLiClass Ray Serve deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default=None, help="Model name or path")
    model_group.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    model_group.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "fp16", "bfloat16", "bf16"],
        help="Data type for model weights",
    )
    model_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Model quantization mode, if supported by the GLiClass model",
    )

    limits_group = parser.add_argument_group("Model Limits")
    limits_group.add_argument("--max-model-len", type=int, default=None, help="Maximum sequence length")
    limits_group.add_argument("--max-batch-size", type=int, default=None, help="Maximum batch size")
    limits_group.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Maximum number of labels (-1 for unlimited)",
    )
    limits_group.add_argument(
        "--max-labels-alloc",
        type=str,
        default=None,
        help='Label memory allocation: "dynamic", "fixed", or integer (e.g., "50")',
    )

    threshold_group = parser.add_argument_group("Thresholds")
    threshold_group.add_argument(
        "--default-threshold",
        type=float,
        default=None,
        help="Default confidence threshold",
    )

    replica_group = parser.add_argument_group("Replica Configuration")
    replica_group.add_argument("--num-replicas", type=int, default=None, help="Number of model replicas")
    replica_group.add_argument(
        "--num-gpus-per-replica",
        type=float,
        default=None,
        help="Number of GPUs per replica",
    )
    replica_group.add_argument(
        "--num-cpus-per-replica",
        type=float,
        default=None,
        help="Number of CPUs per replica",
    )

    batch_group = parser.add_argument_group("Batching Configuration")
    batch_group.add_argument(
        "--batch-wait-timeout-ms",
        type=float,
        default=None,
        help="Batch wait timeout in milliseconds",
    )
    batch_group.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="Request timeout in seconds",
    )
    batch_group.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=None,
        help="Maximum number of ongoing requests",
    )
    batch_group.add_argument(
        "--queue-capacity",
        type=int,
        default=None,
        help="Request queue capacity",
    )
    batch_group.add_argument(
        "--precompiled-batch-sizes",
        type=_parse_int_list,
        default=None,
        help="Comma-separated list of batch sizes to precompile",
    )

    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    server_group.add_argument("--port", type=int, default=None, help="HTTP port for Ray Serve")
    server_group.add_argument("--route-prefix", type=str, default=None, help="HTTP route prefix")
    server_group.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (default: local)",
    )

    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--tokenizer-threads",
        type=int,
        default=None,
        help="Number of tokenizer threads",
    )
    perf_group.add_argument(
        "--enable-compilation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile",
    )
    perf_group.add_argument(
        "--precompile-on-startup",
        action="store_true",
        default=None,
        help="Run warmup inference for configured precompiled batch sizes",
    )
    perf_group.add_argument(
        "--calibrate-on-startup",
        action="store_true",
        default=None,
        help="Run memory calibration on startup",
    )
    perf_group.add_argument(
        "--use-memory-aware-batching",
        action="store_true",
        default=None,
        help="Use memory-aware dynamic batching",
    )
    perf_group.add_argument(
        "--warmup-iterations",
        type=int,
        default=None,
        help="Number of warmup iterations per batch size",
    )

    memory_group = parser.add_argument_group("Memory Configuration")
    memory_group.add_argument(
        "--target-memory-fraction",
        type=float,
        default=None,
        help="Target GPU memory fraction (0.0-1.0)",
    )
    memory_group.add_argument(
        "--memory-overhead-factor",
        type=float,
        default=None,
        help="Memory overhead factor for safety margin",
    )
    memory_group.add_argument(
        "--calibration-min-batch-size",
        type=int,
        default=None,
        help="Minimum batch size for calibration",
    )
    memory_group.add_argument(
        "--calibration-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for calibration",
    )
    memory_group.add_argument(
        "--calibration-min-seq-len",
        type=int,
        default=None,
        help="Minimum sequence length for calibration",
    )
    memory_group.add_argument(
        "--calibration-probe-batch-size",
        type=int,
        default=None,
        help="Probe batch size for memory calibration",
    )

    polylora_group = parser.add_argument_group("PolyLoRA Configuration")
    polylora_group.add_argument(
        "--enable-polylora",
        action="store_true",
        default=None,
        help="Enable PolyLoRA adapter serving",
    )
    polylora_group.add_argument(
        "--polylora-adapter-weight-modules",
        type=_parse_str_list,
        default=None,
        help="Comma-separated target module names for adapter weights",
    )
    polylora_group.add_argument("--polylora-max-rank", type=int, default=None, help="Maximum LoRA rank")
    polylora_group.add_argument(
        "--polylora-max-gpu-adapters",
        type=int,
        default=None,
        help="Maximum PolyLoRA GPU adapter slots",
    )
    polylora_group.add_argument(
        "--polylora-max-cpu-adapters",
        type=int,
        default=None,
        help="Maximum PolyLoRA CPU adapters",
    )
    polylora_group.add_argument(
        "--polylora-disk-cache-dir",
        type=str,
        default=None,
        help="PolyLoRA disk cache directory",
    )
    polylora_group.add_argument(
        "--polylora-max-disk-adapters",
        type=int,
        default=None,
        help="Maximum PolyLoRA disk-cached adapters",
    )
    polylora_group.add_argument(
        "--polylora-base-adapter-id",
        type=str,
        default=None,
        help="Adapter id reserved for base-only inference",
    )
    polylora_group.add_argument(
        "--polylora-use-triton-kernels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Triton kernels for PolyLoRA",
    )
    polylora_group.add_argument(
        "--polylora-adapter-id-pattern",
        type=str,
        default=None,
        help="Regular expression for valid adapter ids",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> GLiClassServeConfig:
    if args.config:
        logger.info("Loading config from: %s", args.config)
        config = GLiClassServeConfig.from_yaml(args.config)
    else:
        config = GLiClassServeConfig(model=args.model or "knowledgator/gliclass-edge-v3.0")

    max_labels_alloc_value = args.max_labels_alloc
    if max_labels_alloc_value and max_labels_alloc_value.isdigit():
        max_labels_alloc_value = int(max_labels_alloc_value)

    cli_overrides = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "max_batch_size": args.max_batch_size,
        "max_labels": args.max_labels,
        "max_labels_alloc": max_labels_alloc_value,
        "default_threshold": args.default_threshold,
        "num_replicas": args.num_replicas,
        "num_gpus_per_replica": args.num_gpus_per_replica,
        "num_cpus_per_replica": args.num_cpus_per_replica,
        "batch_wait_timeout_ms": args.batch_wait_timeout_ms,
        "request_timeout_s": args.request_timeout_s,
        "max_ongoing_requests": args.max_ongoing_requests,
        "queue_capacity": args.queue_capacity,
        "precompiled_batch_sizes": args.precompiled_batch_sizes,
        "http_port": args.port,
        "route_prefix": args.route_prefix,
        "ray_address": args.ray_address,
        "tokenizer_threads": args.tokenizer_threads,
        "enable_compilation": args.enable_compilation,
        "precompile_on_startup": args.precompile_on_startup,
        "calibrate_on_startup": args.calibrate_on_startup,
        "use_memory_aware_batching": args.use_memory_aware_batching,
        "warmup_iterations": args.warmup_iterations,
        "target_memory_fraction": args.target_memory_fraction,
        "memory_overhead_factor": args.memory_overhead_factor,
        "calibration_min_batch_size": args.calibration_min_batch_size,
        "calibration_max_batch_size": args.calibration_max_batch_size,
        "calibration_min_seq_len": args.calibration_min_seq_len,
        "calibration_probe_batch_size": args.calibration_probe_batch_size,
        "enable_polylora": args.enable_polylora,
        "polylora_adapter_weight_modules": args.polylora_adapter_weight_modules,
        "polylora_max_rank": args.polylora_max_rank,
        "polylora_max_gpu_adapters": args.polylora_max_gpu_adapters,
        "polylora_max_cpu_adapters": args.polylora_max_cpu_adapters,
        "polylora_disk_cache_dir": args.polylora_disk_cache_dir,
        "polylora_max_disk_adapters": args.polylora_max_disk_adapters,
        "polylora_base_adapter_id": args.polylora_base_adapter_id,
        "polylora_use_triton_kernels": args.polylora_use_triton_kernels,
        "polylora_adapter_id_pattern": args.polylora_adapter_id_pattern,
    }
    config.update(**cli_overrides)
    config.__post_init__()
    return config


def main():
    """Main entry point for GLiClass serving."""
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    logger.info("=" * 60)
    logger.info("GLiClass Serve Configuration:")
    logger.info("  Model: %s", config.model)
    logger.info("  Device: %s", config.device)
    logger.info("  Dtype: %s", config.dtype)
    logger.info("  Quantization: %s", config.quantization or "disabled")
    logger.info("  Max model length: %s", config.max_model_len)
    logger.info("  Max batch size: %s", config.max_batch_size)
    logger.info("  Precompiled batch sizes: %s", config.precompiled_batch_sizes)
    logger.info("  Max labels: %s", config.max_labels)
    logger.info("  Max labels alloc: %s", config.max_labels_alloc)
    logger.info("  HTTP port: %s", config.http_port)
    logger.info("  Route prefix: %s", config.route_prefix)
    logger.info("  Num replicas: %s", config.num_replicas)
    logger.info("  GPUs per replica: %s", config.num_gpus_per_replica)
    logger.info("  CPUs per replica: %s", config.num_cpus_per_replica)
    logger.info("  Compile: %s", config.enable_compilation)
    logger.info("  Precompile on startup: %s", config.precompile_on_startup)
    logger.info("  Calibrate on startup: %s", config.calibrate_on_startup)
    logger.info("  Memory-aware batching: %s", config.use_memory_aware_batching)
    logger.info("  PolyLoRA: %s", config.enable_polylora)
    logger.info("=" * 60)

    logger.info("Deploying GLiClass with model: %s", config.model)
    _app = serve_gliclass(config, blocking=False, host=args.host)

    logger.info("GLiClass server running at http://%s:%d%s", args.host, config.http_port, config.route_prefix)
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
