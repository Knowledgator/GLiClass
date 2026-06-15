"""Configuration for GLiClass Ray Serve deployment."""

from pathlib import Path
from dataclasses import field, asdict, dataclass

import yaml


@dataclass
class GLiClassServeConfig:
    """Configuration for GLiClass Ray Serve deployment.

    This config controls model loading, serving parameters, and dynamic batching behavior.
    """

    model: str
    device: str = "cuda"
    dtype: str = "bfloat16"

    quantization: str | None = None

    max_model_len: int = 2048
    max_labels: int = -1
    max_labels_alloc: str | int = "dynamic"

    default_threshold: float = 0.5

    num_replicas: int = 1
    num_gpus_per_replica: float = 1.0
    num_cpus_per_replica: float = 1.0

    max_batch_size: int = 32
    batch_wait_timeout_ms: float = 20.0
    request_timeout_s: float = 30.0
    max_ongoing_requests: int = 256
    queue_capacity: int = 4096

    route_prefix: str = "/gliclass"

    tokenizer_threads: int = 4

    enable_compilation: bool = False
    calibrate_on_startup: bool = False
    precompile_on_startup: bool = False
    use_memory_aware_batching: bool = False

    precompiled_batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    target_memory_fraction: float = 0.8
    memory_overhead_factor: float = 1.3

    calibration_min_seq_len: int = 64
    calibration_min_batch_size: int = 1
    calibration_max_batch_size: int = 64
    calibration_probe_batch_size: int = 2

    warmup_iterations: int = 3

    http_port: int = 8000

    ray_address: str | None = None

    enable_polylora: bool = False
    polylora_adapter_weight_modules: list[str] | None = None
    polylora_max_rank: int = 16
    polylora_max_gpu_adapters: int = 8
    polylora_max_cpu_adapters: int | None = 128
    polylora_disk_cache_dir: str | None = None
    polylora_max_disk_adapters: int | None = None
    polylora_base_adapter_id: str = "__base__"
    polylora_use_triton_kernels: bool = True
    polylora_adapter_id_pattern: str = r"^[A-Za-z0-9_.-]{1,128}$"

    def __post_init__(self):
        if self.max_batch_size not in self.precompiled_batch_sizes:
            self.precompiled_batch_sizes = sorted(set(self.precompiled_batch_sizes) | {self.max_batch_size})
        self.precompiled_batch_sizes = sorted(self.precompiled_batch_sizes)

    def to_env_vars(self) -> dict:
        """Convert config to environment variables for model loading."""
        env = {}
        if self.tokenizer_threads > 0:
            env["TOKENIZERS_PARALLELISM"] = "true"
        return env

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GLiClassServeConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            GLiClassServeConfig instance
        """
        config_path = Path(config_path)
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f) or {}
        legacy_polylora_keys = {
            "enable_mlora": "enable_polylora",
            "mlora_adapter_weight_modules": "polylora_adapter_weight_modules",
            "mlora_max_rank": "polylora_max_rank",
            "mlora_max_gpu_adapters": "polylora_max_gpu_adapters",
            "mlora_max_cpu_adapters": "polylora_max_cpu_adapters",
            "mlora_disk_cache_dir": "polylora_disk_cache_dir",
            "mlora_max_disk_adapters": "polylora_max_disk_adapters",
            "mlora_base_adapter_id": "polylora_base_adapter_id",
            "mlora_use_triton_kernels": "polylora_use_triton_kernels",
            "mlora_adapter_id_pattern": "polylora_adapter_id_pattern",
            "mlora_target_modules": "polylora_target_modules",
        }
        for old_key, new_key in legacy_polylora_keys.items():
            if old_key in config_dict and new_key not in config_dict:
                config_dict[new_key] = config_dict.pop(old_key)
            else:
                config_dict.pop(old_key, None)
        config_dict.pop("mlora_lazy_load_adapters", None)
        config_dict.pop("mlora_allow_runtime_adapter_loading", None)
        config_dict.pop("mlora_allow_unsafe_bin_adapters", None)
        config_dict.pop("polylora_lazy_load_adapters", None)
        config_dict.pop("polylora_allow_runtime_adapter_loading", None)
        config_dict.pop("polylora_allow_unsafe_bin_adapters", None)
        if "polylora_target_modules" in config_dict and "polylora_adapter_weight_modules" not in config_dict:
            config_dict["polylora_adapter_weight_modules"] = config_dict.pop("polylora_target_modules")
        else:
            config_dict.pop("polylora_target_modules", None)
        return cls(**config_dict)

    def to_yaml(self, config_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path to save YAML config
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def update(self, **kwargs) -> "GLiClassServeConfig":
        """Update config with provided kwargs (for CLI override).

        Args:
            **kwargs: Fields to update (None values are ignored)

        Returns:
            Updated config instance
        """
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self
