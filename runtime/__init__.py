"""Shared container runtime contract for spark-inference TensorFold images.

Baked directly into recipes/glm53-kelchm so the exact same image runs unchanged under SparkRun/Docker on DGX OS and
under Kubernetes LeaderWorkerSet + CDI + RDMA on Talos.
"""

RUNTIME_CONTRACT_VERSION = "v1"

from .contract import (  # noqa: E402
    DEFAULT_BINARY,
    DEFAULT_HF_HOME,
    DEFAULT_LEARN_ROOT,
    DEFAULT_RUNTIME_INPUTS_DIR,
    LaunchSpec,
    MountContract,
    RdmaConfig,
    build_launch_spec,
    parse_rdma_hcas,
    redact_secrets,
    sanitize_argv,
)
from .memguard import (  # noqa: E402
    MemGuard,
    MemSample,
    available_memory_gib,
    cgroup_available_gib,
    parked_bytes,
)

__all__ = [
    "DEFAULT_BINARY",
    "DEFAULT_HF_HOME",
    "DEFAULT_LEARN_ROOT",
    "DEFAULT_RUNTIME_INPUTS_DIR",
    "LaunchSpec",
    "MemGuard",
    "MemSample",
    "MountContract",
    "RUNTIME_CONTRACT_VERSION",
    "RdmaConfig",
    "available_memory_gib",
    "build_launch_spec",
    "cgroup_available_gib",
    "parked_bytes",
    "parse_rdma_hcas",
    "redact_secrets",
    "sanitize_argv",
]
