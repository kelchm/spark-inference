"""3-way mount contract, parameterized RDMA/fabric config, and launch builder for TensorFold images.

Everything here is orchestrator-neutral: SparkRun/Docker on DGX OS and Kubernetes LeaderWorkerSet + CDI + RDMA on Talos
supply environment variables and mounts, and this module validates and translates them into the native engine invocation
without hardcoding host interface names, HCA device names, or host-specific mount paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import shlex
import socket
from typing import Callable, Mapping

DEFAULT_BINARY = "/opt/tensorfold/native/bin/tensorfold-native"
DEFAULT_HF_HOME = "/cache/huggingface"
DEFAULT_RUNTIME_INPUTS_DIR = "/opt/tensorfold-runtime/inputs"
DEFAULT_LEARN_ROOT = "/var/lib/tensorfold/learn"

HCA_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$"
)
IFNAME_RE = re.compile(r"^[A-Za-z0-9_.,^=:-]+$")
VALID_KV_PAGES = frozenset({"auto", "on", "1", "all", "off", "0"})
VALID_KV_CACHES = frozenset({"fp8", "bf16"})
VALID_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
VALID_LAUNCH_STYLES = frozenset({"flags", "legacy"})
FORBIDDEN_LEARN_NAMES = frozenset({"tensorfold-spill", "spill"})
SECRET_ENV_NAMES = frozenset({"TENSORFOLD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
SECRET_CLI_FLAGS = frozenset({"--api-key", "--hf-token", "--token"})
PROTECTED_EXTRA_FLAGS = frozenset({
    "--tp", "--rank", "--master", "--master-port",
    "--learn-dir", "--learn-conversations",
    "--prompt-cache-disk", "--prompt-cache-disk-gib", "--prompt-cache-dir",
    "--port",
})


def sanitize_argv(argv: tuple[str, ...] | list[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Redact credential-bearing CLI flags (e.g. --api-key) and env secrets from argv before printing/logging."""
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        matched = False
        for flag in SECRET_CLI_FLAGS:
            if arg == flag:
                out.append(arg)
                redact_next = True
                matched = True
                break
            if arg.startswith(flag + "="):
                out.append(f"{flag}=<redacted>")
                matched = True
                break
        if not matched:
            out.append(redact_secrets(arg, env=env))
    return out


@dataclass(frozen=True)
class MountContract:
    """The 3-way container mount contract: HF cache, runtime inputs/drafter, and persistent learn store."""

    hf_home: Path
    runtime_inputs_dir: Path
    model_path: str
    drafter: str | None
    chat_template: Path | None
    learn_enabled: bool
    learn_dir: Path | None
    learn_root: Path | None
    learn_gib: int | None
    learn_min_free_gib: int | None
    learn_daily_gib: int | None


@dataclass(frozen=True)
class RdmaConfig:
    """Parameterized RDMA HCA and fabric transport settings (zero hardcoded host interfaces)."""

    enabled: bool
    hcas: tuple[str, ...]
    gid_index: int | None
    max_kb: int
    socket_ifname: str | None
    min_nchannels: int
    max_nchannels: int


@dataclass(frozen=True)
class LaunchSpec:
    """Validated engine argv, environment, and metadata for one container rank."""

    rank: int
    world: int
    master_raw: str | None
    master_ipv4: str | None
    master_port: int | None
    nccl_port: int | None
    launch_style: str
    mounts: MountContract
    rdma: RdmaConfig
    argv: tuple[str, ...]
    env: dict[str, str]


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (0/1/true/false), got {raw!r}")


def _int_env(env: Mapping[str, str], name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        text = raw.strip()
        if not text.isascii() or not re.fullmatch(r"-?[0-9]+", text):
            raise ValueError(f"{name} must be an integer in {minimum}..{maximum}, got {raw!r}")
        value = int(text)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}, got {value}")
    return value


def _first_set(env: Mapping[str, str], *names: str) -> tuple[str | None, str | None]:
    for name in names:
        raw = env.get(name)
        if raw is not None and raw.strip():
            return name, raw.strip()
    return None, None


def _validate_abs_clean_path(raw: str, name: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {raw!r}")
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain '..' path traversal components: {raw!r}")
    return path


def resolve_master_ipv4(
    master: str,
    *,
    resolver: Callable[[str], str] | None = None,
) -> str:
    """Validate an IPv4 literal or RFC-1123 hostname (e.g. Kubernetes LWS leader DNS) and resolve to IPv4."""
    text = master.strip()
    if not text:
        raise ValueError("Master fabric address must be non-empty")
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        if not HOSTNAME_RE.fullmatch(text):
            raise ValueError(f"Master fabric address must be an IPv4 address or hostname, got {master!r}") from None
        if resolver is not None:
            resolved = resolver(text)
        else:
            infos = socket.getaddrinfo(text, None, socket.AF_INET, socket.SOCK_STREAM)
            if not infos:
                raise ValueError(f"Master hostname {text!r} did not resolve to an IPv4 address")
            resolved = infos[0][4][0]
        try:
            resolved_ip = ipaddress.ip_address(resolved)
        except ValueError as err:
            raise ValueError(f"Master hostname {text!r} resolved to invalid address {resolved!r}") from err
        if not isinstance(resolved_ip, ipaddress.IPv4Address):
            raise ValueError(f"Master hostname {text!r} must resolve to IPv4, got {resolved!r}")
        return str(resolved_ip)
    if not isinstance(ip, ipaddress.IPv4Address):
        raise ValueError(f"TensorFold TP rendezvous requires IPv4, got {master!r}")
    return str(ip)


def parse_rdma_hcas(
    raw: str,
    *,
    sysfs_ib_dir: Path | None = None,
    max_devices: int = 2,
) -> tuple[str, ...]:
    """Parse and validate RDMA HCA names from TF_ROCE_HCA or NCCL_IB_HCA.

    Accepts optional leading '=' and optional ':1' port suffixes (as NCCL_IB_HCA uses), refuses '^' exclusion lists,
    non-port-1 suffixes, duplicate names, and >max_devices names (TensorFold RoCE stripes across at most two HCAs).
    """
    text = raw.strip()
    if not text:
        return ()
    if text.startswith("^"):
        raise ValueError(
            f"RDMA HCA exclusion list {raw!r} has no TF_ROCE_HCA equivalent; "
            "name explicit active RDMA devices in TF_ROCE_HCA or NCCL_IB_HCA"
        )
    body = text.removeprefix("=")
    devices: list[str] = []
    seen: set[str] = set()
    for token in body.split(","):
        item = token.strip()
        if not item:
            raise ValueError(f"Empty RDMA HCA entry in {raw!r}")
        name, sep, port = item.partition(":")
        if not name or not HCA_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid RDMA HCA device name {item!r}")
        if sep and port != "1":
            raise ValueError(f"RDMA HCA {item!r} specifies port {port!r}; TensorFold RoCE uses port 1 only")
        if name in seen:
            raise ValueError(f"Duplicate RDMA HCA device {name!r} in {raw!r}")
        seen.add(name)
        if sysfs_ib_dir is not None and sysfs_ib_dir.is_dir() and not (sysfs_ib_dir / name).exists():
            raise ValueError(f"RDMA HCA {name!r} is not present in {sysfs_ib_dir}")
        devices.append(name)
    if len(devices) > max_devices:
        raise ValueError(
            f"TensorFold RoCE supports at most {max_devices} RDMA HCA devices, got {len(devices)}: {raw!r}"
        )
    return tuple(devices)


_ASCII_WS = " \t\r\n\x0b\x0c"


def parse_api_keys(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated API keys matching TensorFold auth.zig (split ',', trim ASCII whitespace, skip empty)."""
    if not raw:
        return ()
    keys: list[str] = []
    for part in raw.split(","):
        key = part.strip(_ASCII_WS)
        if key:
            keys.append(key)
    return tuple(keys)


def primary_api_key(raw: str | None) -> str | None:
    """Return the first non-empty API key from a single or comma-separated key string, or None."""
    keys = parse_api_keys(raw)
    return keys[0] if keys else None


def redact_secrets(text: str, env: Mapping[str, str] | None = None, extra_secrets: tuple[str, ...] = ()) -> str:
    """Scrub bearer tokens and secret env values (including comma-separated key members) from diagnostic strings."""
    cleaned = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1<redacted>", text)
    cleaned = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1<redacted>", cleaned)
    long_candidates: list[str] = []
    short_candidates: list[str] = []

    def _add_candidate(candidate: str) -> None:
        if not candidate or not candidate.strip(_ASCII_WS):
            return
        if len(candidate) >= 4:
            long_candidates.append(candidate)
        else:
            short_candidates.append(candidate)

    def _add_secret(val: str | None) -> None:
        if not val:
            return
        _add_candidate(val)
        trimmed = val.strip(_ASCII_WS)
        if trimmed and trimmed != val:
            _add_candidate(trimmed)
        if "," in val:
            for member in parse_api_keys(val):
                _add_candidate(member)

    for extra in extra_secrets:
        _add_secret(extra)
    if env is not None:
        for key in SECRET_ENV_NAMES:
            _add_secret(env.get(key))
    for secret in sorted(set(long_candidates), key=len, reverse=True):
        if secret:
            cleaned = cleaned.replace(secret, "<redacted>")
    for secret in sorted(set(short_candidates), key=len, reverse=True):
        if secret:
            cleaned = re.sub(
                rf"(?<![A-Za-z0-9._~+/=-]){re.escape(secret)}(?![A-Za-z0-9._~+/=-])",
                "<redacted>",
                cleaned,
            )
    return cleaned


def _validate_mounts(
    env: Mapping[str, str],
    *,
    rank: int,
    verify_mounts: bool,
) -> MountContract:
    hf_home = _validate_abs_clean_path(env.get("HF_HOME", DEFAULT_HF_HOME).strip() or DEFAULT_HF_HOME, "HF_HOME")
    runtime_inputs_dir = _validate_abs_clean_path(
        env.get("TF_RUNTIME_INPUTS_DIR", DEFAULT_RUNTIME_INPUTS_DIR).strip() or DEFAULT_RUNTIME_INPUTS_DIR,
        "TF_RUNTIME_INPUTS_DIR",
    )

    model_path = (env.get("MODEL_PATH") or "").strip()
    if not model_path:
        raise ValueError("MODEL_PATH is required (path inside the HF cache mount or model checkpoint path)")
    if model_path.startswith("/"):
        model_as_path = _validate_abs_clean_path(model_path, "MODEL_PATH")
        if verify_mounts and not model_as_path.exists():
            raise ValueError(f"MODEL_PATH does not exist in container mount: {model_as_path}")
    if verify_mounts and not hf_home.is_dir():
        raise ValueError(f"HF_HOME cache directory does not exist: {hf_home}")

    no_drafts = _bool_env(env, "NO_DRAFTS", False)
    drafter_raw = (env.get("DRAFTER") or "").strip()
    drafter: str | None
    if no_drafts:
        if drafter_raw:
            raise ValueError("DRAFTER cannot be set when NO_DRAFTS=1")
        drafter = None
    else:
        if not drafter_raw:
            raise ValueError("DRAFTER is required unless NO_DRAFTS=1")
        if drafter_raw.startswith("/"):
            drafter_path = _validate_abs_clean_path(drafter_raw, "DRAFTER")
            if verify_mounts and not drafter_path.exists():
                raise ValueError(f"DRAFTER path does not exist in container mount: {drafter_path}")
        drafter = drafter_raw

    chat_template_raw = (env.get("CHAT_TEMPLATE") or "").strip()
    chat_template: Path | None = None
    if chat_template_raw:
        chat_template = _validate_abs_clean_path(chat_template_raw, "CHAT_TEMPLATE")
        if verify_mounts and not chat_template.is_file():
            raise ValueError(f"CHAT_TEMPLATE file does not exist: {chat_template}")
    else:
        default_template = runtime_inputs_dir / "chat_template.jinja"
        if default_template.is_file():
            chat_template = default_template

    learn_enabled = _bool_env(env, "GLM_LEARN_ENABLED", True)
    learn_root_raw = (env.get("GLM_LEARN_ROOT") or "").strip()
    learn_root = _validate_abs_clean_path(learn_root_raw, "GLM_LEARN_ROOT") if learn_root_raw else None

    learn_dir: Path | None = None
    learn_gib: int | None = None
    learn_min_free_gib: int | None = None
    learn_daily_gib: int | None = None
    if learn_enabled:
        learn_raw = (env.get("GLM_LEARN_DIR") or "").strip()
        if not learn_raw:
            raise ValueError("GLM_LEARN_DIR is required when GLM_LEARN_ENABLED=1")
        learn_dir = _validate_abs_clean_path(learn_raw, "GLM_LEARN_DIR")
        if learn_dir == Path("/") or learn_dir == hf_home:
            raise ValueError(f"GLM_LEARN_DIR must be a dedicated store directory, not {learn_dir}")
        if any(part.lower() in FORBIDDEN_LEARN_NAMES for part in learn_dir.parts):
            raise ValueError(f"GLM_LEARN_DIR must not reuse a legacy spill path: {learn_dir}")
        if learn_root is not None:
            try:
                rel = learn_dir.relative_to(learn_root)
            except ValueError as err:
                raise ValueError(f"GLM_LEARN_DIR {learn_dir} must live under GLM_LEARN_ROOT {learn_root}") from err
            if not rel.parts:
                raise ValueError(f"GLM_LEARN_DIR must be a dedicated subdirectory under GLM_LEARN_ROOT {learn_root}")
        learn_gib = _int_env(env, "GLM_LEARN_GIB", 64, minimum=1, maximum=1024)
        learn_min_free_gib = _int_env(env, "GLM_LEARN_MIN_FREE_GIB", 50, minimum=1, maximum=1024)
        learn_daily_gib = _int_env(env, "GLM_LEARN_DAILY_GIB", 128, minimum=1, maximum=4096)
        if verify_mounts:
            # Also apply /, HF_HOME, and FORBIDDEN_LEARN_NAMES checks to the resolved target path so a symlinked
            # ancestor cannot redirect persistence even when GLM_LEARN_ROOT is unset.
            resolved_learn = (
                learn_dir.resolve()
                if (learn_dir.exists() or learn_dir.is_symlink())
                else (learn_dir.parent.resolve() / learn_dir.name)
            )
            resolved_hf = hf_home.resolve()
            if resolved_learn == Path("/") or resolved_learn == resolved_hf:
                raise ValueError(
                    f"GLM_LEARN_DIR {learn_dir} resolves to forbidden directory {resolved_learn}"
                )
            if any(part.lower() in FORBIDDEN_LEARN_NAMES for part in resolved_learn.parts):
                raise ValueError(
                    f"GLM_LEARN_DIR {learn_dir} resolves to a forbidden legacy spill path: {resolved_learn}"
                )
        if verify_mounts and rank == 0:
            if learn_dir.is_symlink():
                raise ValueError(f"GLM_LEARN_DIR must not be a symlink: {learn_dir}")
            if not learn_dir.exists():
                parent = learn_dir.parent
                if not parent.is_dir():
                    raise ValueError(f"GLM_LEARN_DIR parent mount does not exist on rank 0: {parent}")
                if learn_root is not None:
                    try:
                        parent.resolve().relative_to(learn_root.resolve())
                    except ValueError as err:
                        raise ValueError(
                            f"GLM_LEARN_DIR parent {parent} resolves outside GLM_LEARN_ROOT {learn_root}"
                        ) from err
                learn_dir.mkdir(parents=False, exist_ok=True)
            if learn_root is not None:
                try:
                    rel_resolved = learn_dir.resolve().relative_to(learn_root.resolve())
                except ValueError as err:
                    raise ValueError(
                        f"GLM_LEARN_DIR {learn_dir} resolves outside GLM_LEARN_ROOT {learn_root}"
                    ) from err
                if not rel_resolved.parts:
                    raise ValueError(f"GLM_LEARN_DIR resolves to GLM_LEARN_ROOT itself: {learn_root}")
            if not learn_dir.is_dir() or not os.access(learn_dir, os.W_OK | os.X_OK):
                raise ValueError(f"GLM_LEARN_DIR is not a writable directory on rank 0: {learn_dir}")

    return MountContract(
        hf_home=hf_home,
        runtime_inputs_dir=runtime_inputs_dir,
        model_path=model_path,
        drafter=drafter,
        chat_template=chat_template,
        learn_enabled=learn_enabled,
        learn_dir=learn_dir,
        learn_root=learn_root,
        learn_gib=learn_gib,
        learn_min_free_gib=learn_min_free_gib,
        learn_daily_gib=learn_daily_gib,
    )


def _validate_rdma(
    env: Mapping[str, str],
    *,
    world: int,
    sysfs_ib_dir: Path | None,
) -> RdmaConfig:
    roce_enabled = _bool_env(env, "TF_ROCE", world > 1)
    rdma_required = _bool_env(env, "RDMA_REQUIRED", world > 1 and roce_enabled)
    verify_sysfs = _bool_env(env, "RDMA_VERIFY_SYSFS", True)
    ib_dir = sysfs_ib_dir if verify_sysfs else None

    _, raw_hca = _first_set(env, "TF_ROCE_HCA", "NCCL_IB_HCA")
    hcas = parse_rdma_hcas(raw_hca or "", sysfs_ib_dir=ib_dir) if (roce_enabled or raw_hca) else ()
    if rdma_required and not hcas:
        raise ValueError(
            "Active RDMA HCAs are required (set TF_ROCE_HCA or NCCL_IB_HCA, or set TF_ROCE=0 / RDMA_REQUIRED=0)"
        )

    gid_name, gid_raw = _first_set(env, "TF_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX")
    gid_index: int | None = None
    if gid_raw is not None:
        gid_index = _int_env({gid_name: gid_raw}, gid_name, 0, minimum=0, maximum=2_147_483_647)

    max_kb = _int_env(env, "TF_ROCE_MAX_KB", 4096, minimum=1, maximum=4_194_303)
    min_ch = _int_env(env, "NCCL_MIN_NCHANNELS", 4, minimum=1, maximum=64)
    max_ch = _int_env(env, "NCCL_MAX_NCHANNELS", 4, minimum=1, maximum=64)
    if min_ch > max_ch:
        raise ValueError(f"NCCL_MIN_NCHANNELS ({min_ch}) cannot exceed NCCL_MAX_NCHANNELS ({max_ch})")

    ifname_raw = (env.get("NCCL_SOCKET_IFNAME") or "").strip()
    socket_ifname: str | None = None
    if ifname_raw:
        if not IFNAME_RE.fullmatch(ifname_raw):
            raise ValueError(f"Invalid NCCL_SOCKET_IFNAME value: {ifname_raw!r}")
        socket_ifname = ifname_raw

    return RdmaConfig(
        enabled=roce_enabled and bool(hcas),
        hcas=hcas,
        gid_index=gid_index,
        max_kb=max_kb,
        socket_ifname=socket_ifname,
        min_nchannels=min_ch,
        max_nchannels=max_ch,
    )


def build_launch_spec(
    env: Mapping[str, str],
    *,
    binary: str = DEFAULT_BINARY,
    verify_mounts: bool = True,
    sysfs_ib_dir: Path | None = Path("/sys/class/infiniband"),
    resolve_host: Callable[[str], str] | None = None,
) -> LaunchSpec:
    """Validate runtime env and build the exact TensorFold command and environment for rank 0 or rank 1+."""
    out_env = dict(env)

    _, world_raw = _first_set(env, "WORLD_SIZE", "LWS_GROUP_SIZE", "TF_GLM_WORLD")
    world = _int_env({"WORLD_SIZE": world_raw or "2"}, "WORLD_SIZE", 2, minimum=1, maximum=64)

    _, rank_raw = _first_set(env, "RANK", "LWS_WORKER_INDEX", "TF_GLM_RANK")
    if rank_raw is None:
        raise ValueError("Rank is required (set RANK, LWS_WORKER_INDEX, or TF_GLM_RANK)")
    rank = _int_env({"RANK": rank_raw}, "RANK", 0, minimum=0, maximum=world - 1)

    launch_style = (env.get("TF_TP_LAUNCH_STYLE") or "flags").strip().lower()
    if launch_style not in VALID_LAUNCH_STYLES:
        raise ValueError(f"TF_TP_LAUNCH_STYLE must be one of {sorted(VALID_LAUNCH_STYLES)}, got {launch_style!r}")

    master_raw: str | None = None
    master_ipv4: str | None = None
    master_port: int | None = None
    nccl_port: int | None = None
    if world > 1:
        _, master_raw = _first_set(env, "FABRIC_MASTER", "LWS_LEADER_ADDRESS", "MASTER_ADDR", "TF_TP_MASTER")
        if not master_raw:
            raise ValueError(
                "Master fabric address is required for TP>1 "
                "(set FABRIC_MASTER, LWS_LEADER_ADDRESS, MASTER_ADDR, or TF_TP_MASTER)"
            )
        master_ipv4 = resolve_master_ipv4(master_raw, resolver=resolve_host)
        port_name, port_raw = _first_set(env, "MASTER_PORT", "TF_TP_PORT")
        if port_raw is None:
            master_port = 25000
        else:
            master_port = _int_env({port_name: port_raw}, port_name, 25000, minimum=2, maximum=65535)
        # SparkRun allocates MASTER_PORT for the TP liveness socket (tp_watch listens on NCCL_COMM_ID port + 1),
        # so default offset is -1. Kubernetes LWS deployments that pass the NCCL port directly can set 0.
        port_offset = _int_env(env, "TF_TP_PORT_OFFSET", -1, minimum=-1, maximum=0)
        nccl_port = master_port + port_offset
        if not 1 <= nccl_port <= 65534:
            raise ValueError(
                f"Derived NCCL bootstrap port {nccl_port} (from MASTER_PORT={master_port} + offset {port_offset}) "
                "must lie in 1..65534 to leave port + 1 for tp_watch"
            )

    mounts = _validate_mounts(env, rank=rank, verify_mounts=verify_mounts)
    rdma = _validate_rdma(env, world=world, sysfs_ib_dir=sysfs_ib_dir)

    context = _int_env(env, "CONTEXT", 1_048_576, minimum=1024, maximum=4_194_304)
    parallel = _int_env(env, "PARALLEL", 4, minimum=1, maximum=64)
    max_tokens = _int_env(env, "MAX_TOKENS", 32_768, minimum=1, maximum=context)
    prompt_cache_gib = _int_env(env, "PROMPT_CACHE_GIB", 6, minimum=0, maximum=1024)
    drafter_bits = _int_env(env, "DRAFTER_BITS", 4, minimum=2, maximum=16)
    top_k = _int_env(env, "TOP_K", 20, minimum=0, maximum=1000)
    vision_enabled = _bool_env(env, "VISION_ENABLED", True)
    vision_max_images = _int_env(env, "VISION_MAX_IMAGES", 50, minimum=1, maximum=500)
    vision_video_tokens_raw = (env.get("VISION_VIDEO_TOKENS") or "").strip()
    vision_video_tokens = (
        _int_env(
            {"VISION_VIDEO_TOKENS": vision_video_tokens_raw},
            "VISION_VIDEO_TOKENS",
            64,
            minimum=64,
            maximum=262_144,
        )
        if vision_video_tokens_raw
        else None
    )
    thinking_enabled = _bool_env(env, "THINKING_ENABLED", True)
    dashboard_enabled = _bool_env(env, "DASHBOARD_ENABLED", True)

    kv_cache = (env.get("KV_CACHE") or "fp8").strip().lower()
    if kv_cache not in VALID_KV_CACHES:
        raise ValueError(f"KV_CACHE must be one of {sorted(VALID_KV_CACHES)}, got {kv_cache!r}")

    kv_pages = (env.get("TF_KV_PAGES") or "").strip().lower()
    if kv_pages and kv_pages not in VALID_KV_PAGES:
        raise ValueError(f"TF_KV_PAGES must be one of {sorted(VALID_KV_PAGES)}, got {kv_pages!r}")

    clear_thinking = (env.get("TF_GLM_CLEAR_THINKING") or "0").strip()
    if clear_thinking not in ("0", "1"):
        raise ValueError(f"TF_GLM_CLEAR_THINKING must be '0' or '1', got {clear_thinking!r}")

    kv_share_raw = (env.get("TF_KV_SHARE") or "").strip()
    kv_share = _int_env({"TF_KV_SHARE": kv_share_raw or "0"}, "TF_KV_SHARE", 0, minimum=0, maximum=4096)
    if kv_share > 0:
        if mounts.learn_enabled:
            raise ValueError(
                "TF_KV_SHARE>0 is incompatible with --learn-conversations (GLM_LEARN_ENABLED=1)"
            )
        if (env.get("TF_GLM_REQUEST_STARTS") or "").strip() == "1":
            raise ValueError(
                "TF_KV_SHARE>0 is incompatible with TF_GLM_REQUEST_STARTS=1"
            )

    reasoning_effort = (env.get("REASONING_EFFORT") or "xhigh").strip().lower()
    if reasoning_effort not in VALID_EFFORTS:
        raise ValueError(f"REASONING_EFFORT must be one of {sorted(VALID_EFFORTS)}, got {reasoning_effort!r}")

    host = (env.get("HOST") or "0.0.0.0").strip()
    if not host or any(ch.isspace() for ch in host):
        raise ValueError(f"HOST must be a non-empty bind address without whitespace, got {host!r}")
    port = _int_env(env, "PORT", 8888, minimum=1, maximum=65535)
    if rank == 0 and world > 1 and nccl_port is not None and port in (nccl_port, nccl_port + 1):
        raise ValueError(
            f"HTTP PORT ({port}) collides with TP rendezvous/watch ports ({nccl_port}, {nccl_port + 1})"
        )
    served_name = (env.get("SERVED_NAME") or "GLM-5.3-Flash").strip()
    if not served_name:
        raise ValueError("SERVED_NAME must be non-empty")

    extra_raw = (env.get("EXTRA_ARGS") or "").strip()
    extra_args: list[str] = []
    if extra_raw:
        if not _bool_env(env, "ALLOW_EXTRA_ARGS", False):
            raise ValueError("EXTRA_ARGS is refused by default; record a profile or set ALLOW_EXTRA_ARGS=1 explicitly")
        extra_args = shlex.split(extra_raw)
        for token in extra_args:
            flag = token.split("=", 1)[0]
            if flag in PROTECTED_EXTRA_FLAGS:
                raise ValueError(f"EXTRA_ARGS cannot override contract-managed flag {flag!r}")

    # Populate normalized child environment.
    out_env["HF_HOME"] = str(mounts.hf_home)
    out_env.setdefault("HF_HUB_OFFLINE", "1")
    out_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    out_env.setdefault("TENSORFOLD_NO_UPDATE_CHECK", "1")
    out_env.setdefault("TENSORFOLD_NO_LIVE", "1")
    out_env.setdefault("TF_CUDA_DEVICE", "0")
    out_env.setdefault("NCCL_RUNTIME_CONNECT", "0")
    out_env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
    out_env.setdefault("NCCL_CUMEM_ENABLE", "0")
    out_env["TF_GLM_CLEAR_THINKING"] = clear_thinking
    if kv_pages:
        out_env["TF_KV_PAGES"] = kv_pages
    out_env["NCCL_MIN_NCHANNELS"] = str(rdma.min_nchannels)
    out_env["NCCL_MAX_NCHANNELS"] = str(rdma.max_nchannels)

    if rdma.enabled:
        hca_csv = ",".join(rdma.hcas)
        out_env["TF_ROCE"] = "1"
        out_env["TF_ROCE_HCA"] = hca_csv
        out_env["TF_ROCE_MAX_KB"] = str(rdma.max_kb)
        out_env.setdefault("NCCL_IB_HCA", hca_csv)
        if rdma.gid_index is not None:
            out_env["TF_ROCE_GID_INDEX"] = str(rdma.gid_index)
            out_env["NCCL_IB_GID_INDEX"] = str(rdma.gid_index)
    else:
        out_env["TF_ROCE"] = "0"
    if rdma.socket_ifname is not None:
        out_env["NCCL_SOCKET_IFNAME"] = rdma.socket_ifname

    if world > 1 and master_ipv4 is not None and nccl_port is not None:
        want_comm_id = f"{master_ipv4}:{nccl_port}"
        existing_comm_id = (env.get("NCCL_COMM_ID") or "").strip()
        if existing_comm_id and existing_comm_id != want_comm_id:
            raise ValueError(
                f"NCCL_COMM_ID ({existing_comm_id!r}) disagrees with derived master rendezvous ({want_comm_id!r}); "
                "unset NCCL_COMM_ID or pass matching values"
            )
        out_env["NCCL_COMM_ID"] = want_comm_id

    argv: list[str] = [
        binary,
        "serve",
        mounts.model_path,
        "--name",
        served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--context",
        str(context),
        "--parallel",
        str(parallel),
        "--max-tokens",
        str(max_tokens),
        "--kv-cache",
        kv_cache,
    ]

    if launch_style == "flags":
        argv += ["--tp", str(world), "--rank", str(rank)]
        if world > 1 and master_ipv4 is not None and nccl_port is not None:
            argv += ["--master", master_ipv4, "--master-port", str(nccl_port)]
        # Avoid retired alias warnings/errors in TensorFold >=1.4.0 when launching with canonical flags.
        out_env.pop("TF_GLM_RANK", None)
        out_env.pop("TF_GLM_WORLD", None)
    else:
        out_env["TF_GLM_RANK"] = str(rank)
        out_env["TF_GLM_WORLD"] = str(world)

    if thinking_enabled:
        argv += ["--thinking", "--reasoning-effort", reasoning_effort]
    else:
        argv.append("--no-thinking")
    if top_k > 0:
        argv += ["--top-k", str(top_k)]
    argv.append("--no-update-check")
    if dashboard_enabled:
        argv.append("--dashboard")
    if vision_enabled:
        argv += ["--vision", "--vision-max-images", str(vision_max_images)]
        if vision_video_tokens is not None:
            argv += ["--vision-video-tokens", str(vision_video_tokens)]
    if mounts.chat_template is not None:
        argv += ["--chat-template", str(mounts.chat_template)]
    if mounts.drafter is not None:
        argv += ["--drafter", mounts.drafter, "--drafter-bits", str(drafter_bits)]
    else:
        argv.append("--no-drafts")
    argv += ["--prompt-cache-gib", str(prompt_cache_gib)]
    if mounts.learn_enabled and mounts.learn_dir is not None:
        argv += [
            "--learn-conversations",
            "--learn-dir",
            str(mounts.learn_dir),
            "--learn-gib",
            str(mounts.learn_gib),
            "--learn-min-free-gib",
            str(mounts.learn_min_free_gib),
            "--learn-daily-gib",
            str(mounts.learn_daily_gib),
        ]
    argv += extra_args

    return LaunchSpec(
        rank=rank,
        world=world,
        master_raw=master_raw,
        master_ipv4=master_ipv4,
        master_port=master_port,
        nccl_port=nccl_port,
        launch_style=launch_style,
        mounts=mounts,
        rdma=rdma,
        argv=tuple(argv),
        env=out_env,
    )
