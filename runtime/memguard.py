"""DGX Spark unified-memory guardrail (MemAvailable + parked per-CPU pages, plus cgroup v2 cap awareness).

Adapted in spirit from the host-memory accounting used in Mia's scripts/memavail.sh and home-lab's native GLM guard:
reads /proc/meminfo first, then /proc/zoneinfo so a drained per-CPU page is never double-counted, caps the sum at
MemTotal, and also respects finite cgroup v2 memory limits when running under Kubernetes.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

GIB = 1024**3
DEFAULT_FLOOR_GIB = 2.5
DEFAULT_SUSTAIN_SECONDS = 4.0
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_OOM_SCORE_ADJ = 1000
MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class MemSample:
    """One point-in-time memory reading."""

    mem_available_gib: float
    parked_gib: float
    host_effective_gib: float
    cgroup_available_gib: float | None
    effective_gib: float


def parked_bytes(zoneinfo_text: str, page_size: int) -> int:
    """Free per-CPU pages held above resting high_min (512 MiB conservative default on kernels without high_min)."""
    if page_size <= 0:
        return 0
    count, rest, in_pagesets = 0, 0, False
    for row in zoneinfo_text.splitlines():
        words = row.split()
        if row.strip() == "pagesets":
            in_pagesets = True
        elif in_pagesets and not row.startswith(("    ", "  vm stats")):
            in_pagesets = False
        elif in_pagesets and words and words[0] in ("count:", "high_min:"):
            if len(words) < 2 or not words[1].isascii() or not words[1].isdigit():
                return 0
            val = int(words[1])
            if words[0] == "count:":
                count += val
            else:
                rest += val
            if max(count, rest) > MAX_U64:
                return 0
    held = count * page_size
    resting = rest * page_size if rest else 512 * 1024**2
    if max(held, resting) > MAX_U64:
        return 0
    return max(0, held - resting)


def _parse_cgroup_v2_path(proc_cgroup_text: str) -> str | None:
    """Extract the cgroup v2 relative path from /proc/self/cgroup (the '0::/...' line)."""
    for line in proc_cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            return parts[2].strip() or "/"
    return None


def _read_cgroup_dir_available_bytes(cg_dir: Path) -> int | None:
    try:
        max_text = (cg_dir / "memory.max").read_text().strip()
        if not max_text or max_text == "max" or not max_text.isascii() or not max_text.isdigit():
            return None
        cur_text = (cg_dir / "memory.current").read_text().strip()
        if not cur_text or not cur_text.isascii() or not cur_text.isdigit():
            return None
    except OSError:
        return None
    limit_bytes = int(max_text)
    current_bytes = int(cur_text)
    if limit_bytes <= 0:
        return None
    return max(0, limit_bytes - current_bytes)


def cgroup_available_gib(
    memory_max: Path = Path("/sys/fs/cgroup/memory.max"),
    memory_current: Path = Path("/sys/fs/cgroup/memory.current"),
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> float | None:
    """Minimum available GiB across visible finite cgroup v2 memory caps (leaf + ancestors), or None."""
    headrooms: list[int] = []

    # Direct paths (supports explicit overrides and flat cgroup-namespace mounts).
    try:
        max_text = memory_max.read_text().strip()
        if max_text and max_text != "max" and max_text.isascii() and max_text.isdigit():
            cur_text = memory_current.read_text().strip()
            if cur_text and cur_text.isascii() and cur_text.isdigit():
                limit_bytes = int(max_text)
                if limit_bytes > 0:
                    headrooms.append(max(0, limit_bytes - int(cur_text)))
    except OSError:
        pass

    # Walk the process cgroup hierarchy from leaf up to cgroup_root so a tighter parent cap is never missed.
    try:
        rel_raw = _parse_cgroup_v2_path(proc_cgroup.read_text())
    except OSError:
        rel_raw = None
    if rel_raw is not None and cgroup_root.is_dir():
        rel_clean = rel_raw.lstrip("/")
        if rel_clean and ".." not in Path(rel_clean).parts:
            cur = cgroup_root / rel_clean
            root_resolved = cgroup_root.resolve()
            while True:
                if cur.is_dir():
                    avail = _read_cgroup_dir_available_bytes(cur)
                    if avail is not None:
                        headrooms.append(avail)
                if cur == cgroup_root or cur.resolve() == root_resolved or cur.parent == cur:
                    break
                cur = cur.parent

    if not headrooms:
        return None
    return min(headrooms) / GIB


def sample_memory(
    meminfo_path: Path | str = Path("/proc/meminfo"),
    zoneinfo_path: Path | str = Path("/proc/zoneinfo"),
    *,
    page_size: int | None = None,
    cgroup_max_path: Path | str = Path("/sys/fs/cgroup/memory.max"),
    cgroup_current_path: Path | str = Path("/sys/fs/cgroup/memory.current"),
    proc_cgroup_path: Path | str = Path("/proc/self/cgroup"),
    cgroup_root_path: Path | str = Path("/sys/fs/cgroup"),
) -> MemSample:
    """Read /proc/meminfo first, then /proc/zoneinfo and cgroup v2 limits, returning effective available GiB."""
    memory: dict[str, int] = {}
    for line in Path(meminfo_path).read_text().splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] in ("MemAvailable:", "MemTotal:"):
            if not words[1].isascii() or not words[1].isdigit():
                raise ValueError(f"Malformed {words[0]} line in {meminfo_path}")
            memory[words[0]] = int(words[1]) * 1024
    if "MemAvailable:" not in memory:
        raise ValueError(f"MemAvailable missing from {meminfo_path}")

    resolved_page = page_size if page_size is not None else os.sysconf("SC_PAGE_SIZE")
    try:
        parked = parked_bytes(Path(zoneinfo_path).read_text(), resolved_page)
    except OSError:
        parked = 0

    host_raw = memory["MemAvailable:"] + parked
    total = memory.get("MemTotal:", 0)
    host_effective = min(total, host_raw) if total > 0 else host_raw

    cg_avail = cgroup_available_gib(
        Path(cgroup_max_path),
        Path(cgroup_current_path),
        proc_cgroup=Path(proc_cgroup_path),
        cgroup_root=Path(cgroup_root_path),
    )
    host_gib = host_effective / GIB
    effective = min(host_gib, cg_avail) if cg_avail is not None else host_gib
    return MemSample(
        mem_available_gib=memory["MemAvailable:"] / GIB,
        parked_gib=parked / GIB,
        host_effective_gib=host_gib,
        cgroup_available_gib=cg_avail,
        effective_gib=effective,
    )


def available_memory_gib(
    meminfo_path: Path | str = Path("/proc/meminfo"),
    zoneinfo_path: Path | str = Path("/proc/zoneinfo"),
    *,
    page_size: int | None = None,
    cgroup_max_path: Path | str = Path("/sys/fs/cgroup/memory.max"),
    cgroup_current_path: Path | str = Path("/sys/fs/cgroup/memory.current"),
) -> float:
    """Convenience wrapper returning effective available GiB."""
    return sample_memory(
        meminfo_path,
        zoneinfo_path,
        page_size=page_size,
        cgroup_max_path=cgroup_max_path,
        cgroup_current_path=cgroup_current_path,
    ).effective_gib


def apply_oom_score_adj(
    score: int = DEFAULT_OOM_SCORE_ADJ,
    *,
    path: Path | str = Path("/proc/self/oom_score_adj"),
    strict: bool = False,
) -> bool:
    """Prefer killing the engine child over host system daemons if kernel OOM fires."""
    if not -1000 <= score <= 1000:
        raise ValueError(f"oom_score_adj must be in -1000..1000, got {score}")
    try:
        Path(path).write_text(f"{score}\n")
        return True
    except OSError:
        if strict:
            raise
        return False


class MemGuard:
    """Stateful low-memory evaluator requiring sustained floor violation before triggering stop."""

    def __init__(
        self,
        *,
        floor_gib: float = DEFAULT_FLOOR_GIB,
        sustain_s: float = DEFAULT_SUSTAIN_SECONDS,
        sampler: Callable[[], MemSample] = sample_memory,
    ):
        if floor_gib <= 0:
            raise ValueError(f"floor_gib must be positive, got {floor_gib}")
        if sustain_s < 0:
            raise ValueError(f"sustain_s must be non-negative, got {sustain_s}")
        self.floor_gib = floor_gib
        self.sustain_s = sustain_s
        self.sampler = sampler
        self.low_since: float | None = None
        self.last_sample: MemSample | None = None

    def check(self, now_monotonic: float) -> tuple[str, str | None]:
        """Evaluate current memory pressure.

        Returns:
          ("ok", None) when above floor or below floor for less than sustain_s
          ("low_memory", reason) when below floor for >= sustain_s (exit 73)
          ("sample_error", reason) when telemetry cannot be read/parsed (exit 72)
        """
        try:
            sample = self.sampler()
        except (OSError, ValueError) as error:
            return "sample_error", f"cannot sample memory telemetry: {error}"
        self.last_sample = sample
        if sample.effective_gib < self.floor_gib:
            if self.low_since is None:
                self.low_since = now_monotonic
            elapsed = now_monotonic - self.low_since
            if elapsed >= self.sustain_s:
                return (
                    "low_memory",
                    f"{sample.effective_gib:.2f} GiB effective available "
                    f"(host {sample.host_effective_gib:.2f} GiB"
                    + (
                        f", cgroup {sample.cgroup_available_gib:.2f} GiB"
                        if sample.cgroup_available_gib is not None
                        else ""
                    )
                    + f"), below {self.floor_gib:.2f} GiB floor for {elapsed:.1f}s",
                )
            return "ok", None
        self.low_since = None
        return "ok", None
