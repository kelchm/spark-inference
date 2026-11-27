"""PID-1 container entrypoint for TensorFold TP ranks (head rank 0 and follower rank 1+).

Responsibilities:
- Validates the 3-way mount contract and parameterized RDMA/fabric env before launching the engine.
- Runs as a clean PID-1 supervisor: reaps orphaned child processes, forwards SIGHUP (key reload), and publishes an
  atomic secret-free state file for local worker readiness/liveness probes.
- Handles Rank-0 shutdown (immediate SIGTERM forwarding for HTTP drain + disk flush, bounded grace before SIGKILL).
- Handles Rank-1+ orderly SIGTERM flush: when the orchestrator (SparkRun/Docker or Kubernetes LeaderWorkerSet) sends
  SIGTERM to both ranks concurrently, rank 1 holds off signaling the follower engine for RANK1_FLUSH_GRACE_SECONDS so
  rank 0 can gather rank 1's pages over the TP link, commit pending --learn-conversations writes to disk, and end the
  follower cleanly (exit 0). A second stop signal or flush timeout escalates cleanly to SIGTERM -> SIGKILL.
- Runs the DGX Spark unified-memory MemGuard loop (exit 73 on sustained breach, exit 72 on unreadable telemetry).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from .contract import DEFAULT_BINARY, LaunchSpec, build_launch_spec, redact_secrets, sanitize_argv
from .health import DEFAULT_STATE_FILE
from .memguard import (
    DEFAULT_FLOOR_GIB,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_OOM_SCORE_ADJ,
    DEFAULT_SUSTAIN_SECONDS,
    MemGuard,
    apply_oom_score_adj,
    sample_memory,
)

# Rank 1's flush fallback MUST exceed rank 0's full drain+flush grace plus orchestrator signal delivery skew.
# Configure outer Docker stop_grace_period / Kubernetes terminationGracePeriodSeconds >= 75s.
DEFAULT_SHUTDOWN_GRACE_S = 45.0
DEFAULT_RANK1_FLUSH_GRACE_S = 60.0
DEFAULT_RANK1_KILL_GRACE_S = 10.0
DEFAULT_MEMGUARD_STOP_GRACE_S = 10.0


@dataclass(frozen=True)
class SupervisorConfig:
    shutdown_grace_s: float = DEFAULT_SHUTDOWN_GRACE_S
    rank1_flush_grace_s: float = DEFAULT_RANK1_FLUSH_GRACE_S
    rank1_kill_grace_s: float = DEFAULT_RANK1_KILL_GRACE_S
    memguard_stop_grace_s: float = DEFAULT_MEMGUARD_STOP_GRACE_S
    memguard_enabled: bool = True
    memguard_floor_gib: float = DEFAULT_FLOOR_GIB
    memguard_sustain_s: float = DEFAULT_SUSTAIN_SECONDS
    memguard_interval_s: float = DEFAULT_INTERVAL_SECONDS
    oom_score_adj: int = DEFAULT_OOM_SCORE_ADJ
    oom_score_strict: bool = False
    rank1_sync_on_exit: bool = True
    state_file: Path = Path(DEFAULT_STATE_FILE)


def _float_env(env: Mapping[str, str], name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError as err:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from err
    if val < minimum or val != val or val == float("inf"):
        raise ValueError(f"{name} must be >= {minimum}, got {val}")
    return val


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be boolean, got {raw!r}")


def supervisor_config_from_env(env: Mapping[str, str]) -> SupervisorConfig:
    raw_oom = (env.get("MEMGUARD_OOM_SCORE_ADJ") or str(DEFAULT_OOM_SCORE_ADJ)).strip()
    try:
        oom = int(raw_oom)
    except ValueError as err:
        raise ValueError(f"MEMGUARD_OOM_SCORE_ADJ must be an integer, got {raw_oom!r}") from err
    if not -1000 <= oom <= 1000:
        raise ValueError(f"MEMGUARD_OOM_SCORE_ADJ must be in -1000..1000, got {oom}")
    return SupervisorConfig(
        shutdown_grace_s=_float_env(env, "SHUTDOWN_GRACE_SECONDS", DEFAULT_SHUTDOWN_GRACE_S, minimum=0.1),
        rank1_flush_grace_s=_float_env(env, "RANK1_FLUSH_GRACE_SECONDS", DEFAULT_RANK1_FLUSH_GRACE_S, minimum=0.0),
        rank1_kill_grace_s=_float_env(env, "RANK1_KILL_GRACE_SECONDS", DEFAULT_RANK1_KILL_GRACE_S, minimum=0.1),
        memguard_stop_grace_s=_float_env(
            env, "MEMGUARD_STOP_GRACE_SECONDS", DEFAULT_MEMGUARD_STOP_GRACE_S, minimum=0.1
        ),
        memguard_enabled=_bool_env(env, "MEMGUARD_ENABLED", True),
        memguard_floor_gib=_float_env(env, "MEMGUARD_FLOOR_GIB", DEFAULT_FLOOR_GIB, minimum=0.05),
        memguard_sustain_s=_float_env(env, "MEMGUARD_SUSTAIN_SECONDS", DEFAULT_SUSTAIN_SECONDS, minimum=0.0),
        memguard_interval_s=_float_env(env, "MEMGUARD_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS, minimum=0.01),
        oom_score_adj=oom,
        oom_score_strict=_bool_env(env, "MEMGUARD_OOM_SCORE_STRICT", False),
        rank1_sync_on_exit=_bool_env(env, "RANK1_SYNC_ON_EXIT", True),
        state_file=Path((env.get("TF_RUNTIME_STATE_FILE") or DEFAULT_STATE_FILE).strip() or DEFAULT_STATE_FILE),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(
    path: Path,
    *,
    rank: int,
    world: int,
    child_pid: int | None,
    phase: str,
    started_at: str,
    detail: str | None = None,
) -> None:
    """Atomically write secret-free supervisor state for local probes."""
    payload: dict[str, Any] = {
        "rank": rank,
        "world": world,
        "pid": os.getpid(),
        "child_pid": child_pid,
        "phase": phase,
        "started_at": started_at,
        "updated_at": _now_iso(),
    }
    if detail:
        payload["detail"] = detail
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass


def _decode_popen_returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return 1


def reap_other_zombies(child: subprocess.Popen[Any]) -> int | None:
    """Reap orphaned zombie grandchildren (PID-1 duty) without losing the tracked child's exit code."""
    child_rc: int | None = None
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        if pid == child.pid:
            child_rc = _decode_popen_returncode(status)
            child.returncode = child_rc
    return child_rc


def _signal_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def stop_process_group(
    child: subprocess.Popen[Any],
    *,
    sig: int = signal.SIGTERM,
    grace_s: float = DEFAULT_SHUTDOWN_GRACE_S,
    poll_interval_s: float = 0.05,
    stop_signals: list[int] | None = None,
    initial_signal_count: int = 1,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Send sig to the child's process group, forward a second stop signal if it arrives, then SIGKILL and reap."""
    _signal_group(child.pid, sig)
    seen_signals = initial_signal_count
    forwarded_second = False
    deadline = clock() + grace_s
    while child.poll() is None and clock() < deadline:
        reap_other_zombies(child)
        if child.poll() is not None:
            break
        remaining = max(0.0, deadline - clock())
        sleeper(min(poll_interval_s, remaining))
        if child.poll() is not None:
            break
        if stop_signals is not None and not forwarded_second and len(stop_signals) > seen_signals:
            _signal_group(child.pid, stop_signals[seen_signals])
            forwarded_second = True
    # Always SIGKILL the private process group even if the leader already exited, so no descendant survives.
    _signal_group(child.pid, signal.SIGKILL)
    if child.poll() is None:
        child.wait()
    reap_other_zombies(child)
    rc = child.returncode
    if rc is None:
        return 1
    return rc if rc >= 0 else 128 - rc


def _fsync_after_flush(learn_dir: Path | None, sync_fn: Callable[[], None] | None) -> None:
    """Flush OS filesystem buffers after clean follower exit so disk-store blocks settle safely."""
    if sync_fn is not None:
        sync_fn()
        return
    if learn_dir is not None and learn_dir.is_dir():
        try:
            fd = os.open(str(learn_dir), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    if hasattr(os, "sync"):
        try:
            os.sync()
        except OSError:
            pass


def supervise(
    spec: LaunchSpec,
    config: SupervisorConfig | None = None,
    *,
    mem_guard: MemGuard | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    sync_fn: Callable[[], None] | None = None,
    install_signals: bool = True,
    external_signals: list[int] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, flush=True),
) -> int:
    """Launch and supervise the TensorFold engine child for either rank 0 or rank 1+."""
    cfg = config or SupervisorConfig()
    guard = (
        mem_guard
        if mem_guard is not None
        else (
            MemGuard(
                floor_gib=cfg.memguard_floor_gib,
                sustain_s=cfg.memguard_sustain_s,
                sampler=sample_memory,
            )
            if cfg.memguard_enabled
            else None
        )
    )

    stop_signals: list[int] = external_signals if external_signals is not None else []
    hup_pending: list[bool] = []
    previous_handlers: dict[int, Any] = {}

    if install_signals:
        def _on_stop(signum: int, _frame: Any) -> None:
            stop_signals.append(signum)

        def _on_hup(_signum: int, _frame: Any) -> None:
            hup_pending.append(True)

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, _on_stop)
        if hasattr(signal, "SIGHUP"):
            previous_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, _on_hup)

    started_at = _now_iso()
    write_state(cfg.state_file, rank=spec.rank, world=spec.world, child_pid=None, phase="starting", started_at=started_at)

    def _preexec() -> None:
        os.setsid()
        apply_oom_score_adj(cfg.oom_score_adj, strict=cfg.oom_score_strict)

    child = subprocess.Popen(list(spec.argv), env=spec.env, preexec_fn=_preexec)
    write_state(
        cfg.state_file,
        rank=spec.rank,
        world=spec.world,
        child_pid=child.pid,
        phase="running",
        started_at=started_at,
    )
    log(
        f"[entrypoint] rank {spec.rank}/{spec.world} started pid={child.pid} "
        f"(roce={'on' if spec.rdma.enabled else 'off'}, hcas={','.join(spec.rdma.hcas) or 'none'})"
    )

    exit_code = 1
    try:
        while child.poll() is None:
            reap_other_zombies(child)
            if child.poll() is not None:
                break

            if hup_pending:
                hup_pending.clear()
                if hasattr(signal, "SIGHUP"):
                    _signal_group(child.pid, signal.SIGHUP)
                    log(f"[entrypoint] rank {spec.rank} forwarded SIGHUP to pid={child.pid}")

            if stop_signals:
                first_sig = stop_signals[0]
                if spec.rank == 0:
                    write_state(
                        cfg.state_file,
                        rank=spec.rank,
                        world=spec.world,
                        child_pid=child.pid,
                        phase="draining",
                        started_at=started_at,
                        detail=f"signal {first_sig}",
                    )
                    log(
                        f"[entrypoint] rank 0 received signal {first_sig}; forwarding to pid={child.pid} "
                        f"(grace {cfg.shutdown_grace_s:.1f}s for drain and disk flush)"
                    )
                    rc = stop_process_group(
                        child,
                        sig=first_sig,
                        grace_s=cfg.shutdown_grace_s,
                        stop_signals=stop_signals,
                        clock=clock,
                        sleeper=sleeper,
                    )
                    exit_code = rc
                    return exit_code

                # Rank 1+ orderly SIGTERM flush: wait for rank 0 to gather state, commit disk writes,
                # and end this follower cleanly before sending SIGTERM directly to the follower engine.
                write_state(
                    cfg.state_file,
                    rank=spec.rank,
                    world=spec.world,
                    child_pid=child.pid,
                    phase="flushing",
                    started_at=started_at,
                    detail=f"waiting for rank 0 flush after signal {first_sig}",
                )
                log(
                    f"[entrypoint] rank {spec.rank} received signal {first_sig}; holding off follower stop for up to "
                    f"{cfg.rank1_flush_grace_s:.1f}s so rank 0 can flush disk cache writes cleanly"
                )
                flush_deadline = clock() + cfg.rank1_flush_grace_s
                cut_short = False
                while child.poll() is None and clock() < flush_deadline:
                    reap_other_zombies(child)
                    if child.poll() is not None:
                        break
                    if len(stop_signals) > 1:
                        cut_short = True
                        log(f"[entrypoint] rank {spec.rank} received second stop signal; ending flush wait early")
                        break
                    remaining = max(0.0, flush_deadline - clock())
                    sleeper(min(0.05, remaining))

                if child.poll() is not None:
                    rc = child.returncode if child.returncode is not None else 0
                    rc = rc if rc >= 0 else 128 - rc
                    if cfg.rank1_sync_on_exit:
                        _fsync_after_flush(spec.mounts.learn_dir, sync_fn)
                    log(f"[entrypoint] rank {spec.rank} exited cleanly during flush window (rc={rc})")
                    exit_code = rc
                    return exit_code

                reason = "second signal" if cut_short else f"flush grace {cfg.rank1_flush_grace_s:.1f}s expired"
                log(f"[entrypoint] rank {spec.rank} {reason}; forwarding signal {first_sig} to pid={child.pid}")
                rc = stop_process_group(
                    child,
                    sig=first_sig,
                    grace_s=cfg.rank1_kill_grace_s,
                    stop_signals=stop_signals,
                    clock=clock,
                    sleeper=sleeper,
                )
                if cfg.rank1_sync_on_exit:
                    _fsync_after_flush(spec.mounts.learn_dir, sync_fn)
                exit_code = rc
                return exit_code

            if guard is not None:
                status, reason = guard.check(clock())
                if status in ("low_memory", "sample_error"):
                    code = 73 if status == "low_memory" else 72
                    write_state(
                        cfg.state_file,
                        rank=spec.rank,
                        world=spec.world,
                        child_pid=child.pid,
                        phase="memguard_stop",
                        started_at=started_at,
                        detail=reason,
                    )
                    log(f"[memguard] rank {spec.rank}: {reason}; stopping rank (exit {code})")
                    stop_process_group(
                        child,
                        sig=signal.SIGTERM,
                        grace_s=cfg.memguard_stop_grace_s,
                        clock=clock,
                        sleeper=sleeper,
                    )
                    exit_code = code
                    return exit_code

            # Responsive wait slice so signals are noticed within 50ms while memguard polls at interval_s.
            wait_end = clock() + cfg.memguard_interval_s
            while child.poll() is None and not stop_signals and not hup_pending and clock() < wait_end:
                reap_other_zombies(child)
                if child.poll() is not None:
                    break
                remaining = max(0.0, wait_end - clock())
                sleeper(min(0.05, remaining))

        rc = child.returncode if child.returncode is not None else 1
        exit_code = rc if rc >= 0 else 128 - rc
        return exit_code
    finally:
        if child.poll() is None:
            stop_process_group(child, sig=signal.SIGTERM, grace_s=cfg.memguard_stop_grace_s, clock=clock, sleeper=sleeper)
        else:
            # Ensure no descendant in the child's private process group outlives the leader.
            _signal_group(child.pid, signal.SIGKILL)
        reap_other_zombies(child)
        write_state(
            cfg.state_file,
            rank=spec.rank,
            world=spec.world,
            child_pid=child.pid,
            phase="exited",
            started_at=started_at,
            detail=f"exit {exit_code}",
        )
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: tensorfold-entrypoint [--dry-run]")
        print("Validates the 3-way mount contract and RDMA/fabric env, then supervises tensorfold-native as PID 1.")
        raise SystemExit(0)
    dry_run = bool(argv and argv[0] == "--dry-run")
    try:
        spec = build_launch_spec(
            os.environ,
            binary=os.environ.get("TENSORFOLD_BINARY", DEFAULT_BINARY),
            verify_mounts=not dry_run,
        )
        cfg = supervisor_config_from_env(os.environ)
    except ValueError as err:
        print(f"[entrypoint] configuration error: {redact_secrets(str(err), env=os.environ)}", file=sys.stderr)
        raise SystemExit(2) from None
    if dry_run:
        summary = {
            "rank": spec.rank,
            "world": spec.world,
            "master_ipv4": spec.master_ipv4,
            "nccl_port": spec.nccl_port,
            "launch_style": spec.launch_style,
            "rdma_enabled": spec.rdma.enabled,
            "rdma_hcas": list(spec.rdma.hcas),
            "argv": sanitize_argv(spec.argv, env=os.environ),
        }
        print(json.dumps(summary, sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(supervise(spec, cfg))


if __name__ == "__main__":
    main()
