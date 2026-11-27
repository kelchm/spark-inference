"""Readiness, liveness, and capacity health wrapper for TensorFold containers.

Works identically for Docker HEALTHCHECK, SparkRun post_exec checks, and Kubernetes LeaderWorkerSet
startupProbe / readinessProbe / livenessProbe (both rank 0 HTTP and rank 1 worker state probes).
Never prints, logs, or echoes API keys or Authorization headers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import urllib.error
import urllib.request

from .contract import redact_secrets

DEFAULT_STATE_FILE = "/tmp/tensorfold-runtime-state.json"
POOL_FLOOR_4X1M = 2_582_528
POOL_LOG_RE = re.compile(
    r"(?:taking|all\s+\d+\s+on)\s+(\d+)\s+pages\s+of\s+(\d+)\s+rows",
    re.I,
)
STORE_LOG_RE = re.compile(
    r"glm:\s+conversation states on disk in\s+(.+?)\s+"
    r"\((\d+)\s+states,\s+(\d+)\s+MiB\s+of\s+(\d+)\s+MiB;\s+(\d+)\s+ranks'? parts(?:;\s*[^)]*)?\)"
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects when probing (prevents Authorization header leakage to another origin)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect refused ({code})", headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect())


def _redact_obj(value: Any, *, env: Mapping[str, str] | None, extra_secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return redact_secrets(value, env=env, extra_secrets=extra_secrets)
    if isinstance(value, dict):
        return {
            (redact_secrets(k, env=env, extra_secrets=extra_secrets) if isinstance(k, str) else k):
            _redact_obj(v, env=env, extra_secrets=extra_secrets)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_obj(v, env=env, extra_secrets=extra_secrets) for v in value]
    return value


def _fetch_json(url: str, *, timeout: float, api_key: str | None = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(1 << 20).decode("utf-8", errors="replace")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as err:
        raw = err.read(4096).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        return err.code, body


def extract_pool_tokens(health_body: Mapping[str, Any], startup_log: str | None = None) -> int | None:
    """Extract aggregate KV pool token capacity from /health or startup log lines."""
    if isinstance(health_body.get("pool_tokens"), int) and not isinstance(health_body["pool_tokens"], bool):
        return int(health_body["pool_tokens"])
    shard = health_body.get("kv_shard")
    if isinstance(shard, dict):
        parts = shard.get("parts")
        page_tokens = shard.get("page_tokens")
        part_pages = shard.get("part_pages")
        if all(type(v) is int and v > 0 for v in (parts, page_tokens, part_pages)):
            return parts * part_pages * page_tokens
    if startup_log:
        matches = POOL_LOG_RE.findall(startup_log)
        if matches:
            pages, rows = int(matches[-1][0]), int(matches[-1][1])
            return pages * rows
    return None


def evaluate_health(
    status_code: int,
    body: Any,
    *,
    mode: str = "ready",
    expected_model: str | None = None,
    expected_lanes: int | None = None,
    min_pool_tokens: int = 0,
    startup_log: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Judge /health response for readiness or liveness probes."""
    if not isinstance(body, dict):
        return False, {"ok": False, "http_status": status_code, "reason": "health response is not a JSON object"}

    state = body.get("status")
    if status_code == 503 and state == "draining":
        ok = mode == "live"
        return ok, {
            "ok": ok,
            "http_status": status_code,
            "status": "draining",
            "reason": None if ok else "server is draining",
        }
    if status_code != 200:
        return False, {"ok": False, "http_status": status_code, "status": state, "reason": f"HTTP {status_code}"}

    # Open health vs keyed minimal health ({"status": "ok"}) vs Python server health ({"ok": true}).
    if "ok" in body:
        if body["ok"] is not True:
            return False, {"ok": False, "http_status": status_code, "reason": "engine reported ok=false"}
        warming = False
        model = body.get("model")
        lanes = (body.get("streams") or {}).get("max") if isinstance(body.get("streams"), dict) else None
    else:
        if state != "ok":
            return False, {"ok": False, "http_status": status_code, "status": state, "reason": f"status={state!r}"}
        warming = body.get("warming")
        if warming is True and mode == "ready":
            return False, {"ok": False, "http_status": status_code, "warming": True, "reason": "engine is still warming"}
        model = body.get("model")
        lanes = body.get("max_batch_size")

    # Keyed servers return minimal {"status": "ok"} without model/lanes; only enforce model/lanes when reported.
    if expected_model is not None and model is not None and model != expected_model:
        return False, {
            "ok": False,
            "http_status": status_code,
            "model": model,
            "reason": f"model {model!r} does not match expected {expected_model!r}",
        }
    if expected_lanes is not None and lanes is not None and lanes != expected_lanes:
        return False, {
            "ok": False,
            "http_status": status_code,
            "max_batch_size": lanes,
            "reason": f"max_batch_size {lanes!r} does not match expected {expected_lanes}",
        }

    pool_tokens = extract_pool_tokens(body, startup_log)
    if min_pool_tokens > 0:
        if pool_tokens is None or pool_tokens < min_pool_tokens:
            return False, {
                "ok": False,
                "http_status": status_code,
                "pool_tokens": pool_tokens,
                "min_pool_tokens": min_pool_tokens,
                "reason": f"pool_tokens {pool_tokens!r} below required floor {min_pool_tokens}",
            }

    return True, {
        "ok": True,
        "http_status": status_code,
        "status": state or "ok",
        "warming": bool(warming) if warming is not None else False,
        "model": model,
        "max_batch_size": lanes,
        "pool_tokens": pool_tokens,
    }


def evaluate_models(
    status_code: int,
    body: Any,
    *,
    expected_model: str,
    expected_context: int | None = None,
    expected_efforts: tuple[str, ...] = (),
) -> tuple[bool, dict[str, Any]]:
    """Verify /v1/models advertises the expected model, context length, and reasoning efforts."""
    if status_code != 200 or not isinstance(body, dict):
        return False, {"ok": False, "http_status": status_code, "reason": f"/v1/models HTTP {status_code}"}
    entries = [e for e in (body.get("data") or []) if isinstance(e, dict) and e.get("id") == expected_model]
    if len(entries) != 1:
        return False, {
            "ok": False,
            "http_status": status_code,
            "reason": f"/v1/models does not list {expected_model!r} exactly once",
        }
    entry = entries[0]
    if expected_context is not None:
        for field in ("context_length", "max_model_len"):
            val = entry.get(field)
            if val is not None and val != expected_context:
                return False, {
                    "ok": False,
                    "http_status": status_code,
                    "reason": f"/v1/models {field}={val!r}, expected {expected_context}",
                }
    if expected_efforts:
        reported = list((entry.get("reasoning") or {}).get("supported_efforts") or [])
        if reported != list(expected_efforts):
            return False, {
                "ok": False,
                "http_status": status_code,
                "reason": f"/v1/models supported_efforts={reported!r}, expected {list(expected_efforts)!r}",
            }
    return True, {"ok": True, "model": expected_model, "context": expected_context}


def _proc_ppid(pid: int) -> int | None:
    """Return Linux parent PID from /proc/<pid>/stat when available, else None."""
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return None
    try:
        text = stat_path.read_text()
        rparen = text.rfind(")")
        if rparen < 0:
            return None
        fields = text[rparen + 2 :].split()
        # fields[0] is state, fields[1] is ppid
        return int(fields[1]) if len(fields) >= 2 and fields[1].isdigit() else None
    except (OSError, ValueError):
        return None


def check_worker_state(
    state_file: Path | str = DEFAULT_STATE_FILE,
    *,
    mode: str = "ready",
    pid_alive: Any = None,
    proc_ppid_fn: Any = _proc_ppid,
) -> tuple[bool, dict[str, Any]]:
    """Verify a non-HTTP follower rank (>0) via its live supervisor and child identities."""
    path = Path(state_file)
    if not path.is_file():
        return False, {"ok": False, "reason": f"worker state file missing: {path}"}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        return False, {"ok": False, "reason": f"unreadable worker state file: {err}"}
    if not isinstance(state, dict):
        return False, {"ok": False, "reason": "worker state file is not a JSON object"}

    rank = state.get("rank")
    if type(rank) is not int or rank <= 0:
        return False, {"ok": False, "rank": rank, "reason": f"worker probe requires follower rank > 0, got {rank!r}"}

    phase = state.get("phase")
    sup_pid = state.get("pid")
    child_pid = state.get("child_pid")
    if type(sup_pid) is not int or sup_pid <= 0:
        return False, {"ok": False, "rank": rank, "phase": phase, "reason": "worker supervisor pid is missing or invalid"}
    if type(child_pid) is not int or child_pid <= 0:
        return False, {"ok": False, "rank": rank, "phase": phase, "reason": "worker child_pid is missing or invalid"}

    if pid_alive is None:
        def _alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        alive_fn = _alive
    else:
        alive_fn = pid_alive

    if not alive_fn(sup_pid):
        return False, {"ok": False, "rank": rank, "phase": phase, "pid": sup_pid, "reason": "worker supervisor is not alive"}
    if not alive_fn(child_pid):
        return False, {"ok": False, "rank": rank, "phase": phase, "child_pid": child_pid, "reason": "worker child is not alive"}

    ppid = proc_ppid_fn(child_pid) if proc_ppid_fn is not None else None
    if ppid is not None and ppid != sup_pid:
        return False, {
            "ok": False,
            "rank": rank,
            "phase": phase,
            "pid": sup_pid,
            "child_pid": child_pid,
            "reason": f"child_pid {child_pid} parent {ppid} does not match supervisor {sup_pid} (stale/reused PID)",
        }

    if mode == "ready":
        ok = phase == "running"
    else:
        ok = phase in ("starting", "running", "flushing", "draining")
    return ok, {
        "ok": ok,
        "rank": rank,
        "world": state.get("world"),
        "pid": sup_pid,
        "phase": phase,
        "child_pid": child_pid,
        "reason": None if ok else f"worker phase is {phase!r}",
    }


def probe(
    *,
    url: str = "http://127.0.0.1:8888",
    mode: str = "ready",
    timeout: float = 10.0,
    api_key: str | None = None,
    expected_model: str | None = None,
    expected_lanes: int | None = None,
    expected_context: int | None = None,
    expected_efforts: tuple[str, ...] = (),
    min_pool_tokens: int = 0,
    check_models: bool = False,
    startup_log_path: str | None = None,
    state_file: str = DEFAULT_STATE_FILE,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Run one readiness/liveness/worker probe with full secret redaction across all emitted fields."""
    extra_secrets = (api_key,) if api_key else ()
    if mode == "worker" or mode == "worker-live":
        ok, rep = check_worker_state(state_file, mode="live" if mode == "worker-live" else "ready")
        return ok, _redact_obj(rep, env=env, extra_secrets=extra_secrets)

    base = url.rstrip("/").removesuffix("/v1")
    startup_log = Path(startup_log_path).read_text() if startup_log_path else None
    try:
        code, body = _fetch_json(base + "/health", timeout=timeout, api_key=api_key)
    except Exception as err:
        return False, _redact_obj(
            {"ok": False, "reason": f"{type(err).__name__}: {err}"}, env=env, extra_secrets=extra_secrets
        )

    ok, detail = evaluate_health(
        code,
        body,
        mode=mode,
        expected_model=expected_model,
        expected_lanes=expected_lanes,
        min_pool_tokens=min_pool_tokens,
        startup_log=startup_log,
    )
    if not ok:
        return False, _redact_obj(detail, env=env, extra_secrets=extra_secrets)

    if check_models and expected_model:
        try:
            mcode, mbody = _fetch_json(base + "/v1/models", timeout=timeout, api_key=api_key)
        except Exception as err:
            return False, _redact_obj(
                {"ok": False, "health": detail, "reason": f"{type(err).__name__}: {err}"},
                env=env,
                extra_secrets=extra_secrets,
            )
        mok, mdetail = evaluate_models(
            mcode,
            mbody,
            expected_model=expected_model,
            expected_context=expected_context,
            expected_efforts=expected_efforts,
        )
        detail["models"] = mdetail
        if not mok:
            detail["ok"] = False
            detail["reason"] = mdetail.get("reason", "")
            return False, _redact_obj(detail, env=env, extra_secrets=extra_secrets)

    return True, _redact_obj(detail, env=env, extra_secrets=extra_secrets)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ready", "live", "worker", "worker-live"), default="ready")
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--api-key-env", default="TENSORFOLD_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lanes", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--efforts", default="", help="comma-separated expected reasoning efforts")
    parser.add_argument("--min-pool-tokens", type=int, default=0)
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument("--log", default=None, help="optional startup log path for pool capacity checks")
    parser.add_argument("--state-file", default=os.environ.get("TF_RUNTIME_STATE_FILE", DEFAULT_STATE_FILE))
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    efforts = tuple(e.strip() for e in args.efforts.split(",") if e.strip())
    ok, report = probe(
        url=args.url,
        mode=args.mode,
        timeout=args.timeout,
        api_key=api_key,
        expected_model=args.model,
        expected_lanes=args.lanes,
        expected_context=args.context,
        expected_efforts=efforts,
        min_pool_tokens=args.min_pool_tokens,
        check_models=args.check_models,
        startup_log_path=args.log,
        state_file=args.state_file,
        env=os.environ,
    )
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
