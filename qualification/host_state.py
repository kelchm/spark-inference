"""Production capacity (2,582,528-token 4x1M pool floor) and host-state receipts.

Records kernel boot_id, MemAvailable (with raw /proc/meminfo & /proc/zoneinfo sidecars for offline re-parsing), and
fresh-boot vs ordinary-relaunch tagging per (host, rank) so promotion benchmarks pair both host states (accounting for
the cross-OS 260 -> 236 GB/s post-unload memory bandwidth drop on DGX Spark).
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .transport import now

POOL_FLOOR_4X1M = 2_582_528  # 5,044 pages x 512 tokens for 4x1M slots
DEFAULT_STREAMS = 4
DEFAULT_CONTEXT = 1_048_576
BOOT_PROFILES = ('fresh-boot', 'ordinary-relaunch')
BANDWIDTH_PAIRING_NOTE = (
    'Promotion benchmarks must pair fresh-boot and ordinary-relaunch runs on each host/rank to account for '
    'the cross-OS 260 -> 236 GB/s post-unload memory bandwidth drop.'
)
POOL_PAGES_RE = re.compile(r'(?:taking|all\s+\d+\s+on)\s+(\d+)\s+pages\s+of\s+(\d+)\s+rows', re.I)
POOL_LANES_OWN_RE = re.compile(r"the engine's own and (\d+) more taking \d+ pages of \d+ rows", re.I)
POOL_LANES_ALL_RE = re.compile(r'all (\d+) on \d+ pages of \d+ rows', re.I)
POOL_CONTEXT_RE = re.compile(r'at a (\d+)-token window', re.I)
STORE_OPEN_RE = re.compile(
    r'glm:\s+conversation states on disk in\s+(.+?)\s+'
    r"\((\d+)\s+states,\s+(\d+)\s+MiB\s+of\s+(\d+)\s+MiB;\s+(\d+)\s+ranks'? parts(?:;\s*[^)]*)?\)"
)
BOOT_ID_RE = re.compile(r'^[0-9a-fA-F-]{8,64}$')
GIB = 1024 ** 3
MAX_U64 = (1 << 64) - 1


def parked_pages_bytes(zoneinfo_text: str, page_size: int) -> int:
    """Free per-CPU pages held above resting high_min (512 MiB fallback on kernels without high_min)."""
    if page_size <= 0:
        return 0
    count, rest, listed = 0, 0, False
    for row in zoneinfo_text.splitlines():
        words = row.split()
        if row.strip() == 'pagesets':
            listed = True
        elif listed and not row.startswith(('    ', '  vm stats')):
            listed = False
        elif listed and words and words[0] in ('count:', 'high_min:'):
            if len(words) < 2 or not words[1].isascii() or not words[1].isdigit():
                return 0
            val = int(words[1])
            if words[0] == 'count:':
                count += val
            else:
                rest += val
            if max(count, rest) > MAX_U64:
                return 0
    held = count * page_size
    resting = rest * page_size if rest else 512 * 1024 ** 2
    if max(held, resting) > MAX_U64:
        return 0
    return max(0, held - resting)


def parse_meminfo(text: str) -> dict[str, Any]:
    """Parse /proc/meminfo KiB counters into exact KiB and rounded GiB metrics."""
    raw: dict[str, int] = {}
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 2 and words[0].endswith(':'):
            key = words[0][:-1]
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                if not words[1].isascii() or not words[1].isdigit():
                    raise ValueError(f'malformed {key} line in meminfo')
                raw[key] = int(words[1])
    if 'MemAvailable' not in raw:
        raise ValueError('MemAvailable missing from meminfo')
    mem_avail_kib = raw['MemAvailable']
    mem_total_kib = raw.get('MemTotal', 0)
    swap_total_kib = raw.get('SwapTotal', 0)
    swap_free_kib = raw.get('SwapFree', swap_total_kib)
    return {
        'mem_available_kib': mem_avail_kib,
        'mem_available_gib': round(mem_avail_kib * 1024 / GIB, 3),
        'mem_total_kib': mem_total_kib,
        'mem_total_gib': round(mem_total_kib * 1024 / GIB, 3) if mem_total_kib else None,
        'swap_used_gib': round(max(0, swap_total_kib - swap_free_kib) * 1024 / GIB, 3),
    }


def capture_host_state(
    boot_profile: str,
    *,
    boot_id: str | None = None,
    boot_id_path: Path | str = Path('/proc/sys/kernel/random/boot_id'),
    meminfo_text: str | None = None,
    meminfo_path: Path | str = Path('/proc/meminfo'),
    zoneinfo_text: str | None = None,
    zoneinfo_path: Path | str = Path('/proc/zoneinfo'),
    page_size: int | None = None,
    rank: str = 'rank0',
    host: str = '',
    run_id: str = '',
    lifecycle_seq: int | None = None,
    note: str = '',
    captured_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a host-state metadata receipt plus raw sidecar bytes for offline re-parsing."""
    if boot_profile not in BOOT_PROFILES:
        raise ValueError(f'boot_profile must be one of {BOOT_PROFILES}, got {boot_profile!r}')
    resolved_boot_id = boot_id.strip() if boot_id is not None else Path(boot_id_path).read_text().strip()
    if not resolved_boot_id or not BOOT_ID_RE.fullmatch(resolved_boot_id):
        raise ValueError(f'invalid boot_id {resolved_boot_id!r}')

    mem_text = meminfo_text if meminfo_text is not None else Path(meminfo_path).read_text()
    mem = parse_meminfo(mem_text)
    sidecars: dict[str, bytes] = {
        'boot_id': (resolved_boot_id + '\n').encode('utf-8'),
        'meminfo': mem_text.encode('utf-8'),
    }

    psize = page_size if page_size is not None else os.sysconf('SC_PAGE_SIZE')
    parked_gib = 0.0
    ztext = zoneinfo_text
    if ztext is None:
        zpath = Path(zoneinfo_path)
        if zpath.is_file():
            try:
                ztext = zpath.read_text()
            except OSError:
                ztext = None
    if ztext is not None:
        sidecars['zoneinfo'] = ztext.encode('utf-8')
        parked_gib = round(parked_pages_bytes(ztext, psize) / GIB, 3)

    effective_gib = mem['mem_available_gib'] + parked_gib
    if mem['mem_total_gib'] is not None:
        effective_gib = min(mem['mem_total_gib'], effective_gib)

    record = {
        'boot_profile': boot_profile,
        'boot_id': resolved_boot_id,
        'rank': rank,
        'host': host,
        'run_id': run_id,
        'lifecycle_seq': lifecycle_seq,
        'page_size': psize,
        'captured_at': captured_at or now(),
        'mem_available_kib': mem['mem_available_kib'],
        'mem_available_gib': mem['mem_available_gib'],
        'mem_total_gib': mem['mem_total_gib'],
        'parked_gib': parked_gib,
        'effective_available_gib': round(effective_gib, 3),
        'swap_used_gib': mem['swap_used_gib'],
        'note': note,
    }
    return record, sidecars


def evaluate_boot_pairing(
    records: list[Mapping[str, Any]],
    *,
    campaign: Any = None,
    all_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate recorded host-state receipts per (host, rank), re-parsing raw sidecars and checking chronology."""
    by_profile: dict[str, list[dict[str, Any]]] = {p: [] for p in BOOT_PROFILES}
    by_target: dict[str, dict[str, list[dict[str, Any]]]] = {}
    problems: list[str] = []
    valid_lifecycle_seqs = (
        {r.get('seq') for r in all_receipts if r.get('kind') == 'lifecycle'}
        if all_receipts is not None else None
    )

    for rec in records:
        profile = rec.get('boot_profile')
        if profile not in BOOT_PROFILES:
            problems.append(f'receipt {rec.get("seq")}: unknown boot_profile {profile!r}')
            continue

        # Prefer re-parsing raw sidecars offline rather than trusting receipt summary fields.
        sidecars = rec.get('sidecars') or {}
        if campaign is not None and 'boot_id' in sidecars:
            boot_id = campaign.sidecar(rec, 'boot_id').decode('utf-8', 'replace').strip()
        else:
            boot_id = rec.get('boot_id')
        if not isinstance(boot_id, str) or not BOOT_ID_RE.fullmatch(boot_id):
            problems.append(f'receipt {rec.get("seq")}: invalid boot_id {boot_id!r}')
            continue

        if campaign is not None and 'meminfo' in sidecars:
            try:
                mem = parse_meminfo(campaign.sidecar(rec, 'meminfo').decode('utf-8', 'replace'))
                mem_gib = mem['mem_available_gib']
                parked_gib = 0.0
                if 'zoneinfo' in sidecars:
                    psize = int(rec.get('page_size') or 4096)
                    parked_gib = round(
                        parked_pages_bytes(campaign.sidecar(rec, 'zoneinfo').decode('utf-8', 'replace'), psize) / GIB,
                        3,
                    )
                effective_gib = mem_gib + parked_gib
                if mem['mem_total_gib'] is not None:
                    effective_gib = min(mem['mem_total_gib'], effective_gib)
                effective_gib = round(effective_gib, 3)
            except (ValueError, KeyError) as err:
                problems.append(f'receipt {rec.get("seq")}: unreadable meminfo sidecar: {err}')
                continue
        else:
            mem_gib = rec.get('mem_available_gib')
            effective_gib = rec.get('effective_available_gib', mem_gib)

        if not isinstance(mem_gib, (int, float)) or isinstance(mem_gib, bool) or mem_gib <= 0:
            problems.append(f'receipt {rec.get("seq")}: invalid mem_available_gib {mem_gib!r}')
            continue

        captured_at = rec.get('captured_at')
        try:
            ts = datetime.fromisoformat(captured_at) if isinstance(captured_at, str) else None
        except ValueError:
            problems.append(f'receipt {rec.get("seq")}: invalid captured_at {captured_at!r}')
            continue

        host = str(rec.get('host') or '')
        rank = str(rec.get('rank') or 'rank0')
        lifecycle_seq = rec.get('lifecycle_seq')
        if lifecycle_seq is not None and valid_lifecycle_seqs is not None and lifecycle_seq not in valid_lifecycle_seqs:
            problems.append(f'receipt {rec.get("seq")}: references nonexistent lifecycle receipt seq={lifecycle_seq!r}')
        target_key = f'{host or "default"}:{rank}'
        entry = {
            'seq': rec.get('seq'),
            'rank': rank,
            'host': host,
            'run_id': rec.get('run_id') or '',
            'lifecycle_seq': lifecycle_seq,
            'boot_id': boot_id,
            'mem_available_gib': mem_gib,
            'effective_available_gib': effective_gib,
            'captured_at': captured_at,
            '_ts': ts,
        }
        by_profile[profile].append({k: v for k, v in entry.items() if k != '_ts'})
        by_target.setdefault(target_key, {p: [] for p in BOOT_PROFILES})[profile].append(entry)

    # Every expected campaign rank and every recorded (host, rank) target must have BOTH profiles with valid chronology.
    expected_ranks: list[str] = []
    if campaign is not None and isinstance(getattr(campaign, 'manifest', None), Mapping):
        n_ranks = campaign.manifest.get('ranks')
        if type(n_ranks) is int and n_ranks >= 1:
            expected_ranks = [f'rank{i}' for i in range(n_ranks)]
    seen_ranks = {e['rank'] for entries in by_profile.values() for e in entries}
    missing_ranks = [r for r in expected_ranks if r not in seen_ranks]

    target_reports: dict[str, dict[str, Any]] = {}
    all_targets_paired = bool(by_target) and not missing_ranks
    missing_profiles_overall: set[str] = set()
    for tkey, buckets in sorted(by_target.items()):
        t_missing = [p for p in BOOT_PROFILES if not buckets[p]]
        missing_profiles_overall.update(t_missing)
        t_problems: list[str] = []
        # If fresh-boot and ordinary-relaunch share a boot_id on the same host/rank, ordinary-relaunch must follow.
        for fb in buckets['fresh-boot']:
            for rl in buckets['ordinary-relaunch']:
                if fb['boot_id'] == rl['boot_id'] and fb['_ts'] is not None and rl['_ts'] is not None:
                    if rl['_ts'] < fb['_ts']:
                        t_problems.append(
                            f'{tkey}: ordinary-relaunch ({rl["captured_at"]}) precedes fresh-boot '
                            f'({fb["captured_at"]}) on the same boot_id {fb["boot_id"]}'
                        )
        problems.extend(t_problems)
        t_paired = not t_missing and not t_problems
        if not t_paired:
            all_targets_paired = False
        target_reports[tkey] = {
            'paired': t_paired,
            'missing_profiles': t_missing,
            'fresh_boot_ids': sorted({x['boot_id'] for x in buckets['fresh-boot']}),
            'relaunch_boot_ids': sorted({x['boot_id'] for x in buckets['ordinary-relaunch']}),
            'problems': t_problems,
        }

    present = [p for p in BOOT_PROFILES if by_profile[p]]
    missing = sorted(missing_profiles_overall) if by_target else list(BOOT_PROFILES)
    paired = all_targets_paired and not problems

    notes: list[str] = []
    if records and not paired:
        parts = []
        if missing:
            parts.append(', '.join(missing))
        if missing_ranks:
            parts.append('missing ranks: ' + ', '.join(missing_ranks))
        missing_desc = '; '.join(parts) if parts else 'per-host/rank pairing'
        notes.append(f'Host-state pairing incomplete ({missing_desc}). {BANDWIDTH_PAIRING_NOTE}')

    if problems:
        status = 'fail'
    elif not records:
        status = 'not_run'
    elif paired:
        status = 'pass'
    else:
        status = 'unknown'

    return {
        'status': status,
        'paired': paired,
        'present_profiles': present,
        'missing_profiles': missing,
        'missing_ranks': missing_ranks,
        'targets': target_reports,
        'records': by_profile,
        'problems': problems,
        'notes': notes,
    }


def _lanes_from_log(startup_log: str | None) -> int | None:
    if not startup_log:
        return None
    own = POOL_LANES_OWN_RE.findall(startup_log)
    if own:
        return 1 + int(own[-1])
    all_m = POOL_LANES_ALL_RE.findall(startup_log)
    if all_m:
        return int(all_m[-1])
    return None


def _context_from_log(startup_log: str | None) -> int | None:
    if not startup_log:
        return None
    m = POOL_CONTEXT_RE.findall(startup_log)
    return int(m[-1]) if m else None


def evaluate_capacity(
    health_body: Any,
    startup_log: str | None = None,
    models_body: Any = None,
    *,
    min_pool_tokens: int = POOL_FLOOR_4X1M,
    expected_streams: int = DEFAULT_STREAMS,
    expected_context: int = DEFAULT_CONTEXT,
    expected_model: str = 'GLM-5.3-Flash',
    learn_gib: int | None = None,
    learn_dir: str | None = None,
    check_learn_store: bool = False,
    ranks: int = 2,
) -> dict[str, Any]:
    """Verify production 4x1M capacity floor (2,582,528 tokens) plus observed model, context, lanes, and store."""
    reasons: list[str] = []
    if not isinstance(health_body, dict):
        return {
            'status': 'fail',
            'pool_tokens': None,
            'min_pool_tokens': min_pool_tokens,
            'pool_floor_met': False,
            'streams': None,
            'observed_context': None,
            'context': expected_context,
            'model': None,
            'source': None,
            'reasons': ['health body is not a JSON object'],
        }

    # Health & streams extraction
    observed_context: int | None = None
    if 'ok' in health_body:
        if health_body['ok'] is not True:
            reasons.append('engine health reported ok != true')
        streams = (health_body.get('streams') or {}).get('max') if isinstance(health_body.get('streams'), dict) else None
        model = health_body.get('model')
        if type(health_body.get('context_length')) is int:
            observed_context = int(health_body['context_length'])
    else:
        if health_body.get('status') != 'ok' or health_body.get('warming') is True:
            reasons.append(
                f'engine unhealthy or warming: status={health_body.get("status")!r}, warming={health_body.get("warming")!r}'
            )
        streams = health_body.get('max_batch_size')
        model = health_body.get('model')
        for k in ('context_length', 'max_model_len'):
            if type(health_body.get(k)) is int:
                observed_context = int(health_body[k])
                break

    # Supplement from /v1/models discovery (required for observed context and for keyed minimal /health)
    if isinstance(models_body, dict):
        entries = [e for e in (models_body.get('data') or []) if isinstance(e, dict) and e.get('id') == expected_model]
        if len(entries) != 1:
            reasons.append(f'/v1/models does not list expected model {expected_model!r} exactly once')
        else:
            entry = entries[0]
            if model is None:
                model = entry.get('id')
            ctx_vals = [entry.get(k) for k in ('context_length', 'max_model_len') if type(entry.get(k)) is int]
            if ctx_vals:
                if len(set(ctx_vals)) != 1:
                    reasons.append(f'/v1/models disagrees on context_length vs max_model_len: {ctx_vals}')
                if observed_context is not None and observed_context != ctx_vals[0]:
                    reasons.append(f'/health context ({observed_context}) disagrees with /v1/models ({ctx_vals[0]})')
                observed_context = ctx_vals[0]

    # Keyed native /health returns {"status": "ok"} without max_batch_size; read lanes/context from startup log if present.
    if streams is None:
        streams = _lanes_from_log(startup_log)
    if observed_context is None:
        observed_context = _context_from_log(startup_log)

    if model is None:
        reasons.append('no observed model identity from /health or /v1/models')
    elif model != expected_model:
        reasons.append(f'observed model {model!r} does not match expected {expected_model!r}')

    if observed_context is None:
        reasons.append('no observed context length from /v1/models, /health, or startup log')
    elif observed_context != expected_context:
        reasons.append(f'observed context {observed_context} does not match expected {expected_context}')

    if type(streams) is not int or streams < expected_streams:
        reasons.append(f'observed streams {streams!r} below required {expected_streams}')

    # Pool token capacity extraction (supports direct pool_tokens, kv_shard, and v1.3.6 paged pool startup log)
    pool_tokens: int | None = None
    source: str | None = None
    page_tokens = 512
    pages_reported: int | None = None

    if type(health_body.get('pool_tokens')) is int:
        pool_tokens = int(health_body['pool_tokens'])
        source = 'health.pool_tokens'
    elif isinstance(health_body.get('kv_shard'), dict):
        shard = health_body['kv_shard']
        parts = shard.get('parts')
        ptok = shard.get('page_tokens')
        ppages = shard.get('part_pages')
        if all(type(v) is int and v > 0 for v in (parts, ptok, ppages)):
            page_tokens = ptok
            pages_reported = parts * ppages
            pool_tokens = pages_reported * page_tokens
            source = 'health.kv_shard'
            required_part_pages = (expected_context + 16 + page_tokens - 1) // page_tokens
            if ppages < required_part_pages:
                reasons.append(f'kv_shard part_pages {ppages} below single-stream 1M requirement {required_part_pages}')
    if pool_tokens is None and startup_log:
        matches = POOL_PAGES_RE.findall(startup_log)
        if matches:
            pages_reported = int(matches[-1][0])
            page_tokens = int(matches[-1][1])
            pool_tokens = pages_reported * page_tokens
            source = 'log.kv_pool'
            required_pages = (expected_context + 16 + page_tokens - 1) // page_tokens
            if pages_reported < required_pages:
                reasons.append(f'pool pages {pages_reported} below single-stream context requirement {required_pages}')

    pool_floor_met = pool_tokens is not None and pool_tokens >= min_pool_tokens
    if pool_tokens is None:
        reasons.append('no pool_tokens or KV page pool receipt found in /health or startup log')
    elif not pool_floor_met:
        reasons.append(f'pool capacity {pool_tokens} tokens below required 4x1M floor {min_pool_tokens}')

    learned_store: dict[str, Any] | None = None
    if check_learn_store or learn_dir is not None:
        if not learn_dir:
            reasons.append('learn-store check requested without a profile learn_dir')
        elif not startup_log:
            reasons.append('learn-store check requested but no startup log was provided')
        else:
            matches = STORE_OPEN_RE.findall(startup_log)
            if not matches:
                reasons.append('missing learned-store opening line in startup log')
            else:
                opened_dir, states, used_mib, cap_mib, opened_ranks = matches[-1]
                learned_store = {
                    'dir': opened_dir,
                    'states': int(states),
                    'used_mib': int(used_mib),
                    'cap_mib': int(cap_mib),
                    'ranks': int(opened_ranks),
                }
                pattern = re.escape(learn_dir.rstrip('/')) + r'/[0-9a-f]{16}'
                if re.fullmatch(pattern, opened_dir) is None:
                    reasons.append(f'learned store {opened_dir!r} is not a 16-hex child of {learn_dir!r}')
                if int(opened_ranks) != ranks:
                    reasons.append(f'learned store opened with {opened_ranks} ranks, expected {ranks}')
                if learn_gib is not None and int(cap_mib) != learn_gib * 1024:
                    reasons.append(f'learned store cap {cap_mib} MiB != expected {learn_gib * 1024} MiB')

    return {
        'status': 'fail' if reasons else 'pass',
        'pool_tokens': pool_tokens,
        'min_pool_tokens': min_pool_tokens,
        'pool_floor_met': pool_floor_met,
        'pages': pages_reported,
        'page_tokens': page_tokens,
        'streams': streams,
        'expected_streams': expected_streams,
        'observed_context': observed_context,
        'context': expected_context,
        'model': model,
        'source': source,
        'learned_store': learned_store,
        'reasons': reasons,
    }
