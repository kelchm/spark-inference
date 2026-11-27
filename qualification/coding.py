"""Loopback coding replay (40K / 262K + cold-prefill concurrent decoder) and 1/2/3/4 concurrency reporting.

Matches the real Pi busy-time profile buckets (C1 / C2-C3 / C4) and explicitly quantifies decode starvation, queueing
TTFT, and SSE event-arrival inter-event latency tail metrics offline from recorded receipts and wire bytes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
from typing import Any, Callable, Mapping

from . import workload
from .transport import canonical, exchange, itl_summary, parse_sse, percentile, sha256, strict_json

CODING_SHAPES = {
    '40k': 40_960,
    '262k': 262_144,
    'cold-prefill-concurrent': 40_960,
}
CONCURRENCY_LEVELS = (1, 2, 3, 4)
DEFAULT_STARVATION_MS = 2000.0
CODING_SYSTEM = (
    'You are a software engineer inspecting a codebase snapshot. Answer symbol and ticket lookups accurately from the '
    'repository context in this conversation.'
)


def pi_bucket(level: int) -> str:
    """Map concurrent stream count to the Pi busy-time profile bucket (C1 / C2-C3 / C4)."""
    if level <= 1:
        return 'C1'
    if level <= 3:
        return 'C2-C3'
    return 'C4'


def coding_fact(seed: str, stream_tag: str, index: int) -> tuple[str, str]:
    """Deterministic symbol name and expected ticket token planted inside synthetic source files."""
    det = workload.Det(seed, f'coding/{stream_tag}/{index}')
    letters = ''.join(chr(65 + (index // 26 ** k) % 26) for k in (2, 1, 0))
    symbol = f'MOD_{stream_tag.upper().replace("-", "_")}_{index:04d}'
    code = f'{workload.CODEWORDS[det.below(len(workload.CODEWORDS))]}-{det.below(9000) + 1000}-{letters}'
    return symbol, code


def coding_file_chunk(seed: str, stream_tag: str, index: int) -> str:
    """Deterministic multi-function synthetic repository file carrying one planted symbol token."""
    det = workload.Det(seed, f'coding-file/{stream_tag}/{index}')
    symbol, code = coding_fact(seed, stream_tag, index)
    lines = [
        f'# file: src/pkg_{index % 17:02d}/module_{index:04d}.py',
        f'{symbol} = "{code}"',
        '',
    ]
    for fn_idx in range(12):
        fn_words = [workload.WORDS[det.below(len(workload.WORDS))] for _ in range(8)]
        fn_name = f'handle_{fn_words[0]}_{index}_{fn_idx}'
        lines += [
            f'def {fn_name}(state: dict, items: list[str]) -> dict:',
            f'    """Process {" ".join(fn_words)} for {symbol}."""',
            f'    tag = "{fn_words[1]}-{fn_words[2]}-{det.below(999)}"',
            '    out = [item.strip().lower() for item in items if item]',
            f'    return {{"tag": tag, "count": len(out), "marker": {symbol}, "ok": bool(state)}}',
            '',
        ]
    return '\n'.join(lines)


def coding_turn_body(
    plan: Mapping[str, Any],
    stream_tag: str,
    turn_index: int,
    *,
    chunks: int,
    chunk_start: int = 0,
    history: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Build deterministic coding replay turn (cold seed on turn 0, warm incremental diffs on turns >=1)."""
    seed = plan['seed']
    if history:
        last_body, last_msg = history[-1]
        messages = list(last_body['messages']) + [workload.replay(last_msg)]
    else:
        messages = [{'role': 'system', 'content': CODING_SYSTEM}]

    files = '\n\n'.join(coding_file_chunk(seed, stream_tag, i) for i in range(chunk_start, chunk_start + chunks))
    total = chunk_start + chunks
    query_indices = list(dict.fromkeys([0, chunk_start, max(0, total - 1)]))
    questions, expected = [], {}
    for qn, idx in enumerate(query_indices, 1):
        sym, code = coding_fact(seed, stream_tag, idx)
        questions.append(f'Q{qn}: What string value is assigned to {sym}?')
        expected[str(qn)] = code

    header = (
        f'Repository snapshot ({stream_tag}, turn {turn_index}):\n'
        if turn_index == 0
        else f'Incremental edit diff ({stream_tag}, turn {turn_index}):\n'
    )
    content = header + (files + '\n\n' if files else '') + 'Questions:\n' + '\n'.join(questions) + '\n' + workload.FORMAT
    messages.append({'role': 'user', 'content': content})

    body = workload.chat_body(plan, 'coding', messages)
    body['temperature'] = 0
    if max_tokens is not None:
        body['max_tokens'] = max_tokens
    meta = {
        'stream_tag': stream_tag,
        'turn_index': turn_index,
        'chunk_start': chunk_start,
        'chunks': chunks,
        'total_chunks': total,
    }
    return body, expected, meta


def decoder_turn_body(
    plan: Mapping[str, Any],
    stream_tag: str,
    *,
    max_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Lightweight deterministic decoding stream run alongside cold prefill to detect decode starvation."""
    sym1, code1 = coding_fact(plan['seed'], f'dec-{stream_tag}', 0)
    sym2, code2 = coding_fact(plan['seed'], f'dec-{stream_tag}', 1)
    content = (
        f'Reference constants:\n{sym1} = "{code1}"\n{sym2} = "{code2}"\n\n'
        'Questions:\n'
        f'Q1: What string value is assigned to {sym1}?\n'
        f'Q2: What string value is assigned to {sym2}?\n'
        + workload.FORMAT
    )
    body = workload.chat_body(
        plan,
        'coding-decoder',
        [{'role': 'system', 'content': CODING_SYSTEM}, {'role': 'user', 'content': content}],
    )
    body['temperature'] = 0
    body['max_tokens'] = max_tokens
    return body, {'1': code1, '2': code2}, {'stream_tag': stream_tag, 'role': 'decoder'}


def _stream_metrics(
    receipt_entry: Mapping[str, Any],
    reply: Mapping[str, Any] | None,
    *,
    starvation_ms: float,
) -> dict[str, Any]:
    event_times = receipt_entry.get('event_times_s') or receipt_entry.get('token_times_s') or []
    profile = itl_summary(
        event_times,
        starvation_ms=starvation_ms,
        max_gap_override_ms=receipt_entry.get('max_gap_ms_observed'),
    )
    completion_tokens = (reply or {}).get('usage', {}).get('completion_tokens') or 0
    decode_s = profile['decode_s']
    elapsed_s = receipt_entry.get('elapsed_s') or 0.0
    ttft_s = profile['ttft_s'] if profile['ttft_s'] is not None else (receipt_entry.get('timings') or {}).get('first_token_s')
    if decode_s > 0 and completion_tokens > 0:
        decode_tok_s = round(completion_tokens / decode_s, 3)
    elif elapsed_s > 0 and completion_tokens > 0:
        decode_tok_s = round(completion_tokens / elapsed_s, 3)
    else:
        decode_tok_s = 0.0
    ttft_starved = ttft_s is not None and (ttft_s * 1000.0) > starvation_ms
    return {
        'timing_basis': 'sse_event_arrival',
        'elapsed_s': elapsed_s,
        'ttft_s': ttft_s,
        'ttft_starved': ttft_starved,
        'decode_s': decode_s,
        'completion_tokens': completion_tokens,
        'decode_tok_s': decode_tok_s,
        'itl_ms': profile['itl_ms'],
        'starvation_events': profile['starvation_events'] + (1 if ttft_starved else 0),
        'max_stall_ms': profile['max_stall_ms'],
    }


def _wall_overlap_s(a: Mapping[str, Any], b: Mapping[str, Any]) -> float | None:
    keys = ('mono_start_s', 'mono_end_s')
    if not all(isinstance(x.get(k), (int, float)) for x in (a, b) for k in keys):
        return None
    overlap = min(float(a['mono_end_s']), float(b['mono_end_s'])) - max(float(a['mono_start_s']), float(b['mono_start_s']))
    return round(max(0.0, overlap), 6)


def run_coding_replay(
    campaign: Any,
    *,
    endpoint: str,
    shape: str,
    turns: int = 3,
    target_tokens: int | None = None,
    deadline: float = 3600.0,
    tokenize_deadline: float = 900.0,
    starvation_ms: float = DEFAULT_STARVATION_MS,
    api_key: str | None = None,
    tokenize_fn: Callable[[dict[str, Any], str, str, str], int],
    chat_limit: int = 16 << 20,
) -> dict[str, Any]:
    """Run deterministic 40k / 262k warm coding replay OR cold-prefill + concurrent decoder check."""
    if shape not in CODING_SHAPES:
        raise ValueError(f'shape must be one of {sorted(CODING_SHAPES)}, got {shape!r}')
    plan = campaign.plan
    goal = target_tokens if target_tokens is not None else CODING_SHAPES[shape]
    limit = plan['ceiling'] - plan['margin']
    stream_tag = f'replay-{shape}'

    if shape == 'cold-prefill-concurrent':
        # 1. Baseline solo decoder
        base_body, _, base_meta = decoder_turn_body(plan, 'baseline')
        base_req = canonical(base_body)
        base_res = exchange(endpoint, 'POST', '/v1/chat/completions', base_req, deadline, stream=True,
                            max_bytes=chat_limit, api_key=api_key)
        base_raw = base_res.pop('raw')

        # 2. Fit cold prefill prompt before launching concurrent pair
        def count_cold(n: int) -> int:
            b, _, _ = coding_turn_body(plan, 'cold-prefill', 0, chunks=n, chunk_start=0)
            return tokenize_fn(b, 'fit', 'coding-cold', 'cold')

        cold_chunks, cold_fitted, _ = workload.fit(count_cold, goal, limit, reach=True)
        cold_body, _, cold_meta = coding_turn_body(plan, 'cold-prefill', 0, chunks=cold_chunks, chunk_start=0)
        cold_measured = tokenize_fn(cold_body, 'measure', 'coding-cold', 'cold')
        cold_req = canonical(cold_body)

        conc_body, _, conc_meta = decoder_turn_body(plan, 'concurrent')
        conc_req = canonical(conc_body)

        # 3. Start decoder first, wait until its first SSE event arrives, THEN launch cold prefill so they genuinely overlap.
        decoder_active = threading.Event()
        decoder_done = threading.Event()

        def run_decoder() -> dict[str, Any]:
            try:
                return exchange(
                    endpoint, 'POST', '/v1/chat/completions', conc_req, deadline, stream=True,
                    max_bytes=chat_limit, api_key=api_key, on_first_token=decoder_active.set,
                )
            finally:
                decoder_done.set()

        def run_cold() -> tuple[bool, dict[str, Any]]:
            # Wait until decoder emits its first SSE event (or finishes / small wait timeout).
            waited = decoder_active.wait(timeout=min(10.0, deadline))
            res = exchange(
                endpoint, 'POST', '/v1/chat/completions', cold_req, deadline, stream=True,
                max_bytes=chat_limit, api_key=api_key,
            )
            return waited, res

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dec = pool.submit(run_decoder)
            f_cold = pool.submit(run_cold)
            conc_res = f_dec.result()
            launched_after_first, cold_res = f_cold.result()

        cold_raw = cold_res.pop('raw')
        conc_raw = conc_res.pop('raw')
        overlap_s = _wall_overlap_s(cold_res, conc_res)
        record = campaign.write(
            'coding-replay',
            shape,
            {
                'endpoint': endpoint,
                'shape': shape,
                'target_tokens': goal,
                'starvation_threshold_ms': starvation_ms,
                'launched_after_decoder_first_event': bool(launched_after_first),
                'wall_overlap_s': overlap_s,
                'baseline_decoder': {'meta': base_meta, 'request_sha256': sha256(base_req), **base_res},
                'cold_prefill': {
                    'meta': cold_meta,
                    'measured_prompt_tokens': cold_measured,
                    'fitted_tokens': cold_fitted,
                    'request_sha256': sha256(cold_req),
                    **cold_res,
                },
                'concurrent_decoder': {'meta': conc_meta, 'request_sha256': sha256(conc_req), **conc_res},
            },
            {
                'baseline_request': base_req,
                'baseline_response': base_raw,
                'cold_request': cold_req,
                'cold_response': cold_raw,
                'concurrent_request': conc_req,
                'concurrent_response': conc_raw,
            },
        )
        return record

    # Warm multi-turn coding replay (40k or 262k history shape)
    planned_turns = max(1, turns)

    def count_seed(n: int) -> int:
        b, _, _ = coding_turn_body(plan, stream_tag, 0, chunks=n, chunk_start=0)
        return tokenize_fn(b, 'fit', stream_tag, 'turn-0')

    seed_chunks, _, _ = workload.fit(count_seed, goal, limit, reach=True)
    history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    chunk_cursor = 0
    turns_meta: list[dict[str, Any]] = []
    sidecars: dict[str, bytes] = {}

    for t in range(planned_turns):
        n_chunks = seed_chunks if t == 0 else 1
        body, _, meta = coding_turn_body(plan, stream_tag, t, chunks=n_chunks, chunk_start=chunk_cursor, history=history)
        measured = tokenize_fn(body, 'measure', stream_tag, f'turn-{t}')
        req_bytes = canonical(body)
        res = exchange(endpoint, 'POST', '/v1/chat/completions', req_bytes, deadline, stream=True,
                       max_bytes=chat_limit, api_key=api_key)
        raw = res.pop('raw')
        sidecars[f't{t}_request'] = req_bytes
        sidecars[f't{t}_response'] = raw
        turns_meta.append({
            'turn_index': t,
            'meta': meta,
            'measured_prompt_tokens': measured,
            'request_sha256': sha256(req_bytes),
            **res,
        })
        chunk_cursor += n_chunks
        if not res['complete']:
            break
        try:
            parsed = parse_sse(raw)
        except ValueError:
            break
        history.append((body, parsed['message']))

    return campaign.write(
        'coding-replay',
        shape,
        {
            'endpoint': endpoint,
            'shape': shape,
            'planned_turns': planned_turns,
            'target_tokens': goal,
            'starvation_threshold_ms': starvation_ms,
            'turns': turns_meta,
        },
        sidecars,
    )


def run_concurrency(
    campaign: Any,
    *,
    endpoint: str,
    levels: tuple[int, ...] = CONCURRENCY_LEVELS,
    chunks_per_stream: int = 2,
    deadline: float = 3600.0,
    starvation_ms: float = DEFAULT_STARVATION_MS,
    api_key: str | None = None,
    chat_limit: int = 16 << 20,
) -> list[dict[str, Any]]:
    """Run 1/2/3/4 multi-stream concurrency probes and write tamper-proof receipts with raw wire bytes."""
    plan = campaign.plan
    written: list[dict[str, Any]] = []
    for level in levels:
        if level < 1 or level > 16:
            raise ValueError(f'concurrency level must be in 1..16, got {level}')
        payloads: list[tuple[int, bytes, dict[str, Any]]] = []
        for idx in range(level):
            tag = f'conc-c{level}-s{idx}'
            body, _, meta = coding_turn_body(plan, tag, 0, chunks=chunks_per_stream, chunk_start=0)
            payloads.append((idx, canonical(body), meta))

        barrier = threading.Barrier(level)

        def fire(item: tuple[int, bytes, dict[str, Any]]) -> tuple[int, bytes, dict[str, Any], dict[str, Any]]:
            idx, req_bytes, meta = item
            try:
                barrier.wait(timeout=min(10.0, deadline))
            except threading.BrokenBarrierError:
                pass
            res = exchange(endpoint, 'POST', '/v1/chat/completions', req_bytes, deadline, stream=True,
                           max_bytes=chat_limit, api_key=api_key)
            return idx, req_bytes, meta, res

        with ThreadPoolExecutor(max_workers=level) as pool:
            completed = list(pool.map(fire, payloads))

        streams_data: list[dict[str, Any]] = []
        sidecars: dict[str, bytes] = {}
        for idx, req_bytes, meta, res in sorted(completed, key=lambda x: x[0]):
            raw = res.pop('raw')
            sidecars[f's{idx}_request'] = req_bytes
            sidecars[f's{idx}_response'] = raw
            streams_data.append({
                'stream_index': idx,
                'meta': meta,
                'request_sha256': sha256(req_bytes),
                **res,
            })

        rec = campaign.write(
            'concurrency',
            f'c{level}',
            {
                'endpoint': endpoint,
                'concurrency': level,
                'pi_bucket': pi_bucket(level),
                'chunks_per_stream': chunks_per_stream,
                'starvation_threshold_ms': starvation_ms,
                'streams': streams_data,
            },
            sidecars,
        )
        written.append(rec)
    return written


# ---------------------------------------------------------------------------------------------------------------------
# Offline verdict evaluator for coding-replay and concurrency receipts (never hides prior failures)


def _merge_attempt(
    store: dict[str, dict[str, Any]],
    attempts: dict[str, list[dict[str, Any]]],
    key: str,
    report: dict[str, Any],
) -> None:
    attempts.setdefault(key, []).append(report)
    failed = [a['seq'] for a in attempts[key] if a['status'] == 'fail']
    merged = dict(report, failed_attempts=failed)
    if failed:
        merged['status'] = 'fail'
    store[key] = merged


def evaluate_performance(
    campaign: Any,
    receipts: list[dict[str, Any]],
    measures: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Re-parse coding-replay and 1/2/3/4 concurrency wire bytes and compute starvation / tail-latency reports."""
    coding_recs = [r for r in receipts if r.get('kind') == 'coding-replay']
    conc_recs = [r for r in receipts if r.get('kind') == 'concurrency']
    if not coding_recs and not conc_recs:
        return {'status': 'not_run', 'coding_replay': {}, 'concurrency': {'status': 'not_run', 'levels': {}, 'pi_buckets': {}}}

    plan = campaign.plan
    coding_out: dict[str, Any] = {}
    coding_attempts: dict[str, list[dict[str, Any]]] = {}
    for rec in coding_recs:
        shape = rec.get('shape', '')
        starvation_ms = float(rec.get('starvation_threshold_ms', DEFAULT_STARVATION_MS))
        if shape == 'cold-prefill-concurrent':
            rep = _eval_cold_concurrent(campaign, plan, rec, measures, starvation_ms=starvation_ms)
        else:
            rep = _eval_warm_coding(campaign, plan, rec, measures, starvation_ms=starvation_ms)
        _merge_attempt(coding_out, coding_attempts, shape, rep)

    conc_out = _eval_concurrency(campaign, plan, conc_recs)

    rank = {'fail': 3, 'unknown': 2, 'pass': 1}
    statuses = [v['status'] for v in coding_out.values()]
    if conc_out['status'] != 'not_run':
        statuses.append(conc_out['status'])
    return {
        'status': max(statuses, key=rank.__getitem__) if statuses else 'not_run',
        'coding_replay': coding_out,
        'concurrency': conc_out,
    }


def _eval_warm_coding(
    campaign: Any,
    plan: Mapping[str, Any],
    rec: dict[str, Any],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    starvation_ms: float,
) -> dict[str, Any]:
    shape = rec['shape']
    target_tokens = int(rec.get('target_tokens', CODING_SHAPES.get(shape, 1)))
    planned_turns = int(rec.get('planned_turns', len(rec.get('turns') or []) or 1))
    history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    turn_reports: list[dict[str, Any]] = []
    reasons_top: list[str] = []
    all_ok = True

    for tinfo in rec.get('turns') or []:
        t = tinfo['turn_index']
        reasons: list[str] = []
        body = campaign.parsed(rec, f't{t}_request', strict_json)
        if tinfo.get('error') or tinfo.get('deadline_exceeded'):
            reasons.append(f'transport: {tinfo.get("error")}')
        try:
            reply = campaign.parsed(rec, f't{t}_response', parse_sse)
        except ValueError as err:
            reply = None
            reasons.append(f'incomplete or malformed reply: {err}')

        meta = tinfo.get('meta') or {}
        exp_body, expected, _ = coding_turn_body(
            plan,
            meta.get('stream_tag', f'replay-{shape}'),
            t,
            chunks=meta.get('chunks', 1),
            chunk_start=meta.get('chunk_start', 0),
            history=history,
        )
        if canonical(exp_body) != canonical(body):
            reasons.append('request does not match deterministic coding replay plan')
        measured = measures.get(('measure', sha256(canonical(workload.tokenize_body(body)))))
        measured_count = measured['count'] if measured else None
        if measured_count is None:
            reasons.append('missing /v1/tokenize measurement')
        else:
            if t == 0 and measured_count < target_tokens:
                reasons.append(f'turn 0 measured {measured_count} tokens, below {target_tokens} target')
            if measured_count > plan['ceiling']:
                reasons.append(f'turn {t} measured {measured_count} tokens, exceeds ceiling {plan["ceiling"]}')

        correct = False
        if reply is not None:
            correct = workload.correct(expected, reply['message']['content'])
            if not correct:
                reasons.append('wrong or missing coding replay answer')
            history.append((body, reply['message']))

        metrics = _stream_metrics(tinfo, reply, starvation_ms=starvation_ms)
        if reasons:
            all_ok = False
        turn_reports.append({
            'turn_index': t,
            'measured_prompt_tokens': measured_count,
            'correct': correct,
            'status': 'fail' if reasons else 'pass',
            'reasons': reasons,
            **metrics,
        })

    if len(turn_reports) != planned_turns:
        all_ok = False
        reasons_top.append(f'completed {len(turn_reports)} of {planned_turns} planned turns')
    return {
        'seq': rec['seq'],
        'shape': shape,
        'planned_turns': planned_turns,
        'target_tokens': target_tokens,
        'status': 'pass' if all_ok else 'fail',
        'reasons': reasons_top,
        'turns': turn_reports,
    }


def _eval_cold_concurrent(
    campaign: Any,
    plan: Mapping[str, Any],
    rec: dict[str, Any],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    starvation_ms: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    parts = {}
    for key, builder in (
        ('baseline', lambda m: decoder_turn_body(plan, 'baseline')),
        ('cold', lambda m: coding_turn_body(plan, 'cold-prefill', 0, chunks=m.get('chunks', 1), chunk_start=0)),
        ('concurrent', lambda m: decoder_turn_body(plan, 'concurrent')),
    ):
        entry_key = 'baseline_decoder' if key == 'baseline' else 'cold_prefill' if key == 'cold' else 'concurrent_decoder'
        entry = rec.get(entry_key) or {}
        part_reasons: list[str] = []
        body = campaign.parsed(rec, f'{key}_request', strict_json)
        if entry.get('error') or entry.get('deadline_exceeded'):
            part_reasons.append(f'transport: {entry.get("error")}')
        try:
            reply = campaign.parsed(rec, f'{key}_response', parse_sse)
        except ValueError as err:
            reply = None
            part_reasons.append(f'incomplete or malformed reply: {err}')
        exp_body, expected, _ = builder(entry.get('meta') or {})
        if canonical(exp_body) != canonical(body):
            part_reasons.append(f'{key} request does not match deterministic plan')
        if key == 'cold':
            measured = measures.get(('measure', sha256(canonical(workload.tokenize_body(body)))))
            if measured is None or measured['count'] < int(rec.get('target_tokens', 1)):
                part_reasons.append(f'cold prefill tokens {measured and measured["count"]} below target')
            elif measured['count'] > plan['ceiling']:
                part_reasons.append(f'cold prefill tokens {measured["count"]} exceed ceiling {plan["ceiling"]}')
        correct = False
        if reply is not None:
            correct = workload.correct(expected, reply['message']['content'])
            if not correct:
                part_reasons.append(f'wrong or missing {key} answer')
        metrics = _stream_metrics(entry, reply, starvation_ms=starvation_ms)
        if part_reasons:
            reasons.extend(part_reasons)
        parts[key] = {'correct': correct, 'status': 'fail' if part_reasons else 'pass', 'reasons': part_reasons, **metrics}

    # Verify wall-clock AND active-decode overlap between cold prefill and concurrent decoder.
    cold_entry = rec.get('cold_prefill') or {}
    conc_entry = rec.get('concurrent_decoder') or {}
    overlap_s = _wall_overlap_s(cold_entry, conc_entry)
    if overlap_s is None:
        overlap_s = rec.get('wall_overlap_s')
    launched_after = bool(rec.get('launched_after_decoder_first_event'))
    active_decode_overlap_s: float | None = None
    cold_after_first: bool | None = None
    if all(isinstance(v, (int, float)) for v in (
        cold_entry.get('mono_start_s'),
        cold_entry.get('mono_end_s'),
        conc_entry.get('first_event_mono_s'),
        conc_entry.get('last_event_mono_s'),
    )):
        cold_start = float(cold_entry['mono_start_s'])
        cold_end = float(cold_entry['mono_end_s'])
        dec_first = float(conc_entry['first_event_mono_s'])
        dec_last = float(conc_entry['last_event_mono_s'])
        cold_after_first = cold_start >= dec_first
        active_decode_overlap_s = round(
            max(0.0, min(cold_end, dec_last) - max(cold_start, dec_first)),
            6,
        )
    if not launched_after or cold_after_first is False:
        reasons.append(
            f'cold prefill did not wait for decoder first event '
            f'(launched_after_decoder_first_event={launched_after}, cold_after_first={cold_after_first})'
        )
    overlapped = (
        overlap_s is not None
        and overlap_s > 0.0
        and active_decode_overlap_s is not None
        and active_decode_overlap_s > 0.0
    )
    if not overlapped:
        reasons.append(
            f'cold prefill did not overlap active decoding '
            f'(wall_overlap_s={overlap_s!r}, active_decode_overlap_s={active_decode_overlap_s!r})'
        )

    conc_metrics = parts['concurrent']
    starved = (
        conc_metrics['starvation_events'] > 0
        or conc_metrics['max_stall_ms'] > starvation_ms
        or conc_metrics['ttft_starved']
    )
    if starved:
        reasons.append(
            f'concurrent decoder starved during cold prefill: max_stall_ms={conc_metrics["max_stall_ms"]}, '
            f'ttft_s={conc_metrics["ttft_s"]} (threshold {starvation_ms} ms)'
        )
    return {
        'seq': rec['seq'],
        'shape': 'cold-prefill-concurrent',
        'starvation_threshold_ms': starvation_ms,
        'launched_after_decoder_first_event': bool(rec.get('launched_after_decoder_first_event')),
        'wall_overlap_s': overlap_s,
        'active_decode_overlap_s': active_decode_overlap_s,
        'overlapped': overlapped,
        'decoder_starved': starved,
        'baseline_decode_tok_s': parts['baseline']['decode_tok_s'],
        'concurrent_decode_tok_s': conc_metrics['decode_tok_s'],
        'concurrent_ttft_s': conc_metrics['ttft_s'],
        'max_stall_ms': conc_metrics['max_stall_ms'],
        'parts': parts,
        'reasons': reasons,
        'status': 'fail' if reasons else 'pass',
    }


def _parse_iso_epoch_s(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _stream_wall_interval(sinfo: Mapping[str, Any]) -> tuple[float, float] | None:
    ms, me = sinfo.get('mono_start_s'), sinfo.get('mono_end_s')
    if isinstance(ms, (int, float)) and isinstance(me, (int, float)) and not isinstance(ms, bool) and not isinstance(me, bool):
        if float(me) >= float(ms):
            return float(ms), float(me)
    ts, te = _parse_iso_epoch_s(sinfo.get('started_at')), _parse_iso_epoch_s(sinfo.get('ended_at'))
    if ts is not None and te is not None and te >= ts:
        return ts, te
    return None


def _stream_decode_interval(sinfo: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[float, float] | None:
    fs, ls = sinfo.get('first_event_mono_s'), sinfo.get('last_event_mono_s')
    if isinstance(fs, (int, float)) and isinstance(ls, (int, float)) and not isinstance(fs, bool) and not isinstance(ls, bool):
        if float(ls) >= float(fs):
            return float(fs), float(ls)
    ttft = metrics.get('ttft_s')
    dec = metrics.get('decode_s')
    if isinstance(ttft, (int, float)) and isinstance(dec, (int, float)):
        base = sinfo.get('mono_start_s')
        if isinstance(base, (int, float)) and not isinstance(base, bool):
            return float(base) + float(ttft), float(base) + float(ttft) + max(0.0, float(dec))
        ts = _parse_iso_epoch_s(sinfo.get('started_at'))
        if ts is not None:
            return ts + float(ttft), ts + float(ttft) + max(0.0, float(dec))
    return None


def _max_concurrency_and_span(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    """Sweep-line maximum overlap count and shared wall span across [start, end] intervals.

    End events sort before start events at the same timestamp so touching disjoint intervals [a, b] and [b, c]
    count as concurrency 1, not 2. Zero-width intervals [t, t] contribute 1 at instant t.
    """
    if not intervals:
        return 0, 0.0
    events: list[tuple[float, int]] = []
    min_s = min(s for s, _ in intervals)
    max_e = max(e for _, e in intervals)
    for start, end in intervals:
        if end > start:
            events.append((start, 1))
            events.append((end, -1))
        else:
            events.append((start, 1))
            events.append((start, 2))  # sentinel processed after start to decrement zero-width point
    # Order: real ends (-1) before starts (+1) before point-ends (2 -> -1) at equal timestamps.
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        if delta == 2:
            active -= 1
        else:
            active += delta
            if active > peak:
                peak = active
    return peak, round(max(0.0, max_e - min_s), 6)


def _eval_concurrency(campaign: Any, plan: Mapping[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
    if not recs:
        return {'status': 'not_run', 'levels': {}, 'pi_buckets': {}}

    levels_out: dict[str, dict[str, Any]] = {}
    level_attempts: dict[str, list[dict[str, Any]]] = {}

    for rec in recs:
        level = int(rec['concurrency'])
        starvation_ms = float(rec.get('starvation_threshold_ms', DEFAULT_STARVATION_MS))
        stream_reports: list[dict[str, Any]] = []
        all_gaps_ms: list[float] = []
        ttfts: list[float] = []
        tok_rates: list[float] = []
        wall_intervals: list[tuple[float, float]] = []
        decode_intervals: list[tuple[float, float]] = []
        total_completion_tokens = 0
        starvation_events = 0
        starved_streams = 0
        passed = 0

        for sinfo in rec.get('streams') or []:
            idx = sinfo['stream_index']
            reasons: list[str] = []
            body = campaign.parsed(rec, f's{idx}_request', strict_json)
            if sinfo.get('error') or sinfo.get('deadline_exceeded'):
                reasons.append(f'transport: {sinfo.get("error")}')
            try:
                reply = campaign.parsed(rec, f's{idx}_response', parse_sse)
            except ValueError as err:
                reply = None
                reasons.append(f'incomplete or malformed reply: {err}')
            meta = sinfo.get('meta') or {}
            exp_body, expected, _ = coding_turn_body(
                plan,
                meta.get('stream_tag', f'conc-c{level}-s{idx}'),
                0,
                chunks=meta.get('chunks', rec.get('chunks_per_stream', 2)),
                chunk_start=0,
            )
            if canonical(exp_body) != canonical(body):
                reasons.append('request does not match deterministic concurrency plan')
            correct = False
            if reply is not None:
                correct = workload.correct(expected, reply['message']['content'])
                if not correct:
                    reasons.append('wrong or missing concurrency answer')

            metrics = _stream_metrics(sinfo, reply, starvation_ms=starvation_ms)
            event_times = sinfo.get('event_times_s') or sinfo.get('token_times_s') or []
            all_gaps_ms.extend(round(max(0.0, (b - a) * 1000.0), 3) for a, b in zip(event_times, event_times[1:]))
            if metrics['ttft_s'] is not None:
                ttfts.append(metrics['ttft_s'])
            tok_rates.append(metrics['decode_tok_s'])
            total_completion_tokens += int(metrics['completion_tokens'])
            wall_iv = _stream_wall_interval(sinfo)
            if wall_iv is not None:
                wall_intervals.append(wall_iv)
            dec_iv = _stream_decode_interval(sinfo, metrics)
            if dec_iv is not None:
                decode_intervals.append(dec_iv)
            starvation_events += metrics['starvation_events']
            if metrics['starvation_events'] > 0:
                starved_streams += 1
            if not reasons:
                passed += 1
            stream_reports.append({
                'stream_index': idx,
                'correct': correct,
                'status': 'fail' if reasons else 'pass',
                'reasons': reasons,
                **metrics,
            })

        total = len(stream_reports)
        max_concurrent_wall, shared_wall_s = _max_concurrency_and_span(wall_intervals)
        max_concurrent_decode, shared_decode_s = _max_concurrency_and_span(decode_intervals)
        level_reasons: list[str] = []
        if total != level or passed != level:
            level_reasons.append(f'only {passed} of {level} streams passed (recorded {total})')
        if starved_streams > 0:
            level_reasons.append(f'{starved_streams} stream(s) experienced decode starvation (> {starvation_ms} ms)')
        if level > 1:
            if max_concurrent_wall < level:
                level_reasons.append(
                    f'streams did not execute concurrently: max_concurrent_streams={max_concurrent_wall} < {level}'
                )
            elif len(decode_intervals) == level and max_concurrent_decode < level:
                level_reasons.append(
                    f'streams did not overlap during active decode: '
                    f'max_concurrent_decode_streams={max_concurrent_decode} < {level}'
                )

        shared_elapsed_s = shared_decode_s if shared_decode_s > 0 else shared_wall_s
        aggregate_decode_tok_s = round(total_completion_tokens / shared_elapsed_s, 3) if shared_elapsed_s > 0 else 0.0
        max_rate = max(tok_rates) if tok_rates else 0.0
        min_rate = min(tok_rates) if tok_rates else 0.0
        fairness = round(min_rate / max_rate, 3) if max_rate > 0 else 0.0
        level_report = {
            'seq': rec['seq'],
            'concurrency': level,
            'pi_bucket': pi_bucket(level),
            'streams_total': total,
            'streams_passed': passed,
            'all_correct': total == level and passed == level,
            'max_concurrent_streams': max_concurrent_wall,
            'max_concurrent_decode_streams': max_concurrent_decode,
            'shared_elapsed_s': shared_elapsed_s,
            'aggregate_decode_tok_s': aggregate_decode_tok_s,
            'per_stream_decode_tok_s': {
                'min': round(min_rate, 3),
                'median': percentile(tok_rates, 50),
                'max': round(max_rate, 3),
            },
            'fairness_ratio': fairness,
            'ttft_s': {
                'p50': percentile(ttfts, 50),
                'p95': percentile(ttfts, 95),
                'p99': percentile(ttfts, 99),
                'max': round(max(ttfts), 6) if ttfts else None,
            },
            'itl_ms': {
                'count': len(all_gaps_ms),
                'p50': percentile(all_gaps_ms, 50),
                'p95': percentile(all_gaps_ms, 95),
                'p99': percentile(all_gaps_ms, 99),
                'max': round(max(all_gaps_ms), 3) if all_gaps_ms else 0.0,
            },
            'starvation_threshold_ms': starvation_ms,
            'starvation_events': starvation_events,
            'starved_streams': starved_streams,
            'reasons': level_reasons,
            'status': 'pass' if not level_reasons else 'fail',
            'streams': stream_reports,
        }
        _merge_attempt(levels_out, level_attempts, str(level), level_report)

    buckets_raw: dict[str, list[dict[str, Any]]] = {}
    for lvl_rep in levels_out.values():
        buckets_raw.setdefault(lvl_rep['pi_bucket'], []).append(lvl_rep)

    pi_buckets: dict[str, dict[str, Any]] = {}
    for bucket, entries in sorted(buckets_raw.items()):
        p95s = [e['itl_ms']['p95'] for e in entries if e['itl_ms']['p95'] is not None]
        p99s = [e['itl_ms']['p99'] for e in entries if e['itl_ms']['p99'] is not None]
        maxes = [e['itl_ms']['max'] for e in entries if e['itl_ms']['max'] is not None]
        pi_buckets[bucket] = {
            'levels': sorted(e['concurrency'] for e in entries),
            'all_passed': all(e['status'] == 'pass' for e in entries),
            'starvation_events': sum(e['starvation_events'] for e in entries),
            'starved_streams': sum(e['starved_streams'] for e in entries),
            'worst_itl_p95_ms': max(p95s) if p95s else None,
            'worst_itl_p99_ms': max(p99s) if p99s else None,
            'worst_itl_max_ms': max(maxes) if maxes else 0.0,
            'min_fairness_ratio': min(e['fairness_ratio'] for e in entries),
        }

    missing_levels = [lvl for lvl in CONCURRENCY_LEVELS if str(lvl) not in levels_out]
    rank = {'fail': 3, 'unknown': 2, 'pass': 1}
    statuses = [v['status'] for v in levels_out.values()] + (['unknown'] if missing_levels else [])
    return {
        'status': max(statuses, key=rank.__getitem__) if statuses else 'unknown',
        'missing_levels': missing_levels,
        'levels': levels_out,
        'pi_buckets': pi_buckets,
    }
