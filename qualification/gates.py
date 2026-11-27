"""Specialized v1.3.6 qualification gates: dense-to-sparse (2051/2052), durable images (#1285), and reasoning matrix.

All generators are deterministic functions of the campaign plan so the offline verdict re-parses recorded wire bytes,
regenerates expected requests/answers, requires full phase/case completion and affirmative persistence evidence, and
never lets a later retry erase an earlier failure.
"""
from __future__ import annotations

import base64
import re
from typing import Any, Callable, Mapping

from . import workload
from .transport import canonical, parse_sse, sha256, strict_json

# Flash DSA indexer geometry: 512 pools of 4 tokens + up to 3 trailing rows = 2,051 visible tokens (still dense!).
# 2,052 visible tokens is the exact first sparse transition; 2,048 is the 4th 512-row KV page boundary.
DSA_POOL_SIZE = 4
DSA_TOPK_POOLS = 512
DENSE_MAX_VISIBLE = DSA_TOPK_POOLS * DSA_POOL_SIZE + (DSA_POOL_SIZE - 1)  # 2,051
SPARSE_MIN_VISIBLE = DENSE_MAX_VISIBLE + 1  # 2,052
KV_PAGE_ROWS = 512
DENSE_RESTORE_PAGE_FLOOR = KV_PAGE_ROWS * 4  # 2,048 tokens: full 4-page dense prefix exercised before 2,052

DENSE_SPARSE_CASES = ('dense-2051', 'sparse-2052', 'page-reuse', 'restart-restore')
DURABLE_IMAGE_PHASES = ('initial', 'restart', 'cold-ref', 'mid-cut', 'swap-pixel', 'swap-geometry')
REASONING_EFFORTS = ('max', 'high', 'low', 'none')
SWAP_GEOMETRY_SIZE = (336, 224)  # distinct from 224x224 so vision token count & aspect geometry change

GATE_SYSTEM = {
    'dense-sparse': 'You are a careful records clerk. Answer only from the records in this conversation.',
    'image-durable': 'You describe images and records precisely. Name colors using only these words: '
                     + ', '.join(workload.PALETTE) + '.',
    'reasoning-matrix': 'You are a careful records clerk. Follow the output format strictly.',
}

# Shared identical prompt rendered at every reasoning_effort so an engine ignoring reasoning_effort always collides.
REASONING_RENDER_MESSAGES = (
    {'role': 'system', 'content': GATE_SYSTEM['reasoning-matrix']},
    {'role': 'user', 'content': 'Reasoning effort render check. Reply with Q1=ready.'},
)

# Filler tokens used for exact single-token prompt fitting (varied words avoid long single-token merges).
PAD_WORDS = ('amber', 'cedar', 'delta', 'ember', 'grove', 'inlet', 'marsh', 'notch', 'pebble', 'ridge', 'shoal', 'vale')


def fit_exact_tokens(
    count_fn: Callable[[int], int],
    target_tokens: int,
    *,
    lo_units: int = 0,
    hi_cap: int = 32_768,
    max_calls: int = 40,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Find padding unit count hitting exact target_tokens (or closest >= target if single-unit step > 1)."""
    memo: dict[int, int] = {}

    def c(n: int) -> int:
        if n not in memo:
            if len(memo) >= max_calls:
                raise RuntimeError(f'fit_exact_tokens used {max_calls} /tokenize calls without converging')
            memo[n] = count_fn(n)
        return memo[n]

    base = c(lo_units)
    if base > target_tokens:
        raise ValueError(f'base prompt ({base} tokens) already exceeds exact target {target_tokens}')
    if base == target_tokens:
        return lo_units, base, sorted(memo.items())

    probe = min(hi_cap, max(lo_units + 8, lo_units + 64))
    per = max(0.25, (c(probe) - base) / max(1, probe - lo_units))
    hi = min(hi_cap, max(lo_units + 1, lo_units + int((target_tokens - base) / per) + 4))
    while c(hi) < target_tokens:
        if hi >= hi_cap:
            raise ValueError(f'{hi_cap} padding units reached only {c(hi)} tokens, below {target_tokens}')
        hi = min(hi_cap, hi * 2)
    lo = lo_units
    while hi - lo > 1:
        mid = (lo + hi) // 2
        val = c(mid)
        if val == target_tokens:
            return mid, val, sorted(memo.items())
        if val < target_tokens:
            lo = mid
        else:
            hi = mid
    for candidate in (lo, hi):
        if c(candidate) == target_tokens:
            return candidate, target_tokens, sorted(memo.items())
    return hi, c(hi), sorted(memo.items())


def _pad_text(seed: str, label: str, units: int) -> str:
    if units <= 0:
        return ''
    det = workload.Det(seed, f'pad/{label}')
    words = [PAD_WORDS[det.below(len(PAD_WORDS))] for _ in range(units)]
    return ' '.join(words)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    """Exact shared prefix length between two rendered token-ID sequences."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------------------------------------------------------------------------------------------------------------
# Gate 1: Dense-to-sparse transition (2,051 vs 2,052 visible tokens) + page-reuse + restart-restore


def dense_sparse_facts(seed: str, case: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Head fact (page 0) and tail fact (partial final page past row 2048) for a dense/sparse case."""
    head_idx = 9001 if case == 'dense-2051' else 9002
    tail_idx = 9101 if case == 'dense-2051' else 9102
    return workload.fact(seed, 'dense', head_idx), workload.fact(seed, 'sparse', tail_idx)


def dense_sparse_turn(
    plan: Mapping[str, Any],
    case: str,
    *,
    pad_units: int = 0,
    history: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Build deterministic chat request + expected answers for a dense-to-sparse gate step."""
    if case not in DENSE_SPARSE_CASES:
        raise ValueError(f'unknown dense-sparse case {case!r}')
    seed = plan['seed']
    if case in ('dense-2051', 'sparse-2052'):
        (head_vault, head_code), (tail_vault, tail_code) = dense_sparse_facts(seed, case)
        padding = _pad_text(seed, case, pad_units)
        content = (
            f'Header record: the archive code for vault {head_vault} is {head_code}.\n'
            + (f'Notes: {padding}.\n' if padding else '')
            + f'Tail record: the archive code for vault {tail_vault} is {tail_code}.\n'
            + 'Questions:\n'
            + f'Q1: What is the archive code for vault {head_vault}?\n'
            + f'Q2: What is the archive code for vault {tail_vault}?\n'
            + workload.FORMAT
        )
        messages = [
            {'role': 'system', 'content': GATE_SYSTEM['dense-sparse']},
            {'role': 'user', 'content': content},
        ]
        expected = {'1': head_code, '2': tail_code}
        target = DENSE_MAX_VISIBLE if case == 'dense-2051' else SPARSE_MIN_VISIBLE
        meta = {'gate': 'dense-sparse', 'case': case, 'pad_units': pad_units, 'target_tokens': target}
    else:
        if not history:
            raise ValueError(f'dense-sparse case {case!r} requires completed sparse-2052 history')
        last_body, last_msg = history[-1]
        messages = list(last_body['messages']) + [workload.replay(last_msg)]
        (head_vault, head_code), (tail_vault, tail_code) = dense_sparse_facts(seed, 'sparse-2052')
        extra_vault, extra_code = workload.fact(seed, 'sparse', 9201 if case == 'page-reuse' else 9202)
        padding = _pad_text(seed, case, pad_units)
        content = (
            f'Follow-up ({case}): the archive code for vault {extra_vault} is {extra_code}.\n'
            + (f'Notes: {padding}.\n' if padding else '')
            + 'Questions:\n'
            + f'Q1: What is the archive code for vault {head_vault}?\n'
            + f'Q2: What is the archive code for vault {tail_vault}?\n'
            + f'Q3: What is the archive code for vault {extra_vault}?\n'
            + workload.FORMAT
        )
        messages.append({'role': 'user', 'content': content})
        expected = {'1': head_code, '2': tail_code, '3': extra_code}
        meta = {'gate': 'dense-sparse', 'case': case, 'pad_units': pad_units, 'target_tokens': SPARSE_MIN_VISIBLE + 1}

    body = workload.chat_body(plan, 'dense-sparse', messages)
    body['temperature'] = 0
    return body, expected, meta


# ---------------------------------------------------------------------------------------------------------------------
# Gate 2: Durable image conversation gate (#1285)


def durable_image_spec(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Two-image durable spec: original img1/img2 plus pixel-swapped and geometry-swapped img2 variants."""
    img1 = list(plan['images'][0])
    img2 = list(plan['images'][1])
    img2_pixel_swap = img2[1:] + img2[:1]  # identical 224x224 geometry, rotated quadrant colors
    img2_geom_swap = img2[2:] + img2[:2]  # different 336x224 geometry AND distinct quadrant colors
    return {
        'image1': {'colors': img1, 'size': (workload.IMAGE_SIZE, workload.IMAGE_SIZE)},
        'image2': {'colors': img2, 'size': (workload.IMAGE_SIZE, workload.IMAGE_SIZE)},
        'image2_pixel_swap': {'colors': img2_pixel_swap, 'size': (workload.IMAGE_SIZE, workload.IMAGE_SIZE)},
        'image2_geometry_swap': {'colors': img2_geom_swap, 'size': SWAP_GEOMETRY_SIZE},
    }


def _image_url(colors: list[str], size: tuple[int, int]) -> str:
    raw = workload.png(colors, size=size[0], height=size[1])
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def durable_image_prefix_bodies(
    plan: Mapping[str, Any],
    *,
    mid_pad_units: int,
    variant: str = 'original',
) -> dict[str, dict[str, Any]]:
    """Build exact /v1/tokenize prefix bodies for image1 end, pre-image2 boundary, and image2 end."""
    spec = durable_image_spec(plan)
    seed = plan['seed']
    img2_cfg = (
        spec['image2_pixel_swap']
        if variant == 'swap-pixel'
        else spec['image2_geometry_swap']
        if variant == 'swap-geometry'
        else spec['image2']
    )
    mid_vault, mid_code = workload.fact(seed, 'imgdur', 1)
    mid_pad = _pad_text(seed, 'imgdur-mid', mid_pad_units)
    mid_text = f'Mid-section record: vault {mid_vault} is {mid_code}.' + (f' {mid_pad}.' if mid_pad else '')

    part_img1 = [
        {'type': 'text', 'text': 'Image 1:'},
        {'type': 'image_url', 'image_url': {'url': _image_url(spec['image1']['colors'], spec['image1']['size'])}},
    ]
    part_pre_img2 = part_img1 + [{'type': 'text', 'text': mid_text + '\nImage 2:'}]
    part_img2 = part_pre_img2 + [
        {'type': 'image_url', 'image_url': {'url': _image_url(img2_cfg['colors'], img2_cfg['size'])}},
    ]

    def wrap(parts: list[dict[str, Any]]) -> dict[str, Any]:
        return workload.chat_body(
            plan,
            'image-durable',
            [{'role': 'system', 'content': GATE_SYSTEM['image-durable']}, {'role': 'user', 'content': parts}],
        )

    return {
        'image1-prefix': wrap(part_img1),
        'pre-image2-prefix': wrap(part_pre_img2),
        'image2-prefix': wrap(part_img2),
    }


def durable_image_turn(
    plan: Mapping[str, Any],
    phase: str,
    *,
    mid_pad_units: int = 60,
    tail_pad_units: int = 0,
    history: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Build deterministic request, expected answers, and metadata for each #1285 durable-image gate phase."""
    if phase not in DURABLE_IMAGE_PHASES:
        raise ValueError(f'unknown durable-image phase {phase!r}')
    spec = durable_image_spec(plan)
    seed = plan['seed']
    mid_vault, mid_code = workload.fact(seed, 'imgdur', 1)
    tail_vault, tail_code = workload.fact(seed, 'imgdur', 2)
    mid_pad = _pad_text(seed, 'imgdur-mid', mid_pad_units)
    tail_pad = _pad_text(seed, 'imgdur-tail', tail_pad_units)
    mid_text = f'Mid-section record: vault {mid_vault} is {mid_code}.' + (f' {mid_pad}.' if mid_pad else '')
    tail_text = f'Tail record: vault {tail_vault} is {tail_code}.' + (f' {tail_pad}.' if tail_pad else '')

    if phase in ('initial', 'swap-pixel', 'swap-geometry'):
        img2_cfg = (
            spec['image2_pixel_swap']
            if phase == 'swap-pixel'
            else spec['image2_geometry_swap']
            if phase == 'swap-geometry'
            else spec['image2']
        )
        # Questions use identical text across initial / swap-pixel / swap-geometry so only image 2 differs.
        q_text = (
            f'{tail_text}\nQuestions:\n'
            f'Q1: In image 1, what color is the {workload.QUADRANTS[0]} quadrant?\n'
            f'Q2: In image 2, what color is the {workload.QUADRANTS[1]} quadrant?\n'
            f'Q3: What is the archive code for vault {tail_vault}?\n'
            + workload.FORMAT
        )
        parts = [
            {'type': 'text', 'text': 'Image 1:'},
            {'type': 'image_url', 'image_url': {'url': _image_url(spec['image1']['colors'], spec['image1']['size'])}},
            {'type': 'text', 'text': mid_text + '\nImage 2:'},
            {'type': 'image_url', 'image_url': {'url': _image_url(img2_cfg['colors'], img2_cfg['size'])}},
            {'type': 'text', 'text': q_text},
        ]
        messages = [
            {'role': 'system', 'content': GATE_SYSTEM['image-durable']},
            {'role': 'user', 'content': parts},
        ]
        expected = {'1': spec['image1']['colors'][0], '2': img2_cfg['colors'][1], '3': tail_code}
        meta = {
            'gate': 'image-durable',
            'phase': phase,
            'mid_pad_units': mid_pad_units,
            'tail_pad_units': tail_pad_units,
            'expected_image1_colors': spec['image1']['colors'],
            'expected_image2_colors': img2_cfg['colors'],
            'expected_image2_size': list(img2_cfg['size']),
        }
    elif phase == 'mid-cut':
        # Identical prefix through image 1 + mid_text, then branches BEFORE image 2 (mid-image sequence cut).
        q_text = (
            f'{mid_text}\nQuestions:\n'
            f'Q1: In image 1, what color is the {workload.QUADRANTS[2]} quadrant?\n'
            f'Q2: What is the archive code for vault {mid_vault}?\n'
            + workload.FORMAT
        )
        parts = [
            {'type': 'text', 'text': 'Image 1:'},
            {'type': 'image_url', 'image_url': {'url': _image_url(spec['image1']['colors'], spec['image1']['size'])}},
            {'type': 'text', 'text': q_text},
        ]
        messages = [
            {'role': 'system', 'content': GATE_SYSTEM['image-durable']},
            {'role': 'user', 'content': parts},
        ]
        expected = {'1': spec['image1']['colors'][2], '2': mid_code}
        meta = {
            'gate': 'image-durable',
            'phase': phase,
            'mid_pad_units': mid_pad_units,
            'expected_image1_colors': spec['image1']['colors'],
        }
    else:
        # restart and cold-ref ask the exact same continuation questions so outputs can be compared verbatim.
        if not history:
            raise ValueError(f'durable-image phase {phase!r} requires initial conversation history')
        last_body, last_msg = history[-1]
        messages = list(last_body['messages']) + [workload.replay(last_msg)]
        q_text = (
            'Continuation questions:\n'
            f'Q1: In image 1, what color is the {workload.QUADRANTS[3]} quadrant?\n'
            f'Q2: In image 2, what color is the {workload.QUADRANTS[2]} quadrant?\n'
            f'Q3: What is the archive code for vault {mid_vault}?\n'
            + workload.FORMAT
        )
        messages.append({'role': 'user', 'content': q_text})
        expected = {'1': spec['image1']['colors'][3], '2': spec['image2']['colors'][2], '3': mid_code}
        meta = {
            'gate': 'image-durable',
            'phase': phase,
            'mid_pad_units': mid_pad_units,
            'tail_pad_units': tail_pad_units,
        }

    body = workload.chat_body(plan, 'image-durable', messages)
    body['temperature'] = 0
    return body, expected, meta


# ---------------------------------------------------------------------------------------------------------------------
# Gate 3: Reasoning & effort matrix (CLEAR_THINKING=0 + max, high, low, none)


def reasoning_render_body(plan: Mapping[str, Any], effort: str) -> dict[str, Any]:
    """Identical shared messages rendered at `effort` so /v1/tokenize proves effort actually changes the prompt."""
    if effort not in REASONING_EFFORTS:
        raise ValueError(f'effort must be one of {REASONING_EFFORTS}, got {effort!r}')
    return {
        'model': plan['model'],
        'messages': [dict(m) for m in REASONING_RENDER_MESSAGES],
        'reasoning_effort': effort,
    }


def reasoning_matrix_turn(
    plan: Mapping[str, Any],
    effort: str,
    step: str,
    *,
    history: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Build turn 1 or turn 2 (replaying reasoning_content verbatim for CLEAR_THINKING=0) for one effort."""
    if effort not in REASONING_EFFORTS:
        raise ValueError(f'effort must be one of {REASONING_EFFORTS}, got {effort!r}')
    if step not in ('turn-1', 'turn-2'):
        raise ValueError(f'step must be turn-1 or turn-2, got {step!r}')
    idx = REASONING_EFFORTS.index(effort)
    v1, c1 = workload.fact(plan['seed'], 'reason', 100 + idx)
    v2, c2 = workload.fact(plan['seed'], 'reason', 200 + idx)

    if step == 'turn-1':
        messages = [
            {'role': 'system', 'content': GATE_SYSTEM['reasoning-matrix']},
            {
                'role': 'user',
                'content': (
                    f'Record A: the archive code for vault {v1} is {c1}.\n'
                    f'Record B: the archive code for vault {v2} is {c2}.\n'
                    'Questions:\n'
                    f'Q1: What is the archive code for vault {v1}?\n'
                    + workload.FORMAT
                ),
            },
        ]
        expected = {'1': c1}
    else:
        if not history:
            raise ValueError('reasoning-matrix turn-2 requires turn-1 history')
        last_body, last_msg = history[-1]
        messages = list(last_body['messages']) + [workload.replay(last_msg)]
        messages.append({
            'role': 'user',
            'content': (
                'Questions:\n'
                f'Q1: What is the archive code for vault {v1}?\n'
                f'Q2: What is the archive code for vault {v2}?\n'
                + workload.FORMAT
            ),
        })
        expected = {'1': c1, '2': c2}

    body = {
        'model': plan['model'],
        'messages': messages,
        'stream': True,
        'stream_options': {'include_usage': True},
        'max_tokens': plan['max_tokens'],
        'reasoning_effort': effort,
        'temperature': 0,
    }
    meta = {'gate': 'reasoning-matrix', 'effort': effort, 'step': step}
    return body, expected, meta


# ---------------------------------------------------------------------------------------------------------------------
# Offline verdict evaluators (wire-byte re-parsing; never trust client_observed; never lose failed attempts)


def _decode_request_images(body: Mapping[str, Any]) -> list[tuple[tuple[int, int], list[str]]]:
    images: list[tuple[tuple[int, int], list[str]]] = []
    for msg in body.get('messages') or []:
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'image_url':
                raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                images.append((workload.png_dimensions(raw), workload.quadrant_colors(raw)))
    return images


def _record_attempt(
    store: dict[str, dict[str, Any]],
    attempts: dict[str, list[dict[str, Any]]],
    key: str,
    entry: dict[str, Any],
) -> None:
    """Keep every attempt; a later passing retry never hides an earlier failure."""
    attempts.setdefault(key, []).append(entry)
    failed = [a['seq'] for a in attempts[key] if a['status'] == 'fail']
    merged = dict(entry, failed_attempts=failed)
    if failed:
        merged['status'] = 'fail'
    store[key] = merged


def _check_store_reopened(
    campaign: Any,
    log_lines: list[dict[str, Any]],
    *,
    p_end: Any,
    c_start: Any,
    c_end: Any,
    skew: Any,
    covered_fn: Callable[[Any, Any], bool],
) -> tuple[str, list[str]]:
    stores = [x for x in log_lines if x['kind'] == 'store' and p_end < x['ts'] <= c_start + skew]
    learn_dir = campaign.manifest['profile']['learn_dir'].rstrip('/')
    ranks = campaign.manifest['ranks']
    good = [
        x for x in stores
        if re.fullmatch(re.escape(learn_dir) + r'/[0-9a-f]{16}', x['groups'][0])
        and int(x['groups'][4]) == ranks
        and int(x['groups'][1]) >= 1
    ]
    if good:
        return 'pass', [x['line'] for x in stores]
    return ('fail' if (stores or covered_fn(c_start, c_end)) else 'unknown'), [x['line'] for x in stores]


def evaluate_gates(
    campaign: Any,
    receipts: list[dict[str, Any]],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    measure_tokens: Mapping[tuple[str, str], list[int]] | None = None,
    measure_history: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    log_lines: list[dict[str, Any]] = (),
    ambiguous_reads: set[tuple[Any, Any]] | None = None,
    skew: Any = None,
    find_logs: Callable[[str, Any, Any, int, int], list[dict[str, Any]]],
    covered_fn: Callable[[Any, Any], bool],
    when_fn: Callable[[str], Any],
    tolerance: int,
) -> dict[str, Any]:
    """Offline evaluation of dense-sparse, image-durable (#1285), and reasoning-matrix gate receipts."""
    gate_receipts = [r for r in receipts if r.get('kind') == 'gate']
    if not gate_receipts:
        return {'status': 'not_run', 'dense_sparse': {'status': 'not_run'},
                'image_durable': {'status': 'not_run'}, 'reasoning_matrix': {'status': 'not_run'}}

    by_gate: dict[str, list[dict[str, Any]]] = {}
    for r in gate_receipts:
        by_gate.setdefault(r.get('gate', ''), []).append(r)

    amb_set = ambiguous_reads if ambiguous_reads is not None else set()
    dense_sparse = _eval_dense_sparse(
        campaign, by_gate.get('dense-sparse', []), receipts, measures,
        log_lines=list(log_lines), ambiguous_reads=amb_set, skew=skew,
        find_logs=find_logs, covered_fn=covered_fn, when_fn=when_fn,
    )
    image_durable = _eval_image_durable(
        campaign, by_gate.get('image-durable', []), receipts, measures,
        measure_tokens=measure_tokens or {}, log_lines=list(log_lines),
        ambiguous_reads=amb_set, skew=skew,
        find_logs=find_logs, covered_fn=covered_fn, when_fn=when_fn, tolerance=tolerance,
    )
    reasoning = _eval_reasoning_matrix(
        campaign, by_gate.get('reasoning-matrix', []), receipts, measures,
        measure_history=measure_history or {},
    )

    active = [g['status'] for g in (dense_sparse, image_durable, reasoning) if g['status'] != 'not_run']
    rank = {'invalid': 4, 'fail': 3, 'unknown': 2, 'pass': 1}
    overall = max(active, key=rank.__getitem__) if active else 'not_run'
    return {
        'status': overall,
        'dense_sparse': dense_sparse,
        'image_durable': image_durable,
        'reasoning_matrix': reasoning,
    }


def _reparse_gate_chat(campaign: Any, rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    body = campaign.parsed(rec, 'request', strict_json)
    if rec.get('error') or rec.get('deadline_exceeded'):
        reasons.append(f'transport: {rec.get("error")}')
    try:
        reply = campaign.parsed(rec, 'response', parse_sse)
    except ValueError as err:
        reply = None
        reasons.append(f'incomplete or malformed reply: {err}')
    return body, reply, reasons


def _eval_dense_sparse(
    campaign: Any,
    recs: list[dict[str, Any]],
    all_receipts: list[dict[str, Any]],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    log_lines: list[dict[str, Any]],
    ambiguous_reads: set[tuple[Any, Any]],
    skew: Any,
    find_logs: Callable[[str, Any, Any, int, int], list[dict[str, Any]]],
    covered_fn: Callable[[Any, Any], bool],
    when_fn: Callable[[str], Any],
) -> dict[str, Any]:
    if not recs:
        return {'status': 'not_run', 'cases': {}}
    plan = campaign.plan
    results: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    sparse_history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    first_sparse_receipt: dict[str, Any] | None = None
    last_prior_receipt: dict[str, Any] | None = None
    last_prior_completion: int = 0

    for rec in recs:
        case = rec.get('case', '')
        body, reply, reasons = _reparse_gate_chat(campaign, rec)
        meta = rec.get('meta') or {}
        measured = measures.get(('measure', sha256(canonical(workload.tokenize_body(body)))))
        measured_count = measured['count'] if measured else None

        expected: dict[str, str] | None = None
        try:
            hist = sparse_history if case in ('page-reuse', 'restart-restore') else None
            exp_body, expected, _ = dense_sparse_turn(plan, case, pad_units=meta.get('pad_units', 0), history=hist)
            if canonical(exp_body) != canonical(body):
                reasons.append('request does not match deterministic dense-sparse plan')
        except (ValueError, KeyError) as err:
            reasons.append(f'request cannot be regenerated: {err}')

        if measured_count is None:
            reasons.append('missing /v1/tokenize measurement for dense-sparse case')
        elif case == 'dense-2051' and measured_count != DENSE_MAX_VISIBLE:
            reasons.append(f'dense-2051 measured {measured_count} tokens, expected exact {DENSE_MAX_VISIBLE}')
        elif case == 'sparse-2052' and measured_count != SPARSE_MIN_VISIBLE:
            reasons.append(f'sparse-2052 measured {measured_count} tokens, expected exact {SPARSE_MIN_VISIBLE}')
        elif case == 'page-reuse':
            if measured_count <= SPARSE_MIN_VISIBLE:
                reasons.append(f'page-reuse measured {measured_count} tokens, expected > {SPARSE_MIN_VISIBLE}')
            elif measured_count >= DENSE_RESTORE_PAGE_FLOOR + KV_PAGE_ROWS:
                reasons.append(
                    f'page-reuse measured {measured_count} tokens, expected inside partial page 4 '
                    f'(< {DENSE_RESTORE_PAGE_FLOOR + KV_PAGE_ROWS})'
                )
        elif case == 'restart-restore' and measured_count <= SPARSE_MIN_VISIBLE:
            reasons.append(f'{case} measured {measured_count} tokens, expected > {SPARSE_MIN_VISIBLE}')

        correct = False
        if reply is not None and expected is not None:
            correct = workload.correct(expected, reply['message']['content'])
            if not correct:
                reasons.append('wrong or missing dense-sparse answer')

        extra: dict[str, Any] = {}
        unknown_evidence = False
        if case == 'page-reuse':
            floor = DENSE_RESTORE_PAGE_FLOOR
            cached = reply.get('cached_tokens') if reply is not None else None
            c_start, c_end = when_fn(rec['started_at']), when_fn(rec['ended_at'])
            reads = (
                find_logs('read', c_start, c_end, floor, measured_count)
                if measured_count is not None
                else []
            )
            reuse_scope = 'unknown'
            reuse_evidence: str | None = None
            if measured_count is not None and cached is not None and floor <= cached <= measured_count:
                reuse_scope = 'in_memory_partial_page_4'
                reuse_evidence = 'usage.cached_tokens'
            elif reads:
                reuse_scope = 'disk_partial_page_4'
                reuse_evidence = 'log.imprint_read'
            else:
                if cached is not None:
                    reasons.append(
                        f'page-reuse reported cached_tokens={cached} below partial-page floor {floor} '
                        '(prefix was recomputed rather than reused)'
                    )
                else:
                    reasons.append(
                        f'no attributable reuse evidence (cached_tokens or disk read >= {floor}) for page-reuse'
                    )
            extra.update(
                floor_tokens=floor,
                cached_tokens=cached,
                reads=[x['line'] for x in reads],
                reuse_evidence=reuse_evidence,
                reuse_scope=reuse_scope,
            )
        elif case == 'restart-restore':
            if last_prior_receipt is None or measured_count is None:
                reasons.append('restart-restore requires a prior completed sparse turn with measured tokens')
            else:
                prior_body = campaign.parsed(last_prior_receipt, 'request', strict_json)
                prior_m = measures.get(('measure', sha256(canonical(workload.tokenize_body(prior_body)))))
                prior_count = prior_m['count'] if prior_m else SPARSE_MIN_VISIBLE
                events = [e for e in all_receipts if e.get('kind') == 'lifecycle' and e.get('event') == 'restart'
                          and last_prior_receipt['seq'] < e['seq'] < rec['seq']]
                if not events:
                    reasons.append('no recorded restart lifecycle event before dense-sparse restart-restore')
                else:
                    save_start = when_fn((first_sparse_receipt or last_prior_receipt)['started_at'])
                    p_end = when_fn(last_prior_receipt['ended_at'])
                    at = when_fn(events[-1]['at'])
                    c_start, c_end = when_fn(rec['started_at']), when_fn(rec['ended_at'])
                    # Floor tied to the 4-page dense boundary (2,048 tokens) exercised by sparse-2052.
                    floor = max(DENSE_RESTORE_PAGE_FLOOR, prior_count - KV_PAGE_ROWS)
                    held = prior_count + last_prior_completion
                    saves = find_logs('save', save_start, min(at, c_start), floor, held)
                    skips = find_logs('skip', save_start, min(at, c_start), floor, held)
                    reads = find_logs('read', c_start, c_end, floor, measured_count)
                    store_status, store_lines = _check_store_reopened(
                        campaign, log_lines, p_end=p_end, c_start=c_start, c_end=c_end, skew=skew, covered_fn=covered_fn,
                    )
                    extra.update(
                        floor_tokens=floor,
                        saves=[x['line'] for x in saves],
                        skipped=[x['line'] for x in skips],
                        reads=[x['line'] for x in reads],
                        store_opened=store_lines,
                    )
                    if not saves:
                        if skips:
                            reasons.append('dense-sparse save was skipped or failed before restart')
                        elif covered_fn(save_start, min(at, c_start)):
                            reasons.append('no retained save line covering dense-sparse state before restart')
                        else:
                            unknown_evidence = True
                    if store_status == 'fail':
                        reasons.append('no valid store-open line after restart for dense-sparse restore')
                    elif store_status == 'unknown':
                        unknown_evidence = True
                    if not reads:
                        if (c_start, c_end) in ambiguous_reads:
                            unknown_evidence = True
                        elif covered_fn(c_start, c_end):
                            reasons.append('no disk read >= 2048 tokens logged during dense-sparse restart-restore (cold prefill)')
                        else:
                            unknown_evidence = True

        status = 'fail' if reasons else ('unknown' if unknown_evidence else 'pass')
        entry = {
            'seq': rec['seq'],
            'measured_prompt_tokens': measured_count,
            'correct': correct,
            'status': status,
            'reasons': reasons,
            **extra,
        }
        _record_attempt(results, attempts, case, entry)

        if case == 'sparse-2052' and reply is not None:
            sparse_history = [(body, reply['message'])]
            first_sparse_receipt = rec
            last_prior_receipt = rec
            last_prior_completion = reply['usage']['completion_tokens']
        elif case == 'page-reuse' and reply is not None:
            sparse_history.append((body, reply['message']))
            last_prior_receipt = rec
            last_prior_completion = reply['usage']['completion_tokens']

    missing = [c for c in DENSE_SPARSE_CASES if c not in results]
    rank = {'fail': 3, 'unknown': 2, 'pass': 1}
    statuses = [v['status'] for v in results.values()] + (['unknown'] if missing else [])
    return {
        'status': max(statuses, key=rank.__getitem__) if statuses else 'unknown',
        'dense_limit_tokens': DENSE_MAX_VISIBLE,
        'sparse_min_tokens': SPARSE_MIN_VISIBLE,
        'restore_page_floor': DENSE_RESTORE_PAGE_FLOOR,
        'missing_required': missing,
        'cases': results,
    }


def _eval_image_durable(
    campaign: Any,
    recs: list[dict[str, Any]],
    all_receipts: list[dict[str, Any]],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    measure_tokens: Mapping[tuple[str, str], list[int]],
    log_lines: list[dict[str, Any]],
    ambiguous_reads: set[tuple[Any, Any]],
    skew: Any,
    find_logs: Callable[[str, Any, Any, int, int], list[dict[str, Any]]],
    covered_fn: Callable[[Any, Any], bool],
    when_fn: Callable[[str], Any],
    tolerance: int,
) -> dict[str, Any]:
    if not recs:
        return {'status': 'not_run', 'phases': {}}
    plan = campaign.plan
    spec = durable_image_spec(plan)
    results: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    initial_history: list[tuple[dict[str, Any], dict[str, Any]]] = []
    initial_receipt: dict[str, Any] | None = None
    initial_measured: int | None = None
    initial_tokens: list[int] | None = None
    initial_completion: int = 0
    initial_mid_pad = 60
    initial_tail_pad = 0
    prefix_counts: dict[str, int] = {}
    restart_receipt: dict[str, Any] | None = None
    restart_replies: list[dict[str, Any]] = []
    cold_ref_replies: list[dict[str, Any]] = []

    for rec in recs:
        phase = rec.get('phase', '')
        body, reply, reasons = _reparse_gate_chat(campaign, rec)
        meta = rec.get('meta') or {}
        tok_req_hash = sha256(canonical(workload.tokenize_body(body)))
        measured = measures.get(('measure', tok_req_hash))
        measured_count = measured['count'] if measured else None
        tok_ids = measure_tokens.get(('measure', tok_req_hash))

        mid_pad = meta.get('mid_pad_units', initial_mid_pad)
        tail_pad = meta.get('tail_pad_units', initial_tail_pad)
        expected: dict[str, str] | None = None
        try:
            hist = initial_history if phase in ('restart', 'cold-ref') else None
            exp_body, expected, _ = durable_image_turn(
                plan, phase, mid_pad_units=mid_pad, tail_pad_units=tail_pad, history=hist
            )
            if canonical(exp_body) != canonical(body):
                reasons.append(f'request does not match deterministic image-durable/{phase} plan')
        except (ValueError, KeyError) as err:
            reasons.append(f'request cannot be regenerated: {err}')

        decoded_images: list[tuple[tuple[int, int], list[str]]] = []
        if phase in ('initial', 'mid-cut', 'swap-pixel', 'swap-geometry'):
            try:
                decoded_images = _decode_request_images(body)
            except (ValueError, KeyError, TypeError, IndexError) as err:
                reasons.append(f'unreadable PNG in request: {err}')

        if phase == 'initial':
            initial_mid_pad = mid_pad
            initial_tail_pad = tail_pad
            if measured_count is None:
                reasons.append('missing /v1/tokenize measurement for image-durable initial')
            elif measured_count <= SPARSE_MIN_VISIBLE:
                reasons.append(
                    f'image-durable initial measured {measured_count} tokens, must cross sparse transition (>{SPARSE_MIN_VISIBLE})'
                )
            if len(decoded_images) != 2 or decoded_images[0] != (spec['image1']['size'], spec['image1']['colors']) or \
                    decoded_images[1] != (spec['image2']['size'], spec['image2']['colors']):
                reasons.append('initial request does not carry the two planned 224x224 quadrant PNGs')
            prefixes = durable_image_prefix_bodies(plan, mid_pad_units=mid_pad, variant='original')
            for pname, pbody in prefixes.items():
                pm = measures.get((pname, sha256(canonical(workload.tokenize_body(pbody, add_generation_prompt=False)))))
                if pm is not None:
                    prefix_counts[pname] = pm['count']
            if 'image2-prefix' not in prefix_counts or 'pre-image2-prefix' not in prefix_counts:
                reasons.append('missing image2-prefix / pre-image2-prefix /v1/tokenize receipts')
        elif phase == 'mid-cut':
            if len(decoded_images) != 1 or decoded_images[0] != (spec['image1']['size'], spec['image1']['colors']):
                reasons.append('mid-cut request must contain only image 1')
        elif phase == 'swap-pixel':
            if len(decoded_images) != 2 or decoded_images[0] != (spec['image1']['size'], spec['image1']['colors']) or \
                    decoded_images[1] != (spec['image2_pixel_swap']['size'], spec['image2_pixel_swap']['colors']):
                reasons.append('swap-pixel request images do not match pixel-swapped image 2')
        elif phase == 'swap-geometry':
            if len(decoded_images) != 2 or decoded_images[0] != (spec['image1']['size'], spec['image1']['colors']) or \
                    decoded_images[1] != (spec['image2_geometry_swap']['size'], spec['image2_geometry_swap']['colors']):
                reasons.append('swap-geometry request images do not match 336x224 geometry-swapped image 2')

        correct = False
        if reply is not None and expected is not None:
            correct = workload.correct(expected, reply['message']['content'])
            if not correct:
                reasons.append(f'wrong or missing answer for image-durable/{phase}')

        extra: dict[str, Any] = {}
        unknown_evidence = False
        if phase == 'restart':
            restart_receipt = rec
            if reply is not None:
                restart_replies.append(reply)
            if initial_receipt is None or initial_measured is None or 'image2-prefix' not in prefix_counts:
                reasons.append('restart phase requires completed initial image-durable receipt with prefix measures')
            else:
                events = [e for e in all_receipts if e.get('kind') == 'lifecycle' and e.get('event') == 'restart'
                          and initial_receipt['seq'] < e['seq'] < rec['seq']]
                if not events:
                    reasons.append('no recorded restart lifecycle event before image-durable/restart')
                else:
                    p_start = when_fn(initial_receipt['started_at'])
                    p_end = when_fn(initial_receipt['ended_at'])
                    at = when_fn(events[-1]['at'])
                    c_start, c_end = when_fn(rec['started_at']), when_fn(rec['ended_at'])
                    floor = max(initial_measured - tolerance, prefix_counts['image2-prefix'])
                    held = initial_measured + initial_completion
                    saves = find_logs('save', p_start, min(at, c_start), floor, held)
                    skips = find_logs('skip', p_start, min(at, c_start), floor, held)
                    reads = find_logs('read', c_start, c_end, floor, measured_count or held)
                    store_status, store_lines = _check_store_reopened(
                        campaign, log_lines, p_end=p_end, c_start=c_start, c_end=c_end, skew=skew, covered_fn=covered_fn,
                    )
                    extra.update(
                        floor_tokens=floor,
                        saves=[x['line'] for x in saves],
                        skipped=[x['line'] for x in skips],
                        reads=[x['line'] for x in reads],
                        store_opened=store_lines,
                    )
                    if not saves:
                        if skips:
                            reasons.append('image-durable save was skipped or failed before restart')
                        elif covered_fn(p_start, min(at, c_start)):
                            reasons.append('no retained save line covering image-durable state before restart')
                        else:
                            unknown_evidence = True
                    if store_status == 'fail':
                        reasons.append('no valid store-open line after restart for image-durable restore')
                    elif store_status == 'unknown':
                        unknown_evidence = True
                    if not reads:
                        if (c_start, c_end) in ambiguous_reads:
                            unknown_evidence = True
                        elif covered_fn(c_start, c_end):
                            reasons.append('no disk read covering both images after restart (cold prefill)')
                        else:
                            unknown_evidence = True
        elif phase == 'cold-ref':
            if reply is not None:
                cold_ref_replies.append(reply)
            c_start, c_end = when_fn(rec['started_at']), when_fn(rec['ended_at'])
            # Verify cold-ref was isolated from the warm restart instance (different endpoint or intervening restart).
            isolated = True
            if restart_receipt is not None and rec.get('endpoint') == restart_receipt.get('endpoint'):
                between = [
                    e for e in all_receipts
                    if e.get('kind') == 'lifecycle' and e.get('event') in ('start', 'restart')
                    and restart_receipt['seq'] < e['seq'] < rec['seq']
                ]
                isolated = bool(between)
            extra['cold_instance_isolated'] = isolated
            if not isolated:
                reasons.append(
                    'cold-ref targeted the same endpoint as restart with no intervening lifecycle restart'
                )
            # Verify /v1/imprint/resume probe before cold-ref confirms no RAM or disk state.
            probe_status = rec.get('cold_probe_status')
            if 'cold_probe' not in (rec.get('sidecars') or {}):
                unknown_evidence = True
            elif probe_status == 404:
                extra['cold_probe'] = {'status': 404, 'learn_disabled': True}
            elif probe_status == 200:
                try:
                    probe_body = campaign.parsed(rec, 'cold_probe', strict_json)
                except ValueError as err:
                    reasons.append(f'unreadable cold-ref imprint probe: {err}')
                else:
                    extra['cold_probe'] = {
                        'status': 200,
                        'resume_from': probe_body.get('resume_from'),
                        'ram_tokens': probe_body.get('ram_tokens'),
                        'disk_tokens': probe_body.get('disk_tokens'),
                    }
                    if (
                        probe_body.get('resume_from') != 'none'
                        or int(probe_body.get('ram_tokens') or 0) > 0
                        or int(probe_body.get('disk_tokens') or 0) > 0
                    ):
                        reasons.append(
                            f'cold-ref ran against a warm/cached instance: {extra["cold_probe"]}'
                        )
            else:
                reasons.append(f'cold-ref imprint probe returned unexpected status {probe_status!r}')
            if measured_count is not None:
                cold_reads = find_logs('read', c_start, c_end, 1, measured_count)
                extra['reads'] = [x['line'] for x in cold_reads]
                if cold_reads:
                    reasons.append('cold-ref logged a disk read instead of running as a cold prefill')
                elif not covered_fn(c_start, c_end):
                    unknown_evidence = True
        elif phase in ('swap-pixel', 'swap-geometry'):
            if 'pre-image2-prefix' not in prefix_counts or measured_count is None:
                reasons.append(f'missing prefix boundary or measurement for {phase}')
            else:
                boundary = prefix_counts['pre-image2-prefix']
                shared_tokens = None
                if initial_tokens is not None and tok_ids is not None:
                    shared_tokens = common_prefix_len(initial_tokens, tok_ids)
                    boundary = min(boundary, shared_tokens)
                c_start, c_end = when_fn(rec['started_at']), when_fn(rec['ended_at'])
                over_reads = find_logs('read', c_start, c_end, boundary + 1, measured_count)
                clean_reads = find_logs('read', c_start, c_end, 1, boundary)
                extra.update(
                    swap_prefix_boundary_tokens=boundary,
                    pre_image2_prefix_tokens=prefix_counts['pre-image2-prefix'],
                    rendered_common_prefix_tokens=shared_tokens,
                    clean_prefix_reads=[x['line'] for x in clean_reads],
                    over_prefix_reads=[x['line'] for x in over_reads],
                )
                if over_reads:
                    reasons.append(
                        f'prefix reuse crossed modified image 2 boundary ({boundary} tokens): '
                        'stale image 2 KV was reused'
                    )
                elif (c_start, c_end) in ambiguous_reads:
                    unknown_evidence = True
                elif not covered_fn(c_start, c_end):
                    unknown_evidence = True
                elif not clean_reads:
                    unknown_evidence = True

        status = 'fail' if reasons else ('unknown' if unknown_evidence else 'pass')
        entry = {
            'seq': rec['seq'],
            'measured_prompt_tokens': measured_count,
            'correct': correct,
            'status': status,
            'reasons': reasons,
            **extra,
        }
        _record_attempt(results, attempts, phase, entry)

        if phase == 'initial' and reply is not None:
            initial_history = [(body, reply['message'])]
            initial_receipt = rec
            initial_measured = measured_count
            initial_tokens = tok_ids
            initial_completion = reply['usage']['completion_tokens']

    continuation_match: bool | None = None
    if restart_replies and cold_ref_replies:
        # Check EVERY recorded restart/cold-ref pair so a later retry cannot erase an earlier divergence.
        continuation_match = all(
            r['message'] == c['message'] and r['finish_reason'] == c['finish_reason']
            for r in restart_replies
            for c in cold_ref_replies
        )
        if not continuation_match and 'cold-ref' in results:
            results['cold-ref']['reasons'].append('restart continuation diverged from cold-prefill reference')
            results['cold-ref']['status'] = 'fail'

    missing_phases = [ph for ph in DURABLE_IMAGE_PHASES if ph not in results]
    rank = {'fail': 3, 'unknown': 2, 'pass': 1}
    statuses = [v['status'] for v in results.values()] + (['unknown'] if missing_phases else [])
    return {
        'status': max(statuses, key=rank.__getitem__) if statuses else 'unknown',
        'prefix_tokens': prefix_counts,
        'restart_matches_cold_prefill': continuation_match,
        'missing_phases': missing_phases,
        'phases': results,
    }


def _eval_reasoning_matrix(
    campaign: Any,
    recs: list[dict[str, Any]],
    all_receipts: list[dict[str, Any]],
    measures: Mapping[tuple[str, str], dict[str, Any]],
    *,
    measure_history: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not recs:
        return {'status': 'not_run', 'efforts': {}}
    plan = campaign.plan
    history_map = measure_history or {}
    efforts_out: dict[str, dict[str, Any]] = {}
    step_attempts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    turn1_history: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    turn1_had_reasoning: dict[str, bool] = {}

    # Require ALL four shared-prompt /v1/tokenize renders bound by exact request_sha256; never let retries hide a prior collapse.
    render_hashes: dict[str, str] = {}
    render_hash_sets: dict[str, set[str]] = {}
    render_reasons: list[str] = []
    missing_renders: list[str] = []
    for eff in REASONING_EFFORTS:
        want_req = sha256(canonical(workload.tokenize_body(reasoning_render_body(plan, eff))))
        key = ('effort-render', want_req)
        all_m = history_map.get(key) or ([measures[key]] if key in measures else [])
        if not all_m:
            missing_renders.append(eff)
        else:
            render_hashes[eff] = all_m[-1]['tokens_sha256']
            seen = {m['tokens_sha256'] for m in all_m}
            render_hash_sets[eff] = seen
            if len(seen) > 1:
                render_reasons.append(f'effort {eff!r} produced inconsistent /tokenize renders across attempts: {sorted(seen)}')

    eff_list = list(render_hash_sets)
    for i, eff_a in enumerate(eff_list):
        for eff_b in eff_list[i + 1:]:
            overlap = render_hash_sets[eff_a] & render_hash_sets[eff_b]
            if overlap:
                render_reasons.append(
                    f'reasoning efforts {eff_a!r} and {eff_b!r} collapsed to identical /tokenize render {sorted(overlap)}'
                )

    clear_checks: dict[str, dict[str, Any]] = {}
    for rec in recs:
        effort = rec.get('effort', '')
        step = rec.get('step', '')
        body, reply, reasons = _reparse_gate_chat(campaign, rec)
        expected: dict[str, str] | None = None
        exp_body: dict[str, Any] | None = None
        try:
            hist = turn1_history.get(effort) if step == 'turn-2' else None
            exp_body, expected, _ = reasoning_matrix_turn(plan, effort, step, history=hist)
            if canonical(exp_body) != canonical(body):
                reasons.append('request does not match deterministic reasoning-matrix plan (exact reasoning replay required)')
        except (ValueError, KeyError) as err:
            reasons.append(f'request cannot be regenerated: {err}')

        correct = False
        has_reasoning = False
        if reply is not None and expected is not None:
            correct = workload.correct(expected, reply['message']['content'])
            has_reasoning = bool(reply['message'].get('reasoning_content'))
            if not correct:
                reasons.append('wrong or missing answer')
            if step == 'turn-1' and effort in ('max', 'high', 'low') and not has_reasoning:
                reasons.append(f'effort {effort!r} produced empty reasoning_content')
            if effort == 'none' and has_reasoning:
                reasons.append("effort 'none' emitted non-empty reasoning_content")

        if step == 'turn-2' and effort in ('max', 'high', 'low') and turn1_had_reasoning.get(effort) and exp_body is not None:
            # Bind CLEAR_THINKING=0 tokenize checks directly to the exact regenerated turn-2 requests.
            with_hash = sha256(canonical(workload.tokenize_body(exp_body)))
            t1_body, t1_msg = turn1_history[effort][-1]
            no_think_msg = dict(t1_msg)
            no_think_msg['reasoning_content'] = ''
            exp_without, _, _ = reasoning_matrix_turn(plan, effort, 'turn-2', history=[(t1_body, no_think_msg)])
            without_hash = sha256(canonical(workload.tokenize_body(exp_without)))
            m_with_all = history_map.get(('clear-thinking-with', with_hash)) or (
                [measures[('clear-thinking-with', with_hash)]] if ('clear-thinking-with', with_hash) in measures else []
            )
            m_without_all = history_map.get(('clear-thinking-without', without_hash)) or (
                [measures[('clear-thinking-without', without_hash)]]
                if ('clear-thinking-without', without_hash) in measures else []
            )
            m_with = m_with_all[-1] if m_with_all else None
            m_without = m_without_all[-1] if m_without_all else None
            clear_checks[effort] = {
                'clear-thinking-with': m_with and m_with['count'],
                'clear-thinking-without': m_without and m_without['count'],
            }
            if m_with is None or m_without is None:
                reasons.append('missing CLEAR_THINKING=0 tokenize pair bound to exact turn-2 requests')
            elif any(mw['count'] <= mwo['count'] for mw in m_with_all for mwo in m_without_all):
                reasons.append(
                    f'CLEAR_THINKING=0 check failed: prompt with reasoning ({min(x["count"] for x in m_with_all)}) '
                    f'<= without reasoning ({max(x["count"] for x in m_without_all)})'
                )

        eff_entry = efforts_out.setdefault(effort, {'steps': {}, 'status': 'pass'})
        eff_attempts = step_attempts.setdefault(effort, {})
        step_entry = {
            'seq': rec['seq'],
            'correct': correct,
            'has_reasoning': has_reasoning,
            'status': 'fail' if reasons else 'pass',
            'reasons': reasons,
        }
        _record_attempt(eff_entry['steps'], eff_attempts, step, step_entry)
        if any(s['status'] == 'fail' for s in eff_entry['steps'].values()):
            eff_entry['status'] = 'fail'
        if step == 'turn-1' and reply is not None:
            turn1_history[effort] = [(body, reply['message'])]
            turn1_had_reasoning[effort] = has_reasoning

    missing_efforts = [e for e in REASONING_EFFORTS if e not in efforts_out or set(efforts_out[e]['steps']) != {'turn-1', 'turn-2'}]
    rank = {'fail': 3, 'unknown': 2, 'pass': 1}
    statuses = [v['status'] for v in efforts_out.values()]
    if render_reasons:
        statuses.append('fail')
    if missing_efforts or missing_renders:
        statuses.append('unknown')
    return {
        'status': max(statuses, key=rank.__getitem__) if statuses else 'unknown',
        'distinct_renders': not render_reasons and not missing_renders and len(render_hashes) == len(REASONING_EFFORTS),
        'missing_renders': missing_renders,
        'render_hashes': render_hashes,
        'render_reasons': render_reasons,
        'clear_thinking_tokens': clear_checks,
        'missing_efforts': missing_efforts,
        'efforts': efforts_out,
    }
