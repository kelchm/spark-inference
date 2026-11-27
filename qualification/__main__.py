"""GLM-5.3 agent-1m qualification client.

Offline: prepare, record-lifecycle, import-log, verdict. Online, and only against an explicit --endpoint during an
approved window: server, turn, imprint. Nothing here starts, stops or restarts a server; home-lab does that by hand and
the operator records it with record-lifecycle.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from . import coding, gates, host_state, workload
from .evidence import EVENTS, SCHEMA, Campaign, harness_sha256, tokenize_result, verdict
from .transport import canonical, exchange, now, parse_sse, sha256, strict_json

LIMITS = {'chat': 16 << 20, 'tokenize': 64 << 20, 'small': 1 << 20}


def prepare(args):
    release_bytes, profile_bytes = Path(args.release).read_bytes(), Path(args.profile).read_bytes()
    release, profile = strict_json(release_bytes), profile_bytes.decode()
    container = re.search(r'^container:\s*"?([^"\s]+)"?\s*$', profile, re.M)
    learn_dir = re.search(r'^\s*GLM_LEARN_DIR:\s*"?([^"\s]+)"?\s*$', profile, re.M)
    name = re.search(r'^\s*profile:\s*"?([^"\s]+)"?\s*$', profile, re.M)
    if not container or container[1] != release['image']:
        raise SystemExit('the profile does not pin the release image')
    if not learn_dir or not name:
        raise SystemExit('the profile names no GLM_LEARN_DIR or metadata profile')
    p = workload.plan(args.seed, args.model, args.reasoning_effort, args.max_tokens, margin=args.margin,
                      near_floor=args.near_floor, evictor_target=args.evictor_target)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    for sub in ('receipts', 'artifacts', 'prepared'):
        (root / sub).mkdir()
    plan_bytes = json.dumps(p, indent=1, sort_keys=True).encode() + b'\n'
    (root / 'plan.json').write_bytes(plan_bytes)
    prepared = {}
    for n, colors in enumerate(p['images'], 1):
        data = workload.png(colors)
        (root / 'prepared' / f'image-{n}.png').write_bytes(data)
        prepared[f'image-{n}.png'] = sha256(data)
    for session, step in (('image', 'first'), ('pi', 'call')):
        body = canonical(workload.next_turn(p, session, step, [])[0])
        (root / 'prepared' / f'{session}-{step}.json').write_bytes(body)
        prepared[f'{session}-{step}.json'] = sha256(body)
    manifest = {'schema': SCHEMA, 'created_at': now(), 'plan_sha256': sha256(plan_bytes),
                'harness_sha256': harness_sha256(), 'prepared': prepared, 'ranks': 2,
                'release': {'file': Path(args.release).name, 'sha256': sha256(release_bytes),
                            'release': release['release'], 'image': release['image'],
                            'engine_commit': release['engine']['commit']},
                'profile': {'file': Path(args.profile).name, 'sha256': sha256(profile_bytes), 'name': name[1],
                            'learn_dir': learn_dir[1]}}
    (root / 'manifest.json').write_bytes(json.dumps(manifest, indent=1, sort_keys=True).encode() + b'\n')
    print(f'prepared {root} (plan {manifest["plan_sha256"][:12]}); no endpoint was contacted')


def key(args):
    return os.environ.get(args.api_key_env) if args.api_key_env else None


def server(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    for path in ('/health', '/v1/models'):
        result = exchange(args.endpoint, 'GET', path, None, args.deadline, max_bytes=LIMITS['small'], api_key=key(args))
        raw = result.pop('raw')
        c.write('server', path.strip('/').replace('/', '-'), {'endpoint': args.endpoint, 'path': path, **result},
                {'response': raw})
        print(f'{path}: HTTP {result["http_status"]} {"ok" if result["complete"] else result["error"]}')


def tokenize(c, args, body, purpose, session, step, add_generation_prompt=True, extra=None):
    request = canonical(workload.tokenize_body(body, add_generation_prompt))
    result = exchange(args.endpoint, 'POST', '/v1/tokenize', request, args.tokenize_deadline,
                      max_bytes=LIMITS['tokenize'], api_key=key(args))
    raw = result.pop('raw')
    measured = None
    if result['complete']:
        try:
            measured = tokenize_result(raw)
        except ValueError as error:
            result.update(complete=False, error=f'ValueError: {error}')
    sidecars = {'request': request, 'response': raw} if purpose != 'fit' else {}  # fit probes keep hashes only
    c.write('tokenize', f'{session}-{step}-{purpose}',
            {'endpoint': args.endpoint, 'purpose': purpose, 'session': session, 'step': step,
             'request_sha256': sha256(request), 'chat_request_sha256': sha256(canonical(body)),
             **(extra or {}), **result, **(measured or {})}, sidecars)
    if measured is None:
        raise SystemExit(f'/tokenize failed ({result["error"]}); recorded, not retried')
    return measured['count']


def next_step(c, session, receipts, retry, continuation):
    attempts = [r for r in receipts if r['kind'] == 'chat' and r['session'] == session]
    done = c.history(session, receipts=receipts)
    done_steps = [r['step'] for _, _, r in done]
    if attempts and attempts[-1]['seq'] != (done[-1][2]['seq'] if done else None):
        if not retry:
            raise SystemExit(f'the last {session} attempt ({attempts[-1]["step"]}) did not complete; '
                             'pass --retry to run it again (the failure stays on record)')
        return attempts[-1]['step'], done
    base = [s for s in workload.BASE_STEPS[session] if s not in done_steps]
    if continuation:
        if session not in workload.CONTINUABLE or base:
            raise SystemExit(f'{session} cannot continue until its base steps are done: {base}')
        return f'continue-{sum(s.startswith("continue-") for s in done_steps) + 1}', done
    if not base:
        raise SystemExit(f'{session} has run every base step; use --continue')
    return base[0], done


def turn(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    p = c.plan
    session = args.session
    step, done = next_step(c, session, c.receipts(), args.retry, args.continuation)
    history = [h[:2] for h in done]
    chunks, limit = None, p['ceiling'] - p['margin']
    if session in ('ledger', 'evictor') and not step.startswith('continue-'):
        index = workload.BASE_STEPS[session].index(step)
        if session == 'evictor':
            target, reach = p['evictor_target'], True
        elif index < len(p['ledger_targets']):
            target, reach = p['ledger_targets'][index], True
        else:
            target, reach = p['near_floor'], False

        def count(n):
            return tokenize(c, args, workload.next_turn(p, session, step, history, n)[0], 'fit', session, step)

        chunks, fitted, probes = workload.fit(count, target, limit, reach=reach)
        if fitted < target:
            raise SystemExit(f'filling to {limit} tokens reaches only {fitted}, under the {target}-token floor')
        print(f'{session}/{step}: {chunks} chunks fit {fitted} tokens in {len(probes)} /tokenize calls')
    body, expected, meta = workload.next_turn(p, session, step, history, chunks)
    measured = tokenize(c, args, body, 'measure', session, step)
    if measured > (p['ceiling'] if step.startswith('continue-') else limit):
        raise SystemExit(f'{measured} prompt tokens exceed the limit; nothing was sent')
    if session in ('image', 'image-fork') and step == 'first':
        prefix = workload.chat_body(p, session, [{'role': 'system', 'content': workload.SYSTEM[session]},
                 {'role': 'user', 'content': workload.image_parts(workload.image_plan(p, session))}])
        tokenize(c, args, prefix, 'image-prefix', session, step, add_generation_prompt=False)
    request = canonical(body)
    result = exchange(args.endpoint, 'POST', '/v1/chat/completions', request, args.deadline, stream=True,
                      max_bytes=LIMITS['chat'], api_key=key(args))
    raw = result.pop('raw')
    observed = {'correct': None}
    try:
        reply = parse_sse(raw)
        if session == 'pi' and step == 'call':
            results = workload.tool_results(p, reply['message'])
            correct = reply['finish_reason'] == 'tool_calls' and bool(results) and all(
                x['content'] == p['pi']['value'] for x in results)
        else:
            correct = workload.correct(expected, reply['message']['content'])
        observed = {'finish_reason': reply['finish_reason'], 'usage': reply['usage'],
                    'correct': correct}
    except ValueError as error:
        observed['parse_error'] = str(error)
    record = c.write('chat', f'{session}-{step}',
                     {'endpoint': args.endpoint, 'session': session, 'step': step, 'meta': meta,
                      'request_sha256': sha256(request), 'measured_prompt_tokens': measured, **result,
                      'client_observed': observed}, {'request': request, 'response': raw})
    state = 'complete' if result['complete'] else f'FAILED: {result["error"]}'
    print(f'{session}/{step}: receipt {record["seq"]}, {measured} prompt tokens, {state}, '
          f'answer {"correct" if observed["correct"] else "NOT correct"} ({result["elapsed_s"]:.1f}s)')
    if not result['complete'] or observed['correct'] is not True:
        sys.exit(1)


def imprint(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    receipts = c.receipts()
    step, done = next_step(c, args.session, receipts, False, True)
    body = workload.next_turn(c.plan, args.session, step, [h[:2] for h in done])[0]
    request = canonical(body)
    result = exchange(args.endpoint, 'POST', f'/v1/imprint/{args.op}', request, args.deadline,
                      max_bytes=LIMITS['small'], api_key=key(args))
    raw = result.pop('raw')
    parsed = strict_json(raw) if result['complete'] else None
    c.write('imprint', f'{args.session}-{args.op}',
            {'endpoint': args.endpoint, 'session': args.session, 'op': args.op, 'for_step': step,
             'request_sha256': sha256(request), **result}, {'request': request, 'response': raw})
    print(json.dumps(parsed) if parsed is not None else f'imprint {args.op} failed: {result["error"]}')


def copy_artifact(c, path):
    data = Path(path).read_bytes()
    seq = 1 + max((int(p.name.split('-', 1)[0]) for p in (c.root / 'receipts').glob('[0-9]*')), default=0)
    name = f'{seq:04d}-{Path(path).name}'
    with (c.root / 'artifacts' / name).open('xb') as out:
        out.write(data)
    return {'file': name, 'sha256': sha256(data), 'bytes': len(data)}


def instant(text):
    if text is None:
        return now()
    if datetime.fromisoformat(text).tzinfo is None:
        raise SystemExit(f'{text} needs a UTC offset')
    return text


def record_lifecycle(args):
    c = Campaign(args.campaign)
    identity = [copy_artifact(c, path) for path in args.identity]
    record = c.write('lifecycle', args.event, {'event': args.event, 'at': instant(args.at), 'note': args.note,
                                               'identity': identity})
    print(f'lifecycle {args.event} recorded as receipt {record["seq"]}')


def import_log(args):
    c = Campaign(args.campaign)
    artifact = copy_artifact(c, args.file)
    record = c.write('log', args.label, {'label': args.label, 'captured_at': instant(args.captured_at),
                                         'artifact': artifact})
    print(f'log {args.label} recorded as receipt {record["seq"]} ({artifact["sha256"][:12]})')


def record_host_state(args):
    c = Campaign(args.campaign)
    info, sidecars = host_state.capture_host_state(
        args.boot_profile,
        boot_id=args.boot_id,
        boot_id_path=args.boot_id_path,
        meminfo_path=args.meminfo,
        zoneinfo_path=args.zoneinfo,
        rank=args.rank,
        host=args.host,
        run_id=args.run_id,
        lifecycle_seq=args.lifecycle_seq,
        note=args.note,
        captured_at=instant(args.at),
    )
    record = c.write('host-state', f'{args.boot_profile}-{args.rank}', info, sidecars)
    print(f'host-state {args.boot_profile} ({args.rank}) recorded as receipt {record["seq"]} '
          f'(boot_id={info["boot_id"][:12]}, MemAvailable={info["mem_available_gib"]:.2f} GiB)')


def capacity(args):
    c = Campaign(args.campaign)
    models_raw = None
    if args.endpoint:
        c.require_current_harness()
        result = exchange(args.endpoint, 'GET', '/health', None, args.deadline,
                          max_bytes=LIMITS['small'], api_key=key(args))
        health_raw = result.pop('raw')
        if not result['complete']:
            c.write('capacity', 'failed', {'endpoint': args.endpoint, **result}, {'health': health_raw})
            raise SystemExit(f'/health failed ({result["error"]})')
        models_res = exchange(args.endpoint, 'GET', '/v1/models', None, args.deadline,
                              max_bytes=LIMITS['small'], api_key=key(args))
        models_raw = models_res.pop('raw') if models_res['complete'] else None
        result['models_http_status'] = models_res.get('http_status')
        result['models_error'] = models_res.get('error')
    elif args.health_file:
        health_raw = Path(args.health_file).read_bytes()
        result = {'endpoint': None, 'complete': True, 'http_status': 200}
        if args.models_file:
            models_raw = Path(args.models_file).read_bytes()
    else:
        raise SystemExit('capacity requires --endpoint or --health-file')

    sidecars = {'health': health_raw}
    models_body = None
    if models_raw is not None:
        sidecars['models'] = models_raw
        models_body = strict_json(models_raw)
    log_text = None
    if args.log:
        log_bytes = Path(args.log).read_bytes()
        sidecars['log'] = log_bytes
        log_text = log_bytes.decode('utf-8', 'replace')
    health_body = strict_json(health_raw)
    checked = host_state.evaluate_capacity(
        health_body,
        log_text,
        models_body,
        min_pool_tokens=args.min_pool_tokens,
        expected_streams=args.streams,
        expected_context=c.plan['context'],
        expected_model=c.plan['model'],
        learn_gib=args.learn_gib,
        learn_dir=c.manifest['profile']['learn_dir'] if args.check_learn_store else None,
        check_learn_store=bool(args.check_learn_store),
        ranks=c.manifest['ranks'],
    )
    record = c.write(
        'capacity',
        'check',
        {
            **result,
            'min_pool_tokens': args.min_pool_tokens,
            'expected_streams': args.streams,
            'expected_context': c.plan['context'],
            'learn_gib': args.learn_gib,
            'check_learn_store': bool(args.check_learn_store),
            'client_observed': checked,
        },
        sidecars,
    )
    print(f'capacity receipt {record["seq"]}: {checked["status"]} '
          f'(pool_tokens={checked["pool_tokens"]}, floor={args.min_pool_tokens})')
    if checked['status'] != 'pass':
        sys.exit(1)


def _gate_history(c, gate_name, cases):
    turns = []
    for r in c.receipts():
        if r.get('kind') != 'gate' or r.get('gate') != gate_name:
            continue
        step_id = r.get('case') or r.get('phase') or r.get('step')
        if step_id not in cases:
            continue
        try:
            msg = c.parsed(r, 'response', parse_sse)['message']
        except (ValueError, KeyError):
            continue
        turns.append((c.parsed(r, 'request', strict_json), msg, r))
    return turns


def dense_sparse(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    p = c.plan
    case = args.case
    history = []
    pad_units = 0
    if case in ('dense-2051', 'sparse-2052'):
        target = gates.DENSE_MAX_VISIBLE if case == 'dense-2051' else gates.SPARSE_MIN_VISIBLE

        def count_fn(units):
            b, _, _ = gates.dense_sparse_turn(p, case, pad_units=units)
            return tokenize(c, args, b, 'fit', 'dense-sparse', case)

        pad_units, fitted, _ = gates.fit_exact_tokens(count_fn, target)
        if fitted != target:
            raise SystemExit(f'{case}: exact fit reached {fitted} tokens, not {target}')
    else:
        prior_cases = ('sparse-2052',) if case == 'page-reuse' else ('sparse-2052', 'page-reuse')
        done = _gate_history(c, 'dense-sparse', prior_cases)
        if not done:
            raise SystemExit(f'dense-sparse/{case} requires a completed sparse-2052 receipt first')
        history = [h[:2] for h in done]
        pad_units = args.pad_units

    body, expected, meta = gates.dense_sparse_turn(p, case, pad_units=pad_units, history=history)
    measured = tokenize(c, args, body, 'measure', 'dense-sparse', case)
    request = canonical(body)
    result = exchange(args.endpoint, 'POST', '/v1/chat/completions', request, args.deadline, stream=True,
                      max_bytes=LIMITS['chat'], api_key=key(args))
    raw = result.pop('raw')
    observed = {'correct': None}
    try:
        reply = parse_sse(raw)
        observed = {'finish_reason': reply['finish_reason'], 'usage': reply['usage'],
                    'correct': workload.correct(expected, reply['message']['content'])}
    except ValueError as err:
        observed['parse_error'] = str(err)
    record = c.write(
        'gate',
        f'dense-sparse-{case}',
        {'endpoint': args.endpoint, 'gate': 'dense-sparse', 'case': case, 'meta': meta,
         'request_sha256': sha256(request), 'measured_prompt_tokens': measured, **result,
         'client_observed': observed},
        {'request': request, 'response': raw},
    )
    print(f'dense-sparse/{case}: receipt {record["seq"]}, {measured} prompt tokens, '
          f'answer {"correct" if observed["correct"] else "NOT correct"}')
    if not result['complete'] or observed['correct'] is not True:
        sys.exit(1)


def image_durable(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    p = c.plan
    phase = args.phase
    prior_initial = _gate_history(c, 'image-durable', ('initial',))
    mid_pad = args.mid_pad_units
    tail_pad = args.tail_pad_units
    history = []

    if phase == 'initial':
        if args.target_tokens <= gates.SPARSE_MIN_VISIBLE:
            raise SystemExit(f'--target-tokens must exceed sparse boundary ({gates.SPARSE_MIN_VISIBLE})')

        def count_fn(units):
            b, _, _ = gates.durable_image_turn(p, 'initial', mid_pad_units=mid_pad, tail_pad_units=units)
            return tokenize(c, args, b, 'fit', 'image-durable', 'initial')

        tail_pad, fitted, _ = gates.fit_exact_tokens(count_fn, args.target_tokens)
        if fitted <= gates.SPARSE_MIN_VISIBLE:
            raise SystemExit(f'image-durable/initial reached only {fitted} tokens')
    else:
        if prior_initial:
            init_meta = prior_initial[-1][2].get('meta') or {}
            mid_pad = init_meta.get('mid_pad_units', mid_pad)
            tail_pad = init_meta.get('tail_pad_units', tail_pad)
            if phase in ('restart', 'cold-ref'):
                history = [prior_initial[-1][:2]]
        elif phase in ('restart', 'cold-ref'):
            raise SystemExit(f'image-durable/{phase} requires a completed initial receipt first')

    body, expected, meta = gates.durable_image_turn(
        p, phase, mid_pad_units=mid_pad, tail_pad_units=tail_pad, history=history
    )
    measured = tokenize(c, args, body, 'measure', 'image-durable', phase)
    if phase == 'initial':
        for pname, pbody in gates.durable_image_prefix_bodies(p, mid_pad_units=mid_pad, variant='original').items():
            tokenize(c, args, pbody, pname, 'image-durable', phase, add_generation_prompt=False)
    request = canonical(body)
    extra_fields = {}
    sidecars = {'request': request}
    if phase == 'cold-ref':
        # Probe /v1/imprint/resume before cold-ref so the offline verdict can verify the instance has no cached state.
        cold_probe = exchange(
            args.endpoint, 'POST', '/v1/imprint/resume', request, args.tokenize_deadline,
            max_bytes=LIMITS['small'], api_key=key(args),
        )
        cold_probe_raw = cold_probe.pop('raw')
        extra_fields['cold_probe_status'] = cold_probe.get('http_status')
        extra_fields['cold_probe_error'] = cold_probe.get('error')
        sidecars['cold_probe'] = cold_probe_raw
    result = exchange(args.endpoint, 'POST', '/v1/chat/completions', request, args.deadline, stream=True,
                      max_bytes=LIMITS['chat'], api_key=key(args))
    raw = result.pop('raw')
    sidecars['response'] = raw
    observed = {'correct': None}
    try:
        reply = parse_sse(raw)
        observed = {'finish_reason': reply['finish_reason'], 'usage': reply['usage'],
                    'correct': workload.correct(expected, reply['message']['content'])}
    except ValueError as err:
        observed['parse_error'] = str(err)
    record = c.write(
        'gate',
        f'image-durable-{phase}',
        {'endpoint': args.endpoint, 'gate': 'image-durable', 'phase': phase, 'meta': meta,
         'request_sha256': sha256(request), 'measured_prompt_tokens': measured, **extra_fields, **result,
         'client_observed': observed},
        sidecars,
    )
    print(f'image-durable/{phase}: receipt {record["seq"]}, {measured} prompt tokens, '
          f'answer {"correct" if observed["correct"] else "NOT correct"}')
    if not result['complete'] or observed['correct'] is not True:
        sys.exit(1)


def reasoning_matrix(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    p = c.plan
    efforts = tuple(e.strip() for e in args.efforts.split(',') if e.strip())
    failed = False
    for effort in efforts:
        if effort not in gates.REASONING_EFFORTS:
            raise SystemExit(f'unknown effort {effort!r}; expected subset of {gates.REASONING_EFFORTS}')
        # 1. Distinct prompt render check via /v1/tokenize on IDENTICAL messages across all efforts
        render_body = gates.reasoning_render_body(p, effort)
        tokenize(c, args, render_body, 'effort-render', 'reasoning-matrix', effort, extra={'effort': effort})

        # 2. Turn 1
        t1_body, t1_expected, t1_meta = gates.reasoning_matrix_turn(p, effort, 'turn-1')
        t1_req = canonical(t1_body)
        t1_res = exchange(args.endpoint, 'POST', '/v1/chat/completions', t1_req, args.deadline, stream=True,
                          max_bytes=LIMITS['chat'], api_key=key(args))
        t1_raw = t1_res.pop('raw')
        t1_reply = None
        t1_obs = {'correct': None}
        try:
            t1_reply = parse_sse(t1_raw)
            t1_obs = {'finish_reason': t1_reply['finish_reason'], 'usage': t1_reply['usage'],
                      'has_reasoning': bool(t1_reply['message'].get('reasoning_content')),
                      'correct': workload.correct(t1_expected, t1_reply['message']['content'])}
        except ValueError as err:
            t1_obs['parse_error'] = str(err)
        c.write(
            'gate',
            f'reasoning-{effort}-turn-1',
            {'endpoint': args.endpoint, 'gate': 'reasoning-matrix', 'effort': effort, 'step': 'turn-1',
             'meta': t1_meta, 'request_sha256': sha256(t1_req), **t1_res, 'client_observed': t1_obs},
            {'request': t1_req, 'response': t1_raw},
        )
        if not t1_res['complete'] or t1_obs['correct'] is not True or t1_reply is None:
            failed = True
            continue

        # 3. Turn 2 replaying reasoning_content verbatim (CLEAR_THINKING=0)
        t2_body, t2_expected, t2_meta = gates.reasoning_matrix_turn(
            p, effort, 'turn-2', history=[(t1_body, t1_reply['message'])]
        )
        tokenize(c, args, t2_body, 'clear-thinking-with', 'reasoning-matrix', effort, extra={'effort': effort})
        no_think_msg = dict(t1_reply['message'])
        no_think_msg['reasoning_content'] = ''
        t2_without, _, _ = gates.reasoning_matrix_turn(p, effort, 'turn-2', history=[(t1_body, no_think_msg)])
        tokenize(c, args, t2_without, 'clear-thinking-without', 'reasoning-matrix', effort, extra={'effort': effort})

        t2_req = canonical(t2_body)
        t2_res = exchange(args.endpoint, 'POST', '/v1/chat/completions', t2_req, args.deadline, stream=True,
                          max_bytes=LIMITS['chat'], api_key=key(args))
        t2_raw = t2_res.pop('raw')
        t2_obs = {'correct': None}
        try:
            t2_reply = parse_sse(t2_raw)
            t2_obs = {'finish_reason': t2_reply['finish_reason'], 'usage': t2_reply['usage'],
                      'correct': workload.correct(t2_expected, t2_reply['message']['content'])}
        except ValueError as err:
            t2_obs['parse_error'] = str(err)
        c.write(
            'gate',
            f'reasoning-{effort}-turn-2',
            {'endpoint': args.endpoint, 'gate': 'reasoning-matrix', 'effort': effort, 'step': 'turn-2',
             'meta': t2_meta, 'request_sha256': sha256(t2_req), **t2_res, 'client_observed': t2_obs},
            {'request': t2_req, 'response': t2_raw},
        )
        if not t2_res['complete'] or t2_obs['correct'] is not True:
            failed = True
        print(f'reasoning-matrix/{effort}: turn-1={t1_obs["correct"]}, turn-2={t2_obs["correct"]}')
    if failed:
        sys.exit(1)


def coding_replay(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    rec = coding.run_coding_replay(
        c,
        endpoint=args.endpoint,
        shape=args.shape,
        turns=args.turns,
        target_tokens=args.target_tokens,
        deadline=args.deadline,
        tokenize_deadline=args.tokenize_deadline,
        starvation_ms=args.starvation_ms,
        api_key=key(args),
        tokenize_fn=lambda body, purpose, sess, step: tokenize(c, args, body, purpose, sess, step),
        chat_limit=LIMITS['chat'],
    )
    print(f'coding-replay/{args.shape}: receipt {rec["seq"]} recorded')


def concurrency_cmd(args):
    c = Campaign(args.campaign)
    c.require_current_harness()
    levels = tuple(int(x.strip()) for x in args.levels.split(',') if x.strip())
    recs = coding.run_concurrency(
        c,
        endpoint=args.endpoint,
        levels=levels,
        chunks_per_stream=args.chunks,
        deadline=args.deadline,
        starvation_ms=args.starvation_ms,
        api_key=key(args),
        chat_limit=LIMITS['chat'],
    )
    print(f'concurrency levels {list(levels)} recorded ({len(recs)} receipts)')


def run_verdict(args):
    result = verdict(args.campaign, args.clock_skew, args.tolerance, require_boot_pairing=args.require_boot_pairing)
    text = json.dumps(result, indent=1, sort_keys=True, default=str) + '\n'
    if args.output:
        with open(args.output, 'x') as out:
            out.write(text)
    sys.stdout.write(text if not args.output else f'qualification {result["qualification"]}, workload '
                     f'{result.get("workload", {}).get("status")}, production {result["production"]}\n')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python3 -m qualification', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    def online(command, help_text, deadline):
        s = sub.add_parser(command, help=help_text)
        s.add_argument('--campaign', required=True)
        s.add_argument('--endpoint', required=True, help='explicit base URL, such as http://host:8888')
        s.add_argument('--deadline', type=float, default=deadline, help='total seconds, headers and trickle included')
        s.add_argument('--api-key-env', help='environment variable holding a bearer token (never recorded)')
        return s

    s = sub.add_parser('prepare', help='write a campaign: plan, manifest and prepared requests (offline)')
    s.add_argument('--out', required=True)
    s.add_argument('--release', required=True)
    s.add_argument('--profile', required=True)
    s.add_argument('--seed', default='glm53-mia-2026.11.07')
    s.add_argument('--model', default='GLM-5.3-Flash')
    s.add_argument('--reasoning-effort', default='low')
    s.add_argument('--max-tokens', type=int, default=8192)
    s.add_argument('--margin', type=int, default=24_576)
    s.add_argument('--near-floor', type=int, default=980_000)
    s.add_argument('--evictor-target', type=int, default=600_000)
    s.set_defaults(func=prepare)

    online('server', 'record /health and /v1/models', 30).set_defaults(func=server)
    s = online('turn', "send a session's next turn, fitted and measured with /tokenize", 3600)
    s.add_argument('--session', required=True, choices=workload.SESSIONS)
    s.add_argument('--continue', dest='continuation', action='store_true', help='a continuation question')
    s.add_argument('--retry', action='store_true', help='run a failed step again; the failure stays on record')
    s.add_argument('--tokenize-deadline', type=float, default=900)
    s.set_defaults(func=turn)
    s = online('imprint', "ask where the session's next continuation would resume (resume) or load it (load)", 900)
    s.add_argument('--session', required=True, choices=workload.CONTINUABLE)
    s.add_argument('--op', choices=('resume', 'load'), default='resume')
    s.set_defaults(func=imprint)

    s = sub.add_parser('record-lifecycle', help='record a manual server event and its identity artifacts (offline)')
    s.add_argument('--campaign', required=True)
    s.add_argument('--event', required=True, choices=EVENTS)
    s.add_argument('--at', help='ISO-8601 time of the event (default: now)')
    s.add_argument('--identity', action='append', default=[], help='e.g. docker inspect output of the running rank 0')
    s.add_argument('--note', default='')
    s.set_defaults(func=record_lifecycle)
    s = sub.add_parser('import-log', help='retain a server log captured with timestamps (offline)')
    s.add_argument('--campaign', required=True)
    s.add_argument('--file', required=True)
    s.add_argument('--label', default='rank0')
    s.add_argument('--captured-at', help='ISO-8601 time the log was captured (default: now)')
    s.set_defaults(func=import_log)
    s = sub.add_parser('record-host-state',
                       help='record host boot_id, MemAvailable and fresh-boot/ordinary-relaunch tag (offline)')
    s.add_argument('--campaign', required=True)
    s.add_argument('--boot-profile', required=True, choices=host_state.BOOT_PROFILES)
    s.add_argument('--boot-id', default=None, help='explicit kernel boot_id UUID (reads boot-id-path if omitted)')
    s.add_argument('--boot-id-path', default='/proc/sys/kernel/random/boot_id')
    s.add_argument('--meminfo', default='/proc/meminfo')
    s.add_argument('--zoneinfo', default='/proc/zoneinfo')
    s.add_argument('--rank', default='rank0')
    s.add_argument('--host', default='')
    s.add_argument('--run-id', default='')
    s.add_argument('--lifecycle-seq', type=int, default=None)
    s.add_argument('--at', default=None, help='ISO-8601 capture time (default: now)')
    s.add_argument('--note', default='')
    s.set_defaults(func=record_host_state)

    s = sub.add_parser('capacity', help='verify 2,582,528-token 4x1M pool floor from /health, /v1/models and optional log')
    s.add_argument('--campaign', required=True)
    s.add_argument('--endpoint', default=None, help='explicit base URL (omit when using --health-file offline)')
    s.add_argument('--health-file', default=None, help='offline /health JSON file')
    s.add_argument('--models-file', default=None, help='offline /v1/models JSON file')
    s.add_argument('--log', default=None, help='optional startup log file with KV pool / learn-store lines')
    s.add_argument('--min-pool-tokens', type=int, default=host_state.POOL_FLOOR_4X1M)
    s.add_argument('--streams', type=int, default=host_state.DEFAULT_STREAMS)
    s.add_argument('--learn-gib', type=int, default=None)
    s.add_argument('--check-learn-store', action='store_true')
    s.add_argument('--deadline', type=float, default=30.0)
    s.add_argument('--api-key-env', default=None)
    s.set_defaults(func=capacity)

    s = online('dense-sparse', 'run 2051/2052 dense-to-sparse, page-reuse, and restart-restore gate', 900)
    s.add_argument('--case', required=True, choices=gates.DENSE_SPARSE_CASES)
    s.add_argument('--pad-units', type=int, default=32)
    s.add_argument('--tokenize-deadline', type=float, default=300)
    s.set_defaults(func=dense_sparse)

    s = online('image-durable', 'run #1285 durable two-image sparse/restart/cut/swap gate', 1800)
    s.add_argument('--phase', required=True, choices=gates.DURABLE_IMAGE_PHASES)
    s.add_argument('--target-tokens', type=int, default=4096)
    s.add_argument('--mid-pad-units', type=int, default=60)
    s.add_argument('--tail-pad-units', type=int, default=0)
    s.add_argument('--tokenize-deadline', type=float, default=300)
    s.set_defaults(func=image_durable)

    s = online('reasoning-matrix', 'run CLEAR_THINKING=0 + max/high/low/none reasoning effort matrix', 1800)
    s.add_argument('--efforts', default=','.join(gates.REASONING_EFFORTS))
    s.add_argument('--tokenize-deadline', type=float, default=300)
    s.set_defaults(func=reasoning_matrix)

    s = online('coding-replay', 'run 40k / 262k warm coding replay or cold-prefill + concurrent decoder', 3600)
    s.add_argument('--shape', required=True, choices=sorted(coding.CODING_SHAPES))
    s.add_argument('--turns', type=int, default=3)
    s.add_argument('--target-tokens', type=int, default=None)
    s.add_argument('--starvation-ms', type=float, default=coding.DEFAULT_STARVATION_MS)
    s.add_argument('--tokenize-deadline', type=float, default=900)
    s.set_defaults(func=coding_replay)

    s = online('concurrency', 'run 1/2/3/4 concurrent streams with Pi C1/C2-C3/C4 starvation reporting', 3600)
    s.add_argument('--levels', default='1,2,3,4')
    s.add_argument('--chunks', type=int, default=2)
    s.add_argument('--starvation-ms', type=float, default=coding.DEFAULT_STARVATION_MS)
    s.set_defaults(func=concurrency_cmd)

    s = sub.add_parser('verdict', help='judge the campaign from its receipts and logs (offline)')
    s.add_argument('--campaign', required=True)
    s.add_argument('--tolerance', type=int, default=4096, help='tokens a disk read may fall short of the prior prompt')
    s.add_argument('--clock-skew', type=float, default=2.0, help='seconds between client and log clocks')
    s.add_argument('--require-boot-pairing', action='store_true',
                   help='require paired fresh-boot and ordinary-relaunch host-state receipts')
    s.add_argument('--output', help='write the full verdict here (created exclusively)')
    s.set_defaults(func=run_verdict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
