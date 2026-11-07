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

from . import workload
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


def tokenize(c, args, body, purpose, session, step, add_generation_prompt=True):
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
    sidecars = {'response': raw} if purpose != 'fit' else {}  # fit probes keep hashes only
    c.write('tokenize', f'{session}-{step}-{purpose}',
            {'endpoint': args.endpoint, 'purpose': purpose, 'session': session, 'step': step,
             'request_sha256': sha256(request), 'chat_request_sha256': sha256(canonical(body)),
             **result, **(measured or {})}, sidecars)
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


def run_verdict(args):
    result = verdict(args.campaign, args.clock_skew, args.tolerance)
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
    s = sub.add_parser('verdict', help='judge the campaign from its receipts and logs (offline)')
    s.add_argument('--campaign', required=True)
    s.add_argument('--tolerance', type=int, default=4096, help='tokens a disk read may fall short of the prior prompt')
    s.add_argument('--clock-skew', type=float, default=2.0, help='seconds between client and log clocks')
    s.add_argument('--output', help='write the full verdict here (created exclusively)')
    s.set_defaults(func=run_verdict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
