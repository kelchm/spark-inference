"""Campaign receipts and the offline verdict.

The verdict trusts nothing a receipt says about its own outcome. It re-parses every reply from its recorded wire bytes,
regenerates every request from the plan and the earlier replies (so reasoning and tool call IDs must have been replayed
exactly), re-reads token counts from recorded /tokenize replies, and takes durability only from retained server logs:
the native `imprint: N tokens on disk` save and `imprint: read N tokens from disk` read lines. A correct answer after a
restart proves nothing about the disk store on its own: the server may simply have prefilled the prompt again.
"""
import base64
import gzip
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import workload
from .transport import canonical, parse_sse, sha256, strict_json

SCHEMA = 'glm53-qualification/1'
REQUIRED_PROBES = (('ledger', 'restart'), ('ledger', 'crash'), ('ledger', 'eviction'), ('image', 'restart'),
                   ('pi', 'restart'))
EVENTS = ('start', 'restart', 'crash', 'eviction')
TIMESTAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)\s')
SAVE = re.compile(r'imprint: (\d+) tokens on disk: (\d+) new blocks')
READ = re.compile(r'imprint: read (\d+) tokens from disk \((\d+) MiB in (\d+) ms')
SKIP = re.compile(r'imprint: (?:saving (\d+) tokens skipped|writing (\d+) tokens failed|(\d+) tokens stay in RAM only)'
                  r' \(([^)]*)\)')
CHECK = re.compile(r'imprint: a disk (?:block|state) failed its check')
STORE = re.compile(r'glm: conversation states on disk in (.+?) '
                   r"\((\d+) states, (\d+) MiB of (\d+) MiB; (\d+) ranks'? parts\)")
RANK = {'invalid': 4, 'fail': 3, 'unknown': 2, 'pass': 1}


def worst(*statuses):
    return max(statuses, key=RANK.__getitem__) if statuses else 'unknown'


def when(text):
    return datetime.fromisoformat(text)


def harness_sha256():
    return sha256(canonical({p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}))


class Campaign:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest_bytes = (self.root / 'manifest.json').read_bytes()
        self.manifest = strict_json(self.manifest_bytes)
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError(f'{root} is not a {SCHEMA} campaign')
        self.plan_bytes = (self.root / 'plan.json').read_bytes()
        self.plan = strict_json(self.plan_bytes)
        self.memo = {}  # verified sidecars and their parses: a 1M-token session re-reads multi-MB bodies otherwise

    @property
    def manifest_sha256(self):
        return sha256(self.manifest_bytes)

    def receipts(self):
        out = []
        for path in sorted((self.root / 'receipts').glob('[0-9]*.json')):
            record = strict_json(path.read_bytes())
            if path.name.split('-', 1)[0] != f'{record.get("seq"):04d}':
                raise ValueError(f'{path.name}: seq does not match its file name')
            out.append(record)
        return out

    def write(self, kind, label, record, sidecars=None):
        """A new receipt, created exclusively; sidecars are gzipped wire bytes hashed before compression."""
        directory = self.root / 'receipts'
        seq = 1 + max((int(p.name.split('-', 1)[0]) for p in directory.glob('[0-9]*')), default=0)
        stem = f'{seq:04d}-{kind}' + (f'-{label}' if label else '')
        files = {}
        for name, data in (sidecars or {}).items():
            path = directory / f'{stem}.{name}.gz'
            with path.open('xb') as out:
                out.write(gzip.compress(data, mtime=0))
            files[name] = {'file': path.name, 'sha256': sha256(data), 'bytes': len(data)}
        record = {'schema': SCHEMA, 'seq': seq, 'kind': kind, 'manifest_sha256': self.manifest_sha256,
                  'harness_sha256': harness_sha256(), **record, 'sidecars': files}
        with (directory / f'{stem}.json').open('xb') as out:
            out.write(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False).encode() + b'\n')
        return record

    def sidecar(self, record, name):
        entry = record['sidecars'][name]
        key = (entry['file'], entry['sha256'], entry['bytes'])
        if key not in self.memo:
            data = gzip.decompress((self.root / 'receipts' / entry['file']).read_bytes())
            if sha256(data) != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError(f'receipt {record["seq"]}: {name} sidecar does not match its hash')
            self.memo[key] = data
        return self.memo[key]

    def parsed(self, record, name, parse):
        key = (record['sidecars'][name]['sha256'], parse.__name__)
        if key not in self.memo:
            try:
                self.memo[key] = parse(self.sidecar(record, name))
            except ValueError as error:
                self.memo[key] = error
        if isinstance(self.memo[key], ValueError):
            raise self.memo[key]
        return self.memo[key]

    def artifact(self, entry):
        data = (self.root / 'artifacts' / entry['file']).read_bytes()
        if sha256(data) != entry['sha256']:
            raise ValueError(f'artifact {entry["file"]} does not match its hash')
        return data

    def history(self, session, before=None, receipts=None):
        """The session's completed turns, oldest first, as (request body, parsed assistant message, receipt)."""
        turns = []
        for r in receipts if receipts is not None else self.receipts():
            if r['kind'] != 'chat' or r['session'] != session or (before is not None and r['seq'] >= before):
                continue
            try:
                message = self.parsed(r, 'response', parse_sse)['message']
            except (ValueError, KeyError):
                continue
            turns.append((self.parsed(r, 'request', strict_json), message, r))
        return turns


def tokenize_result(raw):
    data = strict_json(raw)
    tokens = data.get('tokens')
    if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens) or
            data.get('count') != len(tokens)):
        raise ValueError('tokenize reply without a consistent count and token list')
    return {'count': len(tokens), 'max_model_len': data.get('max_model_len'),
            'tokens_sha256': sha256(','.join(map(str, tokens)).encode())}  # the engine's tokenSha256


def parse_logs(c, receipts):
    """Every timestamped native line that bears on the disk store, plus each log's coverage."""
    lines, coverage, untimed = [], [], 0
    for r in receipts:
        if r['kind'] != 'log':
            continue
        text = c.artifact(r['artifact']).decode('utf-8', 'replace')
        first = None
        for line in text.splitlines():
            m = TIMESTAMP.match(line)
            if not m:
                untimed += any(p.search(line) for p in (SAVE, READ, SKIP, STORE, CHECK))
                continue
            ts = when(f'{m[1]}.{(m[2] or "0")[:6].ljust(6, "0")}{"+00:00" if m[3] == "Z" else m[3]}')
            first = first or ts
            for kind, pattern in (('save', SAVE), ('read', READ), ('skip', SKIP), ('store', STORE), ('check', CHECK)):
                found = pattern.search(line)
                if found:
                    lines.append({'ts': ts, 'kind': kind, 'groups': found.groups(), 'artifact': r['artifact']['file'],
                                  'line': line[:400]})
        if first is not None:
            coverage.append((first, when(r['captured_at'])))
    return lines, coverage, untimed


class Verdict:
    def __init__(self, root, skew_s=2.0, tolerance=4096):
        self.c = Campaign(root)
        self.p = self.c.plan
        self.skew = timedelta(seconds=skew_s)
        self.tolerance = tolerance
        self.problems = []

    def run(self):
        c = self.c
        try:
            receipts = c.receipts()
        except ValueError as error:
            return self.report({'problems': [str(error)]})
        if sha256(c.plan_bytes) != c.manifest['plan_sha256']:
            self.problems.append('plan.json does not match the manifest')
        for r in receipts:
            if r.get('schema') != SCHEMA or r.get('manifest_sha256') != c.manifest_sha256:
                self.problems.append(f'receipt {r["seq"]} belongs to another manifest')
            if r.get('harness_sha256') != c.manifest['harness_sha256']:
                self.problems.append(f'receipt {r["seq"]} was written by a different harness than prepared the campaign')
            for name in r.get('sidecars', {}):
                try:
                    c.sidecar(r, name)
                except (OSError, ValueError) as error:
                    self.problems.append(str(error))
            for entry in ([r['artifact']] if r['kind'] == 'log' else []) + r.get('identity', []):
                try:
                    c.artifact(entry)
                except (OSError, ValueError) as error:
                    self.problems.append(str(error))
        if self.problems:
            return self.report({})
        self.receipts = receipts
        self.measures = {}
        for r in receipts:
            if r['kind'] == 'tokenize' and r['purpose'] in ('measure', 'image-prefix') and 'response' in r['sidecars']:
                try:
                    if r['complete']:
                        self.measures[(r['purpose'], r['request_sha256'])] = tokenize_result(c.sidecar(r, 'response'))
                except ValueError:
                    pass
        self.lines, self.coverage, untimed = parse_logs(c, receipts)
        turns = self.turns()
        workload_status = worst(*(t['status'] for t in turns.values())) if turns else 'unknown'
        missing = [f'{s}/{step}' for s in workload.SESSIONS for step in workload.BASE_STEPS[s] if (s, step) not in turns]
        if missing:
            workload_status = worst(workload_status, 'unknown')
        probes = self.probes()
        durability = {}
        for session, event in REQUIRED_PROBES:
            found = [x['status'] for x in probes if x['session'] == session and x['event'] == event]
            durability[f'{session}/{event}'] = worst(*found) if found else 'unknown'
        identity = self.identity()
        qualification = worst(workload_status, *durability.values(), identity['status'])
        image = durability['image/restart']
        production = {'fail': 'rejected', 'pass': 'eligible'}.get(qualification, 'unknown')
        notes = []
        if image == 'fail':
            notes.append('Images were not reused from disk after a restart: the release stays production-rejected.')
        if untimed:
            notes.append(f'{untimed} store lines had no timestamp and could not be attributed to a request.')
        return self.report({'workload': {'status': workload_status, 'missing': missing,
                                         'steps': [turns[k] for k in sorted(turns, key=lambda k: turns[k]['seq'])]},
                            'durability': {'status': worst(*durability.values()), 'required': durability,
                                           'probes': probes},
                            'identity': identity, 'qualification': qualification, 'production': production,
                            'notes': notes})

    def report(self, body):
        out = {'schema': SCHEMA, 'campaign': str(self.c.root), 'manifest_sha256': self.c.manifest_sha256,
               'release': self.c.manifest['release'], 'profile': self.c.manifest['profile'],
               'tolerance_tokens': self.tolerance, 'clock_skew_s': self.skew.total_seconds(),
               'problems': self.problems, **body}
        if self.problems or body.get('problems'):
            out.update(qualification='invalid', production='unknown')
        out['meaning'] = ('qualification pass means these receipts support recording the release as qualified; it is '
                          'recorded separately, by the release file procedure in AGENTS.md. Workload pass alone is not '
                          'qualification.')
        return out

    # Workload: every chat attempt, re-derived from its wire bytes and the plan.

    def turns(self):
        out, attempts = {}, {}
        for r in self.receipts:
            if r['kind'] != 'chat':
                continue
            t = self.turn(r)
            key = (r['session'], r['step'])
            attempts.setdefault(key, []).append(t)
            out[key] = dict(t, failed_attempts=[a['seq'] for a in attempts[key] if a['status'] != 'pass'])
            if out[key]['failed_attempts']:
                out[key]['status'] = 'fail'  # a later success does not hide an earlier failure
        return out

    def turn(self, r):
        c, p = self.c, self.p
        session, step = r['session'], r['step']
        reasons = []
        body = c.parsed(r, 'request', strict_json)
        out = {'seq': r['seq'], 'session': session, 'step': step, 'started_at': r['started_at'],
               'elapsed_s': r['elapsed_s'], 'timings': r['timings'], 'http_status': r['http_status']}
        if r['error'] or r['deadline_exceeded']:
            reasons.append(f'transport: {r["error"]}')
        try:
            reply = c.parsed(r, 'response', parse_sse)
            out.update(finish_reason=reply['finish_reason'], usage=reply['usage'],
                       cached_tokens_reported=reply['cached_tokens'], content=reply['message']['content'][:2000])
        except ValueError as error:
            reply = None
            reasons.append(f'incomplete or malformed reply: {error}')
        history = c.history(session, before=r['seq'], receipts=self.receipts)
        try:
            expected_body, expected, _ = workload.next_turn(p, session, step, [h[:2] for h in history],
                                                            r['meta'].get('chunks'))
            if canonical(expected_body) != canonical(body):
                reasons.append('request is not the plan with every earlier reply replayed exactly')
        except (ValueError, KeyError, IndexError) as error:
            expected = None
            reasons.append(f'request cannot be regenerated: {error}')
        if session == 'image' and step == 'first':
            try:
                urls = [x['image_url']['url'] for x in body['messages'][-1]['content'] if x['type'] == 'image_url']
                seen = [workload.quadrant_colors(base64.b64decode(u.split(',', 1)[1])) for u in urls]
            except (ValueError, KeyError, TypeError, IndexError) as error:
                seen = f'unreadable: {error}'
            out['image_colors'] = seen
            if seen != p['images']:
                reasons.append('images do not carry the planned quadrant colors')
        if reply and expected is not None:
            if session == 'pi' and step == 'call':
                results = workload.tool_results(p, reply['message'])
                good = reply['finish_reason'] == 'tool_calls' and results and all(
                    x['content'] == p['pi']['value'] for x in results)
                out['tool_call_ids'] = [x['tool_call_id'] for x in results]
            else:
                good = workload.correct(expected, reply['message']['content'])
            out['correct'] = bool(good)
            if not good:
                reasons.append('wrong or missing answer' if step != 'call' else 'no correct read_ledger call')
        measured = self.measures.get(('measure', sha256(canonical(workload.tokenize_body(body)))))
        out['measured_prompt_tokens'] = measured and measured['count']
        target, limit = self.target(session, step), p['ceiling']
        if measured is None:
            reasons.append('no recorded /tokenize measurement of this exact request')
        elif not target <= measured['count'] <= limit:
            reasons.append(f'measured {measured["count"]} tokens, outside {target}..{limit}')
        out['target_tokens'] = target
        out['reasons'] = reasons
        out['status'] = 'fail' if reasons else 'pass'
        return out

    def target(self, session, step):
        p = self.p
        if session == 'ledger' and step in workload.BASE_STEPS['ledger']:
            index = workload.BASE_STEPS['ledger'].index(step)
            return p['ledger_targets'][index] if index < len(p['ledger_targets']) else p['near_floor']
        return p['evictor_target'] if session == 'evictor' else 1

    # Durability: a continuation after a recorded lifecycle event, judged from the native log lines.

    def probes(self):
        out = []
        chats = [r for r in self.receipts if r['kind'] == 'chat']
        for cont in chats:
            if not cont['step'].startswith('continue-'):
                continue
            prior = [r for r in chats if r['session'] == cont['session'] and r['seq'] < cont['seq']]
            prior = [r for r in prior if self.complete(r)]
            events = [r for r in self.receipts if r['kind'] == 'lifecycle' and prior and
                      prior[-1]['seq'] < r['seq'] < cont['seq']]
            if not prior or not events:
                continue  # a warm repeat, not a durability probe
            out.append(self.probe(prior[-1], events[-1], cont))
        return out

    def completion(self, r):
        return self.c.parsed(r, 'response', parse_sse)['usage']['completion_tokens']

    def complete(self, r):
        try:
            self.c.parsed(r, 'response', parse_sse)
            return True
        except ValueError:
            return False

    def found(self, kind, start, end, floor, ceiling):
        """Lines in a request's time window whose token count fits it: a read longer than the request's own prompt
        belongs to another conversation, however close in time."""
        return [x for x in self.lines if x['kind'] == kind and start - self.skew <= x['ts'] <= end + self.skew and
                floor <= int(next(g for g in x['groups'] if g is not None)) <= ceiling]

    def covered(self, start, end):
        return any(first <= start - self.skew and captured >= end + self.skew for first, captured in self.coverage)

    def probe(self, prev, event, cont):
        session, kind = cont['session'], event['event']
        p_start, p_end = when(prev['started_at']), when(prev['ended_at'])
        at, c_start, c_end = when(event['at']), when(cont['started_at']), when(cont['ended_at'])
        out = {'session': session, 'event': kind, 'prior_seq': prev['seq'], 'lifecycle_seq': event['seq'],
               'continuation_seq': cont['seq'], 'reasons': []}
        prev_body = self.c.parsed(prev, 'request', strict_json)
        measured = self.measures.get(('measure', sha256(canonical(workload.tokenize_body(prev_body)))))
        if measured is None:
            out['reasons'].append('the prior request has no measured token count')
            return dict(out, status='unknown')
        floor = measured['count'] - self.tolerance
        if session == 'image':
            image_body = workload.tokenize_body(workload.chat_body(self.p, 'image', [
                {'role': 'system', 'content': workload.SYSTEM['image']},
                {'role': 'user', 'content': workload.image_parts(self.p)}]), add_generation_prompt=False)
            prefix = self.measures.get(('image-prefix', sha256(canonical(image_body))))
            if prefix is None:
                out['reasons'].append('no measured image prefix: cannot tell whether a read covered the images')
                return dict(out, status='unknown')
            floor = max(floor, prefix['count'])
            out['image_prefix_tokens'] = prefix['count']
        out['floor_tokens'] = floor
        cont_turn = self.turn(cont)
        out['continuation_status'] = cont_turn['status']
        status = 'pass'
        if cont_turn['status'] != 'pass':
            out['reasons'] += cont_turn['reasons']
            status = 'fail'
        cont_measured = cont_turn.get('measured_prompt_tokens')
        if cont_measured is None:
            return dict(out, status=worst(status, 'unknown'))
        held = measured['count'] + self.completion(prev)  # a save may include the prior reply
        saves = self.found('save', p_start, min(at, c_start), floor, held)
        skips = self.found('skip', p_start, min(at, c_start), floor, held)
        reads = self.found('read', c_start, c_end, floor, cont_measured)
        checks = [x for x in self.lines if x['kind'] == 'check' and p_start - self.skew <= x['ts'] <= c_end + self.skew]
        out.update(saves=[x['line'] for x in saves], skipped=[x['line'] for x in skips],
                   reads=[x['line'] for x in reads], check_failures=[x['line'] for x in checks])
        if checks:
            out['reasons'].append('the store reported a failed disk check')
            status = 'fail'
        if not saves:
            if skips:
                out['reasons'].append('the save was skipped or failed, so nothing durable existed to reuse')
                status = 'fail'
            else:
                out['reasons'].append('no retained save line for this conversation')
                status = worst(status, 'unknown')
        if kind in ('restart', 'crash'):
            ident = self.event_identity(event)
            out['identity'] = ident
            if ident != 'pass':
                out['reasons'].append(f'server identity after the {kind}: {ident}')
                status = worst(status, ident)
            stores = [x for x in self.lines if x['kind'] == 'store' and p_end < x['ts'] <= c_start + self.skew]
            good = [x for x in stores if x['groups'][0] == self.c.manifest['profile']['learn_dir'] and
                    int(x['groups'][4]) == self.c.manifest['ranks'] and int(x['groups'][1]) >= 1]
            out['store_opened'] = [x['line'] for x in stores]
            if not good:
                out['reasons'].append('no store-open line after the restart with the expected directory, ranks and '
                                      'at least one state')
                status = worst(status, 'fail' if stores or self.covered(c_start, c_end) else 'unknown')
        if not reads:
            if kind == 'eviction':
                resumes = self.resumes(session, event['seq'], cont['seq'])
                out['resume_from'] = resumes
                if 'disk' in resumes:
                    out['reasons'].append('the server said it would resume from disk but logged no read')
                    status = worst(status, 'fail')
                else:
                    out['reasons'].append('no disk read: eviction from RAM was not demonstrated')
                    status = worst(status, 'unknown')
            elif self.covered(c_start, c_end):
                out['reasons'].append('no disk read covering the conversation: the answer came from a cold prefill')
                status = worst(status, 'fail')
            else:
                out['reasons'].append('no retained log covers the continuation')
                status = worst(status, 'unknown')
        out['status'] = status
        return out

    def resumes(self, session, after, before):
        found = []
        for r in self.receipts:
            if r['kind'] == 'imprint' and r['session'] == session and after < r['seq'] < before and r['complete']:
                found.append(strict_json(self.c.sidecar(r, 'response')).get('resume_from'))
        return found

    def event_identity(self, event):
        digest = self.c.manifest['release']['image'].split('@', 1)[1]
        texts = [self.c.artifact(entry).decode('utf-8', 'replace') for entry in event.get('identity', [])]
        if not texts:
            return 'unknown'
        return 'pass' if all(digest in t for t in texts) else 'fail'

    def identity(self):
        events = [r for r in self.receipts if r['kind'] == 'lifecycle' and r['event'] in ('start', 'restart', 'crash')]
        result = {'events': {r['seq']: [r['event'], self.event_identity(r)] for r in events}}
        statuses = [s for _, s in result['events'].values()]
        if not any(r['event'] == 'start' for r in events):
            statuses.append('unknown')
        lengths = {m['max_model_len'] for m in self.measures.values()}
        result['max_model_len'] = sorted(lengths, key=str)
        if lengths and lengths != {self.p['context']}:
            statuses.append('fail')
        result['status'] = worst(*statuses) if statuses else 'unknown'
        return result


def verdict(root, skew_s=2.0, tolerance=4096):
    return Verdict(root, skew_s, tolerance).run()
