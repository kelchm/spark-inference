"""Qualification client tests. Only disposable loopback HTTP servers are contacted; nothing reaches a real endpoint.

Run from the repository root: python3 -m unittest tests.test_qualification
"""
import base64
import contextlib
from datetime import timedelta
import gzip
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification import evidence, transport, workload  # noqa: E402
from qualification.__main__ import main  # noqa: E402

DIGEST = 'sha256:' + 'ab' * 32
IMAGE = 'ghcr.io/kelchm/spark-inference/glm53-mia@' + DIGEST
LEARN = '/cache/huggingface/spark-inference/learn/test-agent-1m'
USAGE = {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13, 'prompt_tokens_details': {'cached_tokens': 0}}


class Loopback:
    """A throwaway HTTP server on 127.0.0.1; `respond(handler)` writes the whole reply.

    `paths` lists every request path, so tests count their own requests even if a local process probes the port.
    """

    def __init__(self, respond):
        outer = self
        self.paths = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *_):
                pass

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                outer.paths.append(self.path)
                length = int(self.headers['Content-Length'] or 0)
                self.body = json.loads(self.rfile.read(length)) if length else None  # read first: no RST on close
                respond(self)

            do_POST = do_GET

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


def send(handler, status, kind, data, length=True):
    handler.send_response(status)
    handler.send_header('Content-Type', kind)
    if length:
        handler.send_header('Content-Length', str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def sse(*events):
    return b''.join(b'data: ' + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b'\n\n' for e in events)


def event(delta, finish=None, usage=None):
    out = {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
    return dict(out, usage=usage) if usage else out


GOOD = sse(event({'role': 'assistant'}), event({'reasoning_content': 'think'}), event({'content': 'Q1=red'}),
           event({}, 'stop'), {'choices': [], 'usage': USAGE}, b'[DONE]')


class TransportTest(unittest.TestCase):
    def post(self, respond, deadline=5.0, **kwargs):
        with Loopback(respond) as server:
            result = transport.exchange(server.url, 'POST', '/v1/chat/completions', b'{}', deadline, stream=True,
                                        **kwargs)
            return result, server.paths.count('/v1/chat/completions')

    def test_complete_stream_is_reparsed_from_its_wire_bytes(self):
        result, _ = self.post(lambda h: send(h, 200, 'text/event-stream', GOOD, length=False))
        self.assertTrue(result['complete'], result['error'])
        reply = transport.parse_sse(result['raw'])
        self.assertEqual(reply['message']['content'], 'Q1=red')
        self.assertEqual(reply['message']['reasoning_content'], 'think')
        self.assertEqual(reply['cached_tokens'], 0)

    def test_stream_closed_before_done_is_partial_and_not_retried(self):
        partial = GOOD[:GOOD.index(b'data: [DONE]')]
        result, count = self.post(lambda h: send(h, 200, 'text/event-stream', partial, length=False))
        self.assertFalse(result['complete'])
        self.assertIn('[DONE]', result['error'])
        self.assertEqual(result['raw'], partial)  # what arrived is kept for the receipt
        self.assertEqual(count, 1)

    def test_length_finish_is_a_failure(self):
        cut = sse(event({'content': 'Q1=re'}), event({}, 'length', USAGE), b'[DONE]')
        result, _ = self.post(lambda h: send(h, 200, 'text/event-stream', cut, length=False))
        self.assertFalse(result['complete'])
        self.assertIn('length', result['error'])

    def test_error_event_after_finish_is_a_failure(self):
        looped = GOOD.replace(b'data: [DONE]', b'data: {"error": {"message": "loop stop"}}\n\ndata: [DONE]')
        result, _ = self.post(lambda h: send(h, 200, 'text/event-stream', looped, length=False))
        self.assertFalse(result['complete'])
        self.assertIn('loop stop', result['error'])

    def test_malformed_lines_and_duplicate_keys_are_refused(self):
        for raw in (b'garbage\n\n' + GOOD, GOOD.replace(b'{"choices": []', b'{"choices": [], "choices": []')):
            result, _ = self.post(lambda h, raw=raw: send(h, 200, 'text/event-stream', raw, length=False))
            self.assertFalse(result['complete'])
            self.assertTrue(result['error'])

    def test_trickle_cannot_outlast_the_total_deadline(self):
        def trickle(h):
            h.send_response(200)
            h.send_header('Content-Type', 'text/event-stream')
            h.end_headers()
            for _ in range(100):
                h.wfile.write(b': keepalive\n')
                h.wfile.flush()
                time.sleep(0.05)

        started = time.monotonic()
        result, _ = self.post(trickle, deadline=0.6)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertTrue(result['deadline_exceeded'])
        self.assertFalse(result['complete'])
        self.assertGreater(len(result['raw']), 0)

    def test_headers_that_never_come_hit_the_deadline(self):
        result, _ = self.post(lambda h: time.sleep(2), deadline=0.4)
        self.assertTrue(result['deadline_exceeded'])
        self.assertIsNone(result['http_status'])

    def test_non_200_and_wrong_type_fail_with_bytes_retained(self):
        result, _ = self.post(lambda h: send(h, 503, 'application/json', b'{"error": {"message": "busy"}}'))
        self.assertFalse(result['complete'])
        self.assertEqual(result['http_status'], 503)
        self.assertIn(b'busy', result['raw'])
        result, _ = self.post(lambda h: send(h, 200, 'application/json', b'{}'))
        self.assertFalse(result['complete'])

    def test_response_size_is_bounded(self):
        result, _ = self.post(lambda h: send(h, 200, 'text/event-stream', b': ' + b'x' * 5000 + b'\n' + GOOD,
                                             length=False), max_bytes=1000)
        self.assertFalse(result['complete'])
        self.assertIn('bound', result['error'])

    def test_json_body_shorter_than_its_length_fails(self):
        def short(h):
            h.send_response(200)
            h.send_header('Content-Type', 'application/json')
            h.send_header('Content-Length', '100')
            h.end_headers()
            h.wfile.write(b'{"count": 1}')

        with Loopback(short) as server:
            result = transport.exchange(server.url, 'POST', '/v1/tokenize', b'{}', 2.0)
        self.assertFalse(result['complete'])

    def test_tool_calls_keep_their_ids_and_must_match_the_finish(self):
        calls = sse(event({'reasoning_content': 'need data'}),
                    event({'tool_calls': [{'index': 0, 'id': 'call_A1', 'function': {'name': 'read_ledger',
                                                                                     'arguments': '{"key"'}}]}),
                    event({'tool_calls': [{'index': 0, 'function': {'arguments': ': "K"}'}}]}),
                    event({}, 'tool_calls', USAGE), b'[DONE]')
        reply = transport.parse_sse(calls)
        self.assertEqual(reply['message']['tool_calls'][0]['id'], 'call_A1')
        self.assertEqual(reply['message']['tool_calls'][0]['function']['arguments'], '{"key": "K"}')
        with self.assertRaises(ValueError):
            transport.parse_sse(calls.replace(b'"finish_reason": "tool_calls"', b'"finish_reason": "stop"'))

    def test_endpoint_must_be_explicit(self):
        for url in ('127.0.0.1:8888', 'http://u:p@host:1', 'http://host:1/?x=1', 'ftp://host'):
            with self.assertRaises(ValueError):
                transport.base_url(url)
        self.assertEqual(transport.base_url('http://h:8888/v1/')[1], '')


class WorkloadTest(unittest.TestCase):
    def test_fit_reaches_and_fills_with_bounded_calls(self):
        calls = []

        def count(n):
            calls.append(n)
            return 1000 + 997 * n

        n, tokens, _ = workload.fit(count, 300_000, 991_232)
        self.assertEqual(n, 300)  # 1000 + 997*300 = 300,100 >= target; 299 falls short
        self.assertEqual(tokens, 300_100)
        n, tokens, _ = workload.fit(count, 980_000, 991_232, reach=False)
        self.assertLessEqual(tokens, 991_232)
        self.assertGreater(1000 + 997 * (n + 1), 991_232)
        self.assertLessEqual(len(calls), 48)
        with self.assertRaisesRegex(ValueError, 'over the'):
            workload.fit(lambda n: 900_000 + 50_000 * n, 995_000, 991_232)  # reachable only past the limit
        with self.assertRaisesRegex(ValueError, 'do not reach'):
            workload.fit(lambda n: n, 10**6, 10**7, hi_cap=64)
        with self.assertRaises(RuntimeError):
            workload.fit(lambda n: n, 10**6, 10**7, max_calls=3, hi_cap=10**9)

    def test_planted_facts_are_distinct_and_present(self):
        codes = set()
        for i in range(1500):
            vault, code = workload.fact('seed', 'ledger', i)
            self.assertIn(f'the archive code for vault {vault} is {code}.', workload.chunk('seed', 'ledger', i))
            codes.add(code)
        self.assertEqual(len(codes), 1500)
        self.assertNotEqual(workload.chunk('seed', 'ledger', 3), workload.chunk('seed', 'evictor', 3))

    def test_images_decode_to_their_planned_colors(self):
        p = workload.plan('s', 'm', 'low', 1024)
        for colors in p['images']:
            self.assertEqual(len(set(colors)), 4)
            self.assertEqual(workload.quadrant_colors(workload.png(colors)), colors)
        with mock.patch.dict(workload.PALETTE, {'pink': (250, 180, 200)}):
            off_palette = workload.png(['red', 'blue', 'white', 'pink'])
        with self.assertRaisesRegex(ValueError, 'palette'):
            workload.quadrant_colors(off_palette)
        corrupt = workload.png(['red', 'blue', 'white', 'black']).replace(b'IEND', b'IENX')
        with self.assertRaisesRegex(ValueError, 'CRC'):
            workload.quadrant_colors(corrupt)

    def test_answers_must_all_be_right_and_unambiguous(self):
        expected = {'1': 'KESTREL-1234-AAB', '2': 'red'}
        self.assertTrue(workload.correct(expected, 'Q1=KESTREL-1234-AAB\nQ2=Red'))
        self.assertFalse(workload.correct(expected, 'Q1=KESTREL-1234-AAC\nQ2=red'))
        self.assertFalse(workload.correct(expected, 'Q1=KESTREL-1234-AAB'))
        self.assertFalse(workload.correct(expected, 'Q1=KESTREL-1234-AAB\nQ2=red\nQ2=blue'))

    def test_plan_keeps_replies_inside_the_reservation(self):
        with self.assertRaises(ValueError):
            workload.plan('s', 'm', 'low', 40_000)

    def test_pi_result_replays_reasoning_and_original_call_ids(self):
        p = workload.plan('s', 'm', 'low', 1024)
        body, _, _ = workload.next_turn(p, 'pi', 'call', [])
        message = {'role': 'assistant', 'content': '', 'reasoning_content': 'I should read it.',
                   'tool_calls': [{'id': 'call_9z', 'type': 'function',
                                   'function': {'name': 'read_ledger', 'arguments': json.dumps({'key': p['pi']['key']})}}]}
        result, _, _ = workload.next_turn(p, 'pi', 'result', [(body, message)])
        assistant, tool = result['messages'][-2:]
        self.assertEqual(assistant['reasoning_content'], 'I should read it.')
        self.assertEqual(assistant['tool_calls'][0]['id'], 'call_9z')
        self.assertEqual(tool, {'role': 'tool', 'tool_call_id': 'call_9z', 'content': p['pi']['value']})


SCALE = 10


class FakeEngine:
    """A loopback stand-in that genuinely reads the conversation: it finds planted facts, decodes images and follows
    the tool exchange. Its token count is monotone in the rendered text, like the real /tokenize."""

    def __init__(self):
        self.mode = 'ok'
        self.resume_from = 'disk'

    @staticmethod
    def count(body):
        text, images = 0, 0
        for m in body['messages']:
            parts = m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}]
            for part in parts:
                text += len(part.get('text', ''))
                images += part['type'] == 'image_url'
            text += len(m.get('reasoning_content', '')) + len(json.dumps(m.get('tool_calls', '')))
        return SCALE * text // 4 + 2000 * images + (5 if body.get('add_generation_prompt', True) else 0)

    def answer(self, body):
        messages = body['messages']
        last = messages[-1]
        if last['role'] == 'tool':
            return {'content': f'Q1={last["content"]}'}, 'stop'
        text = last['content'] if isinstance(last['content'], str) else last['content'][-1]['text']
        if 'read_ledger tool' in text:
            key = re.search(r'ledger key (\S+) with', text)[1]
            return {'tool_calls': [{'index': 0, 'id': 'call_X7', 'type': 'function',
                                    'function': {'name': 'read_ledger', 'arguments': json.dumps({'key': key})}}]}, \
                'tool_calls'
        everything = json.dumps(messages)
        lines = []
        for n, question in re.findall(r'Q(\d+): (.+)', text):
            if (vault := re.search(r'vault (\S+)\?', question)):
                lines.append(f'Q{n}=' + re.search(rf'vault {vault[1]} is (\S+)\.', everything)[1])
            elif (image := re.search(r'image (\d), what color is the (\S+) quadrant', question)):
                urls = [p['image_url']['url'] for m in messages if isinstance(m['content'], list)
                        for p in m['content'] if p['type'] == 'image_url']
                colors = workload.quadrant_colors(base64.b64decode(urls[int(image[1]) - 1].split(',', 1)[1]))
                lines.append(f'Q{n}=' + colors[workload.QUADRANTS.index(image[2])])
            elif 'value did read_ledger' in question:
                lines.append(f'Q{n}=' + next(m['content'] for m in messages if m['role'] == 'tool'))
            elif 'Which key' in question:
                lines.append(f'Q{n}=' + re.search(r'ledger key (\S+) with', everything)[1])
        if self.mode == 'wrong':
            lines = [re.sub(r'=.*', '=WRONG', line) for line in lines]
        return {'content': '\n'.join(lines)}, 'stop'

    def respond(self, h):
        if h.path == '/health':
            return send(h, 200, 'application/json', b'{"status": "ok", "model": "GLM-5.3-Flash"}')
        if h.path == '/v1/models':
            return send(h, 200, 'application/json', b'{"data": [{"id": "GLM-5.3-Flash"}]}')
        body = h.body
        if body is None or not h.path.startswith('/v1/'):
            return send(h, 404, 'application/json', b'{"error": {"message": "not found"}}')
        if h.path == '/v1/tokenize':
            n = self.count(body)
            data = json.dumps({'count': n, 'max_model_len': workload.CONTEXT, 'tokens': [7] * n}).encode()
            return send(h, 200, 'application/json', data)
        if h.path.startswith('/v1/imprint/'):
            data = {'object': 'imprint.resume', 'disk': True, 'tokens': self.count(body), 'resume_from': self.resume_from,
                    'ram_tokens': 0, 'disk_tokens': 0, 'disk_bytes': 0, 'wait_seconds': 0.1}
            return send(h, 200, 'application/json', json.dumps(data).encode())
        delta, finish = self.answer(body)
        events = [event({'role': 'assistant'}), event({'reasoning_content': f'Reading {len(body["messages"])} messages.'}),
                  event(delta), event({}, finish), {'choices': [], 'usage': USAGE}, b'[DONE]']
        raw = sse(*events)
        if self.mode == 'truncate':
            raw = raw[:len(raw) // 2]
        send(h, 200, 'text/event-stream', raw, length=False)


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([str(a) for a in args])
    return out.getvalue()


def fixtures(directory):
    release = directory / 'release.json'
    release.write_text(json.dumps({'release': 'glm53-mia-test', 'image': IMAGE,
                                   'engine': {'commit': 'e' * 40}}))
    profile = directory / 'profile.yaml'
    profile.write_text(f'container: "{IMAGE}"\nmetadata:\n  profile: "test-agent-1m"\nenv:\n  GLM_LEARN_DIR: "{LEARN}"\n')
    identity = directory / 'inspect.json'
    identity.write_text(json.dumps([{'Config': {'Image': IMAGE}}]))
    wrong = directory / 'inspect-other.json'
    wrong.write_text(json.dumps([{'Config': {'Image': IMAGE.replace('ab', 'cd')}}]))
    return release, profile, identity, wrong


def stamp(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f') + '123Z'  # docker's nanosecond form


def server_log(root, *, reads=True, saves=True, skip=False, image_short=False, store=True):
    """The rank 0 log a real store would write around the recorded requests; each flag removes one kind of evidence."""
    c = evidence.Campaign(root)
    receipts = c.receipts()
    chats = [r for r in receipts if r['kind'] == 'chat']
    first = evidence.when(receipts[0]['at'])  # the campaign opens with its start event
    lines = [(first - timedelta(seconds=60), 'info: server listening')]
    for r in chats:
        end = evidence.when(r['ended_at']) + timedelta(milliseconds=1)
        n = r['measured_prompt_tokens']
        if skip:
            lines.append((end, f'info: imprint: saving {n} tokens skipped (StagingFull)'))
        elif saves:
            lines.append((end, f'info: imprint: {n} tokens on disk: 9 new blocks and its own 80 MiB (90 MiB this state, '
                               f'10 MiB of rows it shares) in 40 ms'))
        if r['step'].startswith('continue-') and reads:
            prior = [x for x in chats if x['session'] == r['session'] and x['seq'] < r['seq']][-1]
            n = prior['measured_prompt_tokens']
            if r['session'] == 'image' and image_short:
                n = 2000  # a read that stops before the images
            start, finish = evidence.when(r['started_at']), evidence.when(r['ended_at'])
            lines.append((start + (finish - start) / 2,
                          f'info: imprint: read {n} tokens from disk (80 MiB in 30 ms)'))
    for r in receipts:
        if r['kind'] == 'lifecycle' and r['event'] in ('restart', 'crash') and store:
            lines.append((evidence.when(r['at']), f"info: glm: conversation states on disk in {LEARN} (4 states, "
                                                  f"300 MiB of 65536 MiB; 2 ranks' parts)"))
    return ''.join(f'{stamp(ts)} {text}\n' for ts, text in sorted(lines))


class CampaignTest(unittest.TestCase):
    """One full campaign against the fake engine, then offline verdicts over copies with different evidence."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.release, cls.profile, cls.identity, cls.wrong = fixtures(cls.tmp)
        cls.root = cls.tmp / 'campaign'
        cls.fake = FakeEngine()
        cli('prepare', '--out', cls.root, '--release', cls.release, '--profile', cls.profile, '--near-floor', 900_000)
        with Loopback(cls.fake.respond) as server:
            url = server.url

            def online(*args):
                return cli(*args[:1], '--campaign', cls.root, '--endpoint', url, *args[1:])

            cli('record-lifecycle', '--campaign', cls.root, '--event', 'start', '--identity', cls.identity)
            online('server')
            for _ in workload.BASE_STEPS['ledger']:
                online('turn', '--session', 'ledger')
            online('turn', '--session', 'image')
            online('turn', '--session', 'image-fork')
            online('turn', '--session', 'pi')
            online('turn', '--session', 'pi')
            cli('record-lifecycle', '--campaign', cls.root, '--event', 'restart', '--identity', cls.identity)
            online('imprint', '--session', 'ledger')
            for session in ('ledger', 'image', 'image-fork', 'pi'):
                online('turn', '--session', session, '--continue')
            online('turn', '--session', 'evictor')
            cli('record-lifecycle', '--campaign', cls.root, '--event', 'eviction')
            cls.fake.resume_from = 'ram'
            online('imprint', '--session', 'ledger')
            online('turn', '--session', 'ledger', '--continue')
            cli('record-lifecycle', '--campaign', cls.root, '--event', 'crash', '--identity', cls.identity)
            online('turn', '--session', 'ledger', '--continue')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def variant(self, **log):
        root = Path(tempfile.mkdtemp(dir=self.tmp)) / 'c'
        shutil.copytree(self.root, root)
        if log is not None and not log.pop('none', False):
            path = root.parent / 'rank0.log'
            path.write_text(server_log(root, **log))
            cli('import-log', '--campaign', root, '--file', path)
        return root

    def judge(self, root):
        # Both clocks in the loopback fixture are the same workstation clock.
        return evidence.verdict(root, skew_s=0)

    def test_ambiguous_adjacent_image_reads_cannot_be_credited_twice(self):
        result = evidence.verdict(self.variant(), skew_s=2)
        self.assertNotEqual(result['durability']['required']['image/restart'], 'pass')
        self.assertNotEqual(result['durability']['required']['image-fork/restart'], 'pass')

    def test_targets_were_measured_not_estimated(self):
        steps = {(t['session'], t['step']): t for t in self.judge(self.variant())['workload']['steps']}
        self.assertGreaterEqual(steps[('ledger', 'first')]['measured_prompt_tokens'], 300_000)
        final = steps[('ledger', 'grow-4')]['measured_prompt_tokens']
        self.assertTrue(900_000 <= final <= workload.CEILING - 24_576, final)
        self.assertTrue(all(t['measured_prompt_tokens'] <= workload.CEILING for t in steps.values()))

    def test_full_evidence_supports_qualification(self):
        result = self.judge(self.variant())
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['workload']['status'], 'pass', result['workload'])
        self.assertEqual(result['durability']['required'], {
            'ledger/restart': 'pass', 'ledger/crash': 'pass', 'image/restart': 'pass',
            'image-fork/restart': 'pass', 'pi/restart': 'pass',
            'ledger/eviction': 'pass'})
        self.assertEqual(result['identity']['status'], 'pass')
        self.assertEqual((result['qualification'], result['production']), ('pass', 'pending_manual_review'))
        self.assertEqual(result['large_first_save']['status'], 'pass')
        self.assertTrue(result['manual_gates'])

    def test_later_large_saves_cannot_substitute_for_the_first_save(self):
        root = self.variant(none=True)
        text = server_log(root)
        first = next(r for r in evidence.Campaign(root).receipts()
                     if r['kind'] == 'chat' and r['session'] == 'ledger' and r['step'] == 'first')
        n = first['measured_prompt_tokens']
        text = '\n'.join(line for line in text.splitlines() if f'imprint: {n} tokens on disk:' not in line)
        log = root.parent / 'no-first-save.log'
        log.write_text(text)
        cli('import-log', '--campaign', root, '--file', log)
        result = self.judge(root)
        self.assertEqual(result['large_first_save']['status'], 'unknown')
        self.assertEqual(result['qualification'], 'unknown')

    def test_image_fork_has_identical_text_but_different_pixels_and_answers(self):
        c = evidence.Campaign(self.root)
        turns = {r['session']: c.parsed(r, 'request', transport.strict_json)
                 for r in c.receipts() if r['kind'] == 'chat' and r['step'] == 'first'
                 and r['session'] in ('image', 'image-fork')}
        first, fork = turns['image'], turns['image-fork']
        self.assertEqual(first['messages'][0], fork['messages'][0])
        a, b = first['messages'][1]['content'], fork['messages'][1]['content']
        self.assertEqual([x for x in a if x['type'] == 'text'], [x for x in b if x['type'] == 'text'])
        self.assertNotEqual([x for x in a if x['type'] == 'image_url'], [x for x in b if x['type'] == 'image_url'])

    def test_aliased_image_fork_response_fails_even_with_disk_reads(self):
        root = self.variant()
        c = evidence.Campaign(root)
        first = next(r for r in c.receipts() if r['kind'] == 'chat' and r['session'] == 'image'
                     and r['step'] == 'first')
        fork_path = next((root / 'receipts').glob('*-chat-image-fork-first.json'))
        record = json.loads(fork_path.read_text())
        raw = c.sidecar(first, 'response')
        side = root / 'receipts' / record['sidecars']['response']['file']
        side.write_bytes(gzip.compress(raw))
        record['sidecars']['response'].update(sha256=transport.sha256(raw), bytes=len(raw))
        fork_path.write_text(json.dumps(record))
        result = self.judge(root)
        self.assertEqual(result['workload']['status'], 'fail')
        self.assertEqual(result['production'], 'rejected')

    def test_correct_answers_from_a_cold_prefill_do_not_pass(self):
        result = self.judge(self.variant(reads=False))
        self.assertEqual(result['workload']['status'], 'pass')
        required = result['durability']['required']
        self.assertEqual(required['ledger/restart'], 'fail')
        self.assertEqual(required['ledger/eviction'], 'unknown')  # the server said RAM: eviction not shown
        self.assertEqual((result['qualification'], result['production']), ('fail', 'rejected'))

    def test_missing_logs_leave_durability_unknown(self):
        result = self.judge(self.variant(none=True))
        self.assertEqual(result['workload']['status'], 'pass')
        self.assertEqual(set(result['durability']['required'].values()), {'unknown'})
        self.assertEqual(result['qualification'], 'unknown')

    def test_image_reuse_that_stops_before_the_images_rejects_production(self):
        result = self.judge(self.variant(image_short=True))
        self.assertEqual(result['durability']['required']['image/restart'], 'fail')
        self.assertEqual(result['durability']['required']['ledger/restart'], 'pass')
        self.assertEqual(result['production'], 'rejected')
        self.assertTrue(any('production-rejected' in n for n in result['notes']))

    def test_a_skipped_save_is_a_failure_not_a_pass(self):
        result = self.judge(self.variant(skip=True))
        probe = next(p for p in result['durability']['probes'] if p['session'] == 'ledger' and p['event'] == 'restart')
        self.assertEqual(probe['status'], 'fail')
        self.assertTrue(any('skipped' in r for r in probe['reasons']))

    def test_restart_without_the_store_reopening_fails(self):
        result = self.judge(self.variant(store=False))
        self.assertEqual(result['durability']['required']['ledger/restart'], 'fail')

    def test_identity_must_name_the_release_digest(self):
        root = self.variant()
        cli('record-lifecycle', '--campaign', root, '--event', 'restart', '--identity', self.wrong)
        result = self.judge(root)
        self.assertEqual(result['identity']['status'], 'fail')
        self.assertEqual(result['qualification'], 'fail')

    def test_tampered_wire_bytes_invalidate_the_campaign(self):
        root = self.variant()
        sidecar = next((root / 'receipts').glob('*-chat-ledger-first.response.gz'))
        sidecar.write_bytes(gzip.compress(gzip.decompress(sidecar.read_bytes()).replace(b'Q1=', b'Q1=X')))
        self.assertEqual(self.judge(root)['qualification'], 'invalid')

    def test_changed_harness_invalidates_verdict_and_blocks_endpoint_commands(self):
        root = self.variant()
        with mock.patch.object(evidence, 'harness_sha256', return_value='changed'):
            self.assertEqual(self.judge(root)['qualification'], 'invalid')
            with mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('network used')):
                with self.assertRaisesRegex(ValueError, 'harness changed'):
                    cli('server', '--campaign', root, '--endpoint', 'http://127.0.0.1:1')

    def rewrite_request(self, root, pattern, change):
        path = next((root / 'receipts').glob(pattern))
        record = json.loads(path.read_text())
        side = root / 'receipts' / record['sidecars']['request']['file']
        body = json.loads(gzip.decompress(side.read_bytes()))
        change(body)
        data = transport.canonical(body)
        side.write_bytes(gzip.compress(data))
        record['sidecars']['request'].update(sha256=transport.sha256(data), bytes=len(data))
        path.write_text(json.dumps(record))

    def test_dropped_reasoning_or_rewritten_call_id_fails_exact_replay(self):
        def drop_reasoning(body):
            del body['messages'][-2]['reasoning_content']

        def rename_call(body):
            body['messages'][-2]['tool_calls'][0]['id'] = 'call_other'
            body['messages'][-1]['tool_call_id'] = 'call_other'

        for change in (drop_reasoning, rename_call):
            root = self.variant()
            self.rewrite_request(root, '*-chat-pi-result.json', change)
            steps = {(t['session'], t['step']): t for t in self.judge(root)['workload']['steps']}
            self.assertEqual(steps[('pi', 'result')]['status'], 'fail')
            self.assertIn('replayed exactly', ' '.join(steps[('pi', 'result')]['reasons']))

    def test_offline_commands_open_no_sockets(self):
        root = self.variant(none=True)
        log = root.parent / 'x.log'
        log.write_text(server_log(root))
        with mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('network used')):
            cli('prepare', '--out', root.parent / 'again', '--release', self.release, '--profile', self.profile)
            cli('import-log', '--campaign', root, '--file', log)
            cli('record-lifecycle', '--campaign', root, '--event', 'eviction')
            evidence.verdict(root)


class FailureCampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        release, profile, _, _ = fixtures(self.tmp)
        self.root = self.tmp / 'c'
        cli('prepare', '--out', self.root, '--release', release, '--profile', profile)
        self.fake = FakeEngine()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def turn(self, url, *extra):
        return cli('turn', '--campaign', self.root, '--endpoint', url, '--session', 'image', *extra)

    def test_wrong_answer_fails_even_if_a_receipt_claims_correct(self):
        self.fake.mode = 'wrong'
        with Loopback(self.fake.respond) as server:
            with self.assertRaises(SystemExit) as result:
                self.turn(server.url)
            self.assertEqual(result.exception.code, 1)
        path = next((self.root / 'receipts').glob('*-chat-image-first.json'))
        record = json.loads(path.read_text())
        record['client_observed']['correct'] = True
        path.write_text(json.dumps(record))
        step = evidence.verdict(self.root)['workload']['steps'][0]
        self.assertEqual(step['status'], 'fail')
        self.assertIn('wrong or missing answer', step['reasons'])

    def test_partial_stream_is_recorded_blocks_the_session_and_stays_a_failure(self):
        with Loopback(self.fake.respond) as server:
            self.fake.mode = 'truncate'
            with self.assertRaises(SystemExit):
                self.turn(server.url)
            with self.assertRaisesRegex(SystemExit, 'did not complete'):
                self.turn(server.url)
            self.fake.mode = 'ok'
            self.assertIn('complete', self.turn(server.url, '--retry'))
        step = evidence.verdict(self.root)['workload']['steps'][0]
        self.assertEqual(step['status'], 'fail')
        self.assertEqual(len(step['failed_attempts']), 1)
        self.assertEqual(step['correct'], True)  # the retry answered; the earlier failure is not hidden

    def test_tokenize_deadline_stops_before_anything_is_sent(self):
        def slow(h):
            if h.path == '/v1/tokenize':
                time.sleep(2)
            self.fake.respond(h)

        with Loopback(slow) as server:
            with self.assertRaisesRegex(SystemExit, 'Deadline'):
                cli('turn', '--campaign', self.root, '--endpoint', server.url, '--session', 'image',
                    '--tokenize-deadline', '0.3')
            self.assertEqual(server.paths.count('/v1/tokenize'), 1)
            self.assertNotIn('/v1/chat/completions', server.paths)
        kinds = [p.name for p in (self.root / 'receipts').glob('*.json')]
        self.assertEqual(len(kinds), 1)
        self.assertIn('tokenize', kinds[0])


if __name__ == '__main__':
    unittest.main()
