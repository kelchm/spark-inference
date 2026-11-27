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

from qualification import coding, evidence, gates, host_state, transport, workload  # noqa: E402
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
            lines.append((evidence.when(r['at']), f"info: glm: conversation states on disk in {LEARN}/0123456789abcdef (4 states, "
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


class V136FakeEngine:
    """Loopback fake engine supporting exact token-granular fitting, images, reasoning_effort, and coding lookups."""

    def __init__(self, *, word_scale=1, strip_reasoning_in_tokenize=False, collapse_efforts=False,
                 cold_ref_diverge=False, alias_swap_image2=False, stall_s=0.0, stall_on_substring=None,
                 first_token_delay_s=0.0, chunk_delay_s=0.0, chunk_delay_on_substring='MOD_DEC_CONCURRENT_',
                 context_length=workload.CONTEXT, keyed_minimal_health=False, cold_instance=False,
                 bad_none_reasoning=False):
        self.word_scale = word_scale
        self.strip_reasoning_in_tokenize = strip_reasoning_in_tokenize
        self.collapse_efforts = collapse_efforts
        self.cold_ref_diverge = cold_ref_diverge
        self.alias_swap_image2 = alias_swap_image2
        self.stall_s = stall_s
        self.stall_on_substring = stall_on_substring
        self.first_token_delay_s = first_token_delay_s
        self.chunk_delay_s = chunk_delay_s
        self.chunk_delay_on_substring = chunk_delay_on_substring
        self.context_length = context_length
        self.keyed_minimal_health = keyed_minimal_health
        self.cold_instance = cold_instance
        self.bad_none_reasoning = bad_none_reasoning
        self.health_body = {'status': 'ok'} if keyed_minimal_health else {
            'status': 'ok',
            'model': 'GLM-5.3-Flash',
            'max_batch_size': 4,
            'warming': False,
        }
        self._first_image2_colors = None
        self._continuation_calls = 0
        self._warm_chats = 0

    def tokenize(self, body):
        tokens = []
        effort = body.get('reasoning_effort') or 'default'
        effort_id = 900 if self.collapse_efforts and effort in ('max', 'high') else (
            {'max': 901, 'high': 902, 'low': 903, 'none': 904}.get(effort, 900)
        )
        tokens.append(effort_id)
        for m in body['messages']:
            parts = m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}]
            for part in parts:
                if part.get('type') == 'text':
                    words = part.get('text', '').split()
                    tokens.extend([17] * (len(words) * self.word_scale))
                elif part.get('type') == 'image_url':
                    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                    w, h = workload.png_dimensions(raw)
                    # Vision token count reflects image geometry
                    img_tokens = (w // 16) * (h // 16)
                    tokens.extend([700 + (w % 100)] * img_tokens)
            if not self.strip_reasoning_in_tokenize and m.get('reasoning_content'):
                rwords = m['reasoning_content'].split()
                tokens.extend([500] * max(1, len(rwords)))
        if body.get('add_generation_prompt', True):
            tokens.append(999)
        return tokens

    def _answer(self, body):
        messages = body['messages']
        last = messages[-1]
        text = last['content'] if isinstance(last['content'], str) else last['content'][-1]['text']
        all_texts = []
        for m in messages:
            if isinstance(m.get('content'), str):
                all_texts.append(m['content'])
            elif isinstance(m.get('content'), list):
                for part in m['content']:
                    if part.get('type') == 'text':
                        all_texts.append(part.get('text', ''))
        everything = '\n'.join(all_texts)
        urls = [p['image_url']['url'] for m in messages if isinstance(m['content'], list)
                for p in m['content'] if p['type'] == 'image_url']
        decoded = [workload.quadrant_colors(base64.b64decode(u.split(',', 1)[1])) for u in urls]
        if len(decoded) >= 2 and self._first_image2_colors is None:
            self._first_image2_colors = list(decoded[1])

        lines = []
        for n, question in re.findall(r'Q(\d+): (.+)', text):
            if (vault := re.search(r'vault (\S+)\?', question)):
                lines.append(f'Q{n}=' + re.search(rf'vault {re.escape(vault[1])} is (\S+?)\.', everything)[1])
            elif (sym := re.search(r'assigned to (\S+)\?', question)):
                lines.append(f'Q{n}=' + re.search(rf'{re.escape(sym[1])} = "([^"]+)"', everything)[1])
            elif (image := re.search(r'image (\d), what color is the (\S+) quadrant', question)):
                idx = int(image[1]) - 1
                colors = decoded[idx]
                if idx == 1 and self.alias_swap_image2 and self._first_image2_colors is not None:
                    colors = self._first_image2_colors
                lines.append(f'Q{n}=' + colors[workload.QUADRANTS.index(image[2])])
        if 'Continuation questions:' in text:
            self._continuation_calls += 1
            if self.cold_ref_diverge and (self.cold_instance or self._continuation_calls >= 2):
                lines.append('extra-cold-divergence')
        return '\n'.join(lines)

    def respond(self, h):
        if h.path == '/health':
            return send(h, 200, 'application/json', json.dumps(self.health_body).encode())
        if h.path == '/v1/models':
            payload = {'data': [{'id': 'GLM-5.3-Flash', 'context_length': self.context_length,
                                 'max_model_len': self.context_length}]}
            return send(h, 200, 'application/json', json.dumps(payload).encode())
        body = h.body
        if h.path == '/v1/tokenize':
            toks = self.tokenize(body)
            data = json.dumps({'count': len(toks), 'max_model_len': self.context_length, 'tokens': toks}).encode()
            return send(h, 200, 'application/json', data)
        if h.path == '/v1/imprint/resume':
            if self.cold_instance:
                return send(h, 404, 'application/json', b'{"error": {"message": "learn disabled"}}')
            if self._warm_chats > 0:
                return send(h, 200, 'application/json',
                            b'{"resume_from":"ram","ram_tokens":4096,"disk_tokens":4096,"disk_bytes":65536}')
            return send(h, 200, 'application/json',
                        b'{"resume_from":"none","ram_tokens":0,"disk_tokens":0,"disk_bytes":0}')
        if h.path != '/v1/chat/completions':
            return send(h, 404, 'application/json', b'{"error": {"message": "not found"}}')

        self._warm_chats += 1
        content = self._answer(body)
        effort = body.get('reasoning_effort')
        reasoning = (
            'Unexpected leaked reasoning on none.' if effort == 'none' and self.bad_none_reasoning
            else '' if effort == 'none'
            else f'Thinking with effort {effort} across {len(body["messages"])} msgs.'
        )
        full_text = json.dumps(body)
        should_stall = self.stall_s > 0 and (self.stall_on_substring is None or self.stall_on_substring in full_text)
        should_first_delay = self.first_token_delay_s > 0 and (
            self.stall_on_substring is None or self.stall_on_substring in full_text
        )
        should_chunk_delay = self.chunk_delay_s > 0 and (
            self.chunk_delay_on_substring is None or self.chunk_delay_on_substring in full_text
        )

        if should_first_delay:
            time.sleep(self.first_token_delay_s)

        h.send_response(200)
        h.send_header('Content-Type', 'text/event-stream')
        h.end_headers()
        chunks = [event({'role': 'assistant'})]
        if reasoning:
            chunks.append(event({'reasoning_content': reasoning}))
        half = max(1, len(content) // 2)
        chunks.append(event({'content': content[:half]}))
        chunks.append(event({'content': content[half:]}))
        chunks.append(event({}, 'stop'))
        chunks.append({'choices': [], 'usage': {'prompt_tokens': 20, 'completion_tokens': 12, 'total_tokens': 32}})
        chunks.append(b'[DONE]')
        for i, item in enumerate(chunks):
            if should_stall and i == 3:
                time.sleep(self.stall_s)
            elif should_chunk_delay and i >= 2:
                time.sleep(self.chunk_delay_s)
            h.wfile.write(sse(item))
            h.wfile.flush()


class HostStateAndCapacityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release, self.profile, self.identity, _ = fixtures(self.tmp)
        self.root = self.tmp / 'campaign'
        cli('prepare', '--out', self.root, '--release', self.release, '--profile', self.profile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_host_state_records_boot_id_memavailable_and_pairs_fresh_vs_relaunch(self):
        meminfo = self.tmp / 'meminfo'
        meminfo.write_text(
            f'MemTotal: {128 * 1024 * 1024} kB\n'
            f'MemAvailable: {112 * 1024 * 1024} kB\n'
            'SwapTotal: 0 kB\nSwapFree: 0 kB\n'
        )
        zoneinfo = self.tmp / 'zoneinfo'
        zoneinfo.write_text('Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: 0\n      high_min: 0\n')
        with mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('network used')):
            cli('record-host-state', '--campaign', self.root, '--boot-profile', 'fresh-boot',
                '--boot-id', '11111111-2222-3333-4444-555555555555',
                '--meminfo', meminfo, '--zoneinfo', zoneinfo, '--rank', 'rank0', '--host', 'spark-1')
            partial = evidence.verdict(self.root, require_boot_pairing=True)
            self.assertFalse(partial['host_state']['paired'])
            self.assertEqual(partial['host_state']['missing_profiles'], ['ordinary-relaunch'])
            self.assertTrue(any('260 -> 236 GB/s' in n for n in partial['notes']))

            cli('record-host-state', '--campaign', self.root, '--boot-profile', 'ordinary-relaunch',
                '--boot-id', '11111111-2222-3333-4444-555555555555',
                '--meminfo', meminfo, '--zoneinfo', zoneinfo, '--rank', 'rank0', '--host', 'spark-1')
            # Rank 0 alone on a 2-rank campaign is not yet paired
            rank0_only = evidence.verdict(self.root, require_boot_pairing=True)
            self.assertFalse(rank0_only['host_state']['paired'])
            self.assertEqual(rank0_only['host_state']['missing_ranks'], ['rank1'])

            cli('record-host-state', '--campaign', self.root, '--boot-profile', 'fresh-boot',
                '--boot-id', '22222222-2222-3333-4444-555555555555',
                '--meminfo', meminfo, '--zoneinfo', zoneinfo, '--rank', 'rank1', '--host', 'spark-2')
            cli('record-host-state', '--campaign', self.root, '--boot-profile', 'ordinary-relaunch',
                '--boot-id', '22222222-2222-3333-4444-555555555555',
                '--meminfo', meminfo, '--zoneinfo', zoneinfo, '--rank', 'rank1', '--host', 'spark-2')
            paired = evidence.verdict(self.root, require_boot_pairing=True)
            self.assertTrue(paired['host_state']['paired'])
            self.assertEqual(paired['host_state']['status'], 'pass')
            self.assertEqual(paired['host_state']['missing_ranks'], [])
            rec0 = paired['host_state']['records']['fresh-boot'][0]
            self.assertEqual(rec0['boot_id'], '11111111-2222-3333-4444-555555555555')
            self.assertAlmostEqual(rec0['mem_available_gib'], 112.0, places=2)

    def test_capacity_checks_2582528_pool_floor_v136_log_kv_shard_and_failure(self):
        v136_log = self.tmp / 'v136-start.log'
        v136_log.write_text(
            "info: kv pool: the engine's own and 3 more taking 5044 pages of 512 rows (15.76 GiB)\n"
            f"info: glm: conversation states on disk in {LEARN}/0123456789abcdef "
            "(2 states, 128 MiB of 65536 MiB; 2 ranks' parts; a save stages up to 512 MiB)\n"
        )
        fake = V136FakeEngine()
        with Loopback(fake.respond) as srv:
            cli('capacity', '--campaign', self.root, '--endpoint', srv.url, '--log', v136_log,
                '--learn-gib', '64', '--check-learn-store')
        res = evidence.verdict(self.root)
        self.assertEqual(res['capacity']['status'], 'pass', res['capacity'])
        self.assertEqual(res['capacity']['checks'][0]['pool_tokens'], 2_582_528)
        self.assertEqual(res['capacity']['checks'][0]['learned_store']['ranks'], 2)

        # kv_shard health format also passes offline when context is observed
        shard_health = self.tmp / 'shard_health.json'
        shard_health.write_text(json.dumps({
            'status': 'ok', 'model': 'GLM-5.3-Flash', 'max_batch_size': 4, 'warming': False,
            'context_length': 1_048_576,
            'kv_shard': {'parts': 2, 'page_tokens': 512, 'part_pages': 2522},
        }))
        cli('capacity', '--campaign', self.root, '--health-file', shard_health)
        self.assertEqual(evidence.verdict(self.root)['capacity']['status'], 'pass')

        # Pool below 2,582,528 fails and rejects production
        low_log = self.tmp / 'low.log'
        low_log.write_text("info: kv pool: the engine's own and 3 more taking 5043 pages of 512 rows\n")
        with Loopback(fake.respond) as srv:
            with self.assertRaises(SystemExit) as exc:
                cli('capacity', '--campaign', self.root, '--endpoint', srv.url, '--log', low_log)
            self.assertEqual(exc.exception.code, 1)
        failed = evidence.verdict(self.root)
        self.assertEqual(failed['capacity']['status'], 'fail')
        self.assertEqual(failed['production'], 'rejected')

    def test_capacity_rejects_wrong_context_and_missing_store_log_and_accepts_keyed_minimal_health(self):
        v136_log = self.tmp / 'v136-start.log'
        v136_log.write_text(
            "info: kv pool: the engine's own and 3 more taking 5044 pages of 512 rows (15.76 GiB)\n"
            f"info: glm: conversation states on disk in {LEARN}/0123456789abcdef "
            "(2 states, 128 MiB of 65536 MiB; 2 ranks' parts; a save stages up to 512 MiB)\n"
        )
        # 1. Wrong context (8192) must fail even with 4 lanes and 2,582,528 pool tokens
        bad_ctx = self.tmp / 'bad_ctx'
        cli('prepare', '--out', bad_ctx, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(context_length=8192).respond) as srv:
            with self.assertRaises(SystemExit) as exc:
                cli('capacity', '--campaign', bad_ctx, '--endpoint', srv.url, '--log', v136_log)
            self.assertEqual(exc.exception.code, 1)
        v_bad = evidence.verdict(bad_ctx)
        self.assertEqual(v_bad['capacity']['status'], 'fail')
        self.assertEqual(v_bad['capacity']['checks'][0]['observed_context'], 8192)

        # 2. --check-learn-store without --log must fail instead of silently skipping
        no_log = self.tmp / 'no_log'
        cli('prepare', '--out', no_log, '--release', self.release, '--profile', self.profile)
        shard_health = self.tmp / 'shard_ctx.json'
        shard_health.write_text(json.dumps({
            'status': 'ok', 'model': 'GLM-5.3-Flash', 'max_batch_size': 4, 'warming': False,
            'context_length': 1_048_576,
            'kv_shard': {'parts': 2, 'page_tokens': 512, 'part_pages': 2522},
        }))
        with self.assertRaises(SystemExit) as exc:
            cli('capacity', '--campaign', no_log, '--health-file', shard_health, '--check-learn-store')
        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(evidence.verdict(no_log)['capacity']['status'], 'fail')

        # 3. Keyed minimal /health ({"status": "ok"}) passes using /v1/models + startup log
        keyed = self.tmp / 'keyed'
        cli('prepare', '--out', keyed, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(keyed_minimal_health=True).respond) as srv:
            cli('capacity', '--campaign', keyed, '--endpoint', srv.url, '--log', v136_log,
                '--learn-gib', '64', '--check-learn-store')
        v_keyed = evidence.verdict(keyed)
        self.assertEqual(v_keyed['capacity']['status'], 'pass', v_keyed['capacity'])
        self.assertEqual(v_keyed['capacity']['checks'][0]['streams'], 4)
        self.assertEqual(v_keyed['capacity']['checks'][0]['observed_context'], 1_048_576)

        # 4. HTTP 503 with healthy-looking JSON body must fail offline capacity verdict
        cap503 = self.tmp / 'cap503'
        cli('prepare', '--out', cap503, '--release', self.release, '--profile', self.profile)

        def health_503(h):
            body = json.dumps({
                'status': 'ok', 'model': 'GLM-5.3-Flash', 'max_batch_size': 4, 'warming': False,
                'context_length': 1_048_576, 'pool_tokens': 2_582_528,
            }).encode()
            return send(h, 503, 'application/json', body)

        with Loopback(health_503) as srv:
            with self.assertRaises(SystemExit):
                cli('capacity', '--campaign', cap503, '--endpoint', srv.url)
        v_503 = evidence.verdict(cap503)
        self.assertEqual(v_503['capacity']['status'], 'fail', v_503['capacity'])
        self.assertEqual(v_503['production'], 'rejected')

    def test_host_state_pairing_is_per_target_enforces_chronology_and_reparses_meminfo(self):
        meminfo = self.tmp / 'meminfo'
        meminfo.write_text(
            f'MemTotal: {128 * 1024 * 1024} kB\n'
            f'MemAvailable: {110 * 1024 * 1024} kB\n'
            'SwapTotal: 0 kB\nSwapFree: 0 kB\n'
        )
        # 1. Cross-rank pairing (rank0 fresh-boot + rank1 ordinary-relaunch) must NOT pass as paired
        cli('record-host-state', '--campaign', self.root, '--boot-profile', 'fresh-boot',
            '--boot-id', '11111111-2222-3333-4444-555555555555',
            '--meminfo', meminfo, '--rank', 'rank0', '--host', 'spark-1')
        cli('record-host-state', '--campaign', self.root, '--boot-profile', 'ordinary-relaunch',
            '--boot-id', '22222222-2222-3333-4444-555555555555',
            '--meminfo', meminfo, '--rank', 'rank1', '--host', 'spark-2')
        cross = evidence.verdict(self.root, require_boot_pairing=True)
        self.assertFalse(cross['host_state']['paired'])
        self.assertEqual(cross['host_state']['status'], 'unknown')

        # 2. Tampering with summary mem_available_gib in receipt is ignored because raw meminfo sidecar is re-parsed
        rec_path = next((self.root / 'receipts').glob('*host-state-fresh-boot*.json'))
        rec = json.loads(rec_path.read_text())
        rec['mem_available_gib'] = 999.0
        rec_path.write_text(json.dumps(rec))
        reparsed = evidence.verdict(self.root)
        self.assertAlmostEqual(reparsed['host_state']['records']['fresh-boot'][0]['mem_available_gib'], 110.0, places=2)

        # 3. Inverted chronology on the same (host, rank, boot_id) fails host_state
        inv_root = self.tmp / 'inv'
        cli('prepare', '--out', inv_root, '--release', self.release, '--profile', self.profile)
        cli('record-host-state', '--campaign', inv_root, '--boot-profile', 'ordinary-relaunch',
            '--boot-id', '33333333-2222-3333-4444-555555555555',
            '--meminfo', meminfo, '--rank', 'rank0', '--host', 'spark-1')
        time.sleep(0.01)
        cli('record-host-state', '--campaign', inv_root, '--boot-profile', 'fresh-boot',
            '--boot-id', '33333333-2222-3333-4444-555555555555',
            '--meminfo', meminfo, '--rank', 'rank0', '--host', 'spark-1')
        inv = evidence.verdict(inv_root, require_boot_pairing=True)
        self.assertEqual(inv['host_state']['status'], 'fail', inv['host_state'])
        self.assertFalse(inv['host_state']['paired'])

        # 4. Nonexistent --lifecycle-seq reference fails host_state
        bad_lc = self.tmp / 'bad_lc'
        cli('prepare', '--out', bad_lc, '--release', self.release, '--profile', self.profile)
        cli('record-host-state', '--campaign', bad_lc, '--boot-profile', 'fresh-boot',
            '--boot-id', '44444444-2222-3333-4444-555555555555',
            '--meminfo', meminfo, '--rank', 'rank0', '--host', 'spark-1', '--lifecycle-seq', '999')
        v_lc = evidence.verdict(bad_lc, require_boot_pairing=True)
        self.assertEqual(v_lc['host_state']['status'], 'fail', v_lc['host_state'])
        self.assertFalse(v_lc['host_state']['paired'])


class DenseSparseGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release, self.profile, self.identity, _ = fixtures(self.tmp)
        self.root = self.tmp / 'campaign'
        cli('prepare', '--out', self.root, '--release', self.release, '--profile', self.profile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_dense_2051_sparse_2052_page_reuse_and_restart_restore(self):
        fake = V136FakeEngine(word_scale=1)
        with Loopback(fake.respond) as srv:
            cli('record-lifecycle', '--campaign', self.root, '--event', 'start', '--identity', self.identity)
            cli('dense-sparse', '--campaign', self.root, '--endpoint', srv.url, '--case', 'dense-2051')
            cli('dense-sparse', '--campaign', self.root, '--endpoint', srv.url, '--case', 'sparse-2052')
            cli('dense-sparse', '--campaign', self.root, '--endpoint', srv.url, '--case', 'page-reuse')
            cli('record-lifecycle', '--campaign', self.root, '--event', 'restart', '--identity', self.identity)
            cli('dense-sparse', '--campaign', self.root, '--endpoint', srv.url, '--case', 'restart-restore')

        c = evidence.Campaign(self.root)
        receipts = c.receipts()
        sparse_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('case') == 'sparse-2052')
        restore_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('case') == 'restart-restore')
        restart_ev = next(r for r in receipts if r.get('kind') == 'lifecycle' and r.get('event') == 'restart')
        t0 = evidence.when(sparse_rec['started_at']) - timedelta(seconds=10)
        t_save = evidence.when(sparse_rec['ended_at']) + timedelta(milliseconds=1)
        t_open = evidence.when(restart_ev['at'])
        r_start, r_end = evidence.when(restore_rec['started_at']), evidence.when(restore_rec['ended_at'])
        t_read = r_start + (r_end - r_start) / 2
        log_path = self.tmp / 'rank0.log'
        log_path.write_text(
            f'{stamp(t0)} info: server listening\n'
            f'{stamp(t_save)} info: imprint: 2052 tokens on disk: 2 new blocks and its own 12 MiB (12 MiB this state, 0 MiB of rows it shares) in 10 ms\n'
            f"{stamp(t_open)} info: glm: conversation states on disk in {LEARN}/0123456789abcdef (1 states, 12 MiB of 65536 MiB; 2 ranks' parts; a save stages up to 256 MiB)\n"
            f'{stamp(t_read)} info: imprint: read 2052 tokens from disk (12 MiB in 8 ms)\n'
        )
        cli('import-log', '--campaign', self.root, '--file', log_path)
        verdict_ok = evidence.verdict(self.root, skew_s=0)
        ds = verdict_ok['gates']['dense_sparse']
        self.assertEqual(ds['status'], 'pass', ds)
        self.assertEqual(ds['cases']['dense-2051']['measured_prompt_tokens'], 2051)
        self.assertEqual(ds['cases']['sparse-2052']['measured_prompt_tokens'], 2052)
        self.assertGreater(ds['cases']['page-reuse']['measured_prompt_tokens'], 2052)
        self.assertEqual(ds['cases']['restart-restore']['status'], 'pass')

        # Cold prefill (covered log with no disk read) fails restart-restore
        cold_root = self.tmp / 'cold'
        shutil.copytree(self.root, cold_root)
        # Replace imported log artifact & receipt cleanly via fresh copy before import-log
        cold_log = self.tmp / 'cold.log'
        cold_log.write_text(
            f'{stamp(t0)} info: server listening\n'
            f'{stamp(t_save)} info: imprint: 2052 tokens on disk: 2 new blocks and its own 12 MiB (12 MiB this state, 0 MiB of rows it shares) in 10 ms\n'
        )
        cold_only = self.tmp / 'cold_only'
        cli('prepare', '--out', cold_only, '--release', self.release, '--profile', self.profile)
        with Loopback(fake.respond) as srv:
            cli('dense-sparse', '--campaign', cold_only, '--endpoint', srv.url, '--case', 'dense-2051')
            cli('dense-sparse', '--campaign', cold_only, '--endpoint', srv.url, '--case', 'sparse-2052')
            cli('record-lifecycle', '--campaign', cold_only, '--event', 'restart', '--identity', self.identity)
            cli('dense-sparse', '--campaign', cold_only, '--endpoint', srv.url, '--case', 'restart-restore')
        cold_log.write_text(f'{stamp(t0)} info: server listening\n')
        cli('import-log', '--campaign', cold_only, '--file', cold_log)
        v_cold = evidence.verdict(cold_only, skew_s=0)
        self.assertEqual(v_cold['gates']['dense_sparse']['cases']['restart-restore']['status'], 'fail')
        self.assertEqual(v_cold['production'], 'rejected')

    def test_dense_sparse_partial_campaign_is_unknown_and_one_token_read_fails(self):
        fake = V136FakeEngine(word_scale=1)
        # 1. Partial campaign (only 2051 and 2052) must stay unknown, never pass
        partial = self.tmp / 'partial_ds'
        cli('prepare', '--out', partial, '--release', self.release, '--profile', self.profile)
        with Loopback(fake.respond) as srv:
            cli('dense-sparse', '--campaign', partial, '--endpoint', srv.url, '--case', 'dense-2051')
            cli('dense-sparse', '--campaign', partial, '--endpoint', srv.url, '--case', 'sparse-2052')
        v_partial = evidence.verdict(partial, skew_s=0)
        self.assertEqual(v_partial['gates']['dense_sparse']['status'], 'unknown')
        self.assertEqual(v_partial['gates']['dense_sparse']['missing_required'], ['page-reuse', 'restart-restore'])

        # 2. A 1-token disk read on restart-restore is below the 2048-token page floor and must fail
        tiny = self.tmp / 'tiny_read_ds'
        cli('prepare', '--out', tiny, '--release', self.release, '--profile', self.profile)
        with Loopback(fake.respond) as srv:
            cli('record-lifecycle', '--campaign', tiny, '--event', 'start', '--identity', self.identity)
            cli('dense-sparse', '--campaign', tiny, '--endpoint', srv.url, '--case', 'dense-2051')
            cli('dense-sparse', '--campaign', tiny, '--endpoint', srv.url, '--case', 'sparse-2052')
            cli('dense-sparse', '--campaign', tiny, '--endpoint', srv.url, '--case', 'page-reuse')
            cli('record-lifecycle', '--campaign', tiny, '--event', 'restart', '--identity', self.identity)
            cli('dense-sparse', '--campaign', tiny, '--endpoint', srv.url, '--case', 'restart-restore')
        receipts = evidence.Campaign(tiny).receipts()
        sparse_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('case') == 'sparse-2052')
        restore_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('case') == 'restart-restore')
        restart_ev = next(r for r in receipts if r.get('kind') == 'lifecycle' and r.get('event') == 'restart')
        t0 = evidence.when(sparse_rec['started_at']) - timedelta(seconds=10)
        t_save = evidence.when(sparse_rec['ended_at']) + timedelta(milliseconds=1)
        t_open = evidence.when(restart_ev['at'])
        t_read = evidence.when(restore_rec['started_at']) + (
            evidence.when(restore_rec['ended_at']) - evidence.when(restore_rec['started_at'])
        ) / 2
        tiny_log = self.tmp / 'tiny.log'
        tiny_log.write_text(
            f'{stamp(t0)} info: server listening\n'
            f'{stamp(t_save)} info: imprint: 2052 tokens on disk: 2 new blocks and its own 12 MiB (12 MiB this state, 0 MiB of rows it shares) in 10 ms\n'
            f"{stamp(t_open)} info: glm: conversation states on disk in {LEARN}/0123456789abcdef (1 states, 12 MiB of 65536 MiB; 2 ranks' parts; a save stages up to 256 MiB)\n"
            f'{stamp(t_read)} info: imprint: read 1 tokens from disk (1 MiB in 1 ms)\n'
        )
        cli('import-log', '--campaign', tiny, '--file', tiny_log)
        v_tiny = evidence.verdict(tiny, skew_s=0)
        self.assertEqual(v_tiny['gates']['dense_sparse']['cases']['restart-restore']['status'], 'fail')
        self.assertEqual(v_tiny['gates']['dense_sparse']['status'], 'fail')


class DurableImageGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release, self.profile, self.identity, _ = fixtures(self.tmp)
        self.root = self.tmp / 'campaign'
        cli('prepare', '--out', self.root, '--release', self.release, '--profile', self.profile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run_full_gate(self, root, fake, *, over_read_on=None, cold_ref_diverge=False, omit_swap_reads=False):
        cold_fake = V136FakeEngine(word_scale=1, cold_instance=True, cold_ref_diverge=cold_ref_diverge)
        with Loopback(fake.respond) as srv, Loopback(cold_fake.respond) as cold_srv:
            cli('record-lifecycle', '--campaign', root, '--event', 'start', '--identity', self.identity)
            cli('image-durable', '--campaign', root, '--endpoint', srv.url, '--phase', 'initial', '--target-tokens', '4096')
            cli('record-lifecycle', '--campaign', root, '--event', 'restart', '--identity', self.identity)
            cli('image-durable', '--campaign', root, '--endpoint', srv.url, '--phase', 'restart')
            cli('image-durable', '--campaign', root, '--endpoint', cold_srv.url, '--phase', 'cold-ref')
            cli('image-durable', '--campaign', root, '--endpoint', srv.url, '--phase', 'mid-cut')
            cli('image-durable', '--campaign', root, '--endpoint', srv.url, '--phase', 'swap-pixel')
            cli('image-durable', '--campaign', root, '--endpoint', srv.url, '--phase', 'swap-geometry')

        c = evidence.Campaign(root)
        receipts = c.receipts()
        init_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('phase') == 'initial')
        restart_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('phase') == 'restart')
        pre_img2_tok = next(r['count'] for r in receipts if r.get('kind') == 'tokenize' and r.get('purpose') == 'pre-image2-prefix')
        restart_ev = next(r for r in receipts if r.get('kind') == 'lifecycle' and r.get('event') == 'restart')
        t0 = evidence.when(init_rec['started_at']) - timedelta(seconds=10)
        lines = [
            (t0, 'info: server listening'),
            (evidence.when(init_rec['ended_at']) + timedelta(milliseconds=1),
             f'info: imprint: {init_rec["measured_prompt_tokens"]} tokens on disk: 4 new blocks and its own 24 MiB (24 MiB this state, 0 MiB of rows it shares) in 12 ms'),
            (evidence.when(restart_ev['at']),
             f"info: glm: conversation states on disk in {LEARN}/0123456789abcdef (1 states, 24 MiB of 65536 MiB; 2 ranks' parts; a save stages up to 256 MiB)"),
            (evidence.when(restart_rec['started_at']) + (evidence.when(restart_rec['ended_at']) - evidence.when(restart_rec['started_at'])) / 2,
             f'info: imprint: read {init_rec["measured_prompt_tokens"]} tokens from disk (24 MiB in 9 ms)'),
        ]
        if not omit_swap_reads:
            for phase in ('swap-pixel', 'swap-geometry'):
                srec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('phase') == phase)
                mid_ts = evidence.when(srec['started_at']) + (evidence.when(srec['ended_at']) - evidence.when(srec['started_at'])) / 2
                read_n = (pre_img2_tok + 128) if over_read_on == phase else pre_img2_tok
                lines.append((mid_ts, f'info: imprint: read {read_n} tokens from disk (6 MiB in 4 ms)'))

        log_path = root.parent / f'{root.name}.log'
        log_path.write_text(''.join(f'{stamp(ts)} {text}\n' for ts, text in sorted(lines)))
        cli('import-log', '--campaign', root, '--file', log_path, '--label', 'rank0-primary')
        cold_rec = next(r for r in receipts if r.get('kind') == 'gate' and r.get('phase') == 'cold-ref')
        cold_log = root.parent / f'{root.name}-cold.log'
        cold_log.write_text(f'{stamp(evidence.when(cold_rec["started_at"]) - timedelta(seconds=1))} info: cold server listening\n')
        cli('import-log', '--campaign', root, '--file', cold_log, '--label', 'rank0-cold-ref')
        return evidence.verdict(root, skew_s=0)

    def test_durable_two_image_gate_passes_all_six_phases(self):
        v = self._run_full_gate(self.root, V136FakeEngine(word_scale=1))
        ig = v['gates']['image_durable']
        self.assertEqual(ig['status'], 'pass', ig)
        self.assertTrue(ig['restart_matches_cold_prefill'])
        for phase in gates.DURABLE_IMAGE_PHASES:
            self.assertEqual(ig['phases'][phase]['status'], 'pass', ig['phases'][phase])
        self.assertGreater(ig['phases']['initial']['measured_prompt_tokens'], gates.SPARSE_MIN_VISIBLE)

    def test_swap_pixel_and_geometry_fail_if_prefix_reuse_crosses_image2_boundary(self):
        for bad_phase in ('swap-pixel', 'swap-geometry'):
            with self.subTest(bad_phase=bad_phase):
                root = self.tmp / f'over-{bad_phase}'
                cli('prepare', '--out', root, '--release', self.release, '--profile', self.profile)
                v = self._run_full_gate(root, V136FakeEngine(word_scale=1), over_read_on=bad_phase)
                self.assertEqual(v['gates']['image_durable']['phases'][bad_phase]['status'], 'fail')
                self.assertEqual(v['production'], 'rejected')

    def test_cold_divergence_and_image2_aliasing_both_fail_gate(self):
        root_div = self.tmp / 'div'
        cli('prepare', '--out', root_div, '--release', self.release, '--profile', self.profile)
        v_div = self._run_full_gate(root_div, V136FakeEngine(word_scale=1), cold_ref_diverge=True)
        self.assertFalse(v_div['gates']['image_durable']['restart_matches_cold_prefill'])
        self.assertEqual(v_div['gates']['image_durable']['status'], 'fail')

        root_alias = self.tmp / 'alias'
        cli('prepare', '--out', root_alias, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1, alias_swap_image2=True).respond) as srv:
            cli('image-durable', '--campaign', root_alias, '--endpoint', srv.url, '--phase', 'initial', '--target-tokens', '4096')
            with self.assertRaises(SystemExit) as exc:
                cli('image-durable', '--campaign', root_alias, '--endpoint', srv.url, '--phase', 'swap-pixel')
            self.assertEqual(exc.exception.code, 1)
        self.assertEqual(evidence.verdict(root_alias, skew_s=0)['gates']['image_durable']['status'], 'fail')

    def test_image_durable_partial_unknown_warm_cold_ref_fails_and_absent_swap_read_unknown(self):
        # 1. Only running initial cannot pass image-durable
        partial = self.tmp / 'partial_img'
        cli('prepare', '--out', partial, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1).respond) as srv:
            cli('image-durable', '--campaign', partial, '--endpoint', srv.url, '--phase', 'initial', '--target-tokens', '4096')
        v_partial = evidence.verdict(partial, skew_s=0)
        self.assertEqual(v_partial['gates']['image_durable']['status'], 'unknown')
        self.assertTrue(v_partial['gates']['image_durable']['missing_phases'])

        # 2. Running cold-ref against the SAME warm endpoint fails
        warm_ref = self.tmp / 'warm_ref'
        cli('prepare', '--out', warm_ref, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1).respond) as srv:
            cli('record-lifecycle', '--campaign', warm_ref, '--event', 'start', '--identity', self.identity)
            cli('image-durable', '--campaign', warm_ref, '--endpoint', srv.url, '--phase', 'initial', '--target-tokens', '4096')
            cli('record-lifecycle', '--campaign', warm_ref, '--event', 'restart', '--identity', self.identity)
            cli('image-durable', '--campaign', warm_ref, '--endpoint', srv.url, '--phase', 'restart')
            cli('image-durable', '--campaign', warm_ref, '--endpoint', srv.url, '--phase', 'cold-ref')
        v_warm = evidence.verdict(warm_ref, skew_s=0)
        self.assertEqual(v_warm['gates']['image_durable']['phases']['cold-ref']['status'], 'fail')
        self.assertFalse(v_warm['gates']['image_durable']['phases']['cold-ref']['cold_instance_isolated'])

        # 3. Absent swap reuse trace leaves swap phases unknown, never pass
        no_swap = self.tmp / 'no_swap_reads'
        cli('prepare', '--out', no_swap, '--release', self.release, '--profile', self.profile)
        v_noswap = self._run_full_gate(no_swap, V136FakeEngine(word_scale=1), omit_swap_reads=True)
        self.assertEqual(v_noswap['gates']['image_durable']['phases']['swap-pixel']['status'], 'unknown')
        self.assertEqual(v_noswap['gates']['image_durable']['phases']['swap-geometry']['status'], 'unknown')
        self.assertEqual(v_noswap['gates']['image_durable']['status'], 'unknown')

        # 4. A later clean cold-ref retry cannot erase an earlier restart/cold-ref divergence
        retry_div = self.tmp / 'retry_div'
        cli('prepare', '--out', retry_div, '--release', self.release, '--profile', self.profile)
        self._run_full_gate(retry_div, V136FakeEngine(word_scale=1), cold_ref_diverge=True)
        with Loopback(V136FakeEngine(word_scale=1, cold_instance=True).respond) as clean_cold:
            cli('image-durable', '--campaign', retry_div, '--endpoint', clean_cold.url, '--phase', 'cold-ref')
        v_retry_div = evidence.verdict(retry_div, skew_s=0)
        self.assertFalse(v_retry_div['gates']['image_durable']['restart_matches_cold_prefill'])
        self.assertEqual(v_retry_div['gates']['image_durable']['status'], 'fail')


class ReasoningMatrixGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release, self.profile, _, _ = fixtures(self.tmp)
        self.root = self.tmp / 'campaign'
        cli('prepare', '--out', self.root, '--release', self.release, '--profile', self.profile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_clear_thinking_0_and_max_high_low_none_efforts_pass(self):
        with Loopback(V136FakeEngine(word_scale=1).respond) as srv:
            cli('reasoning-matrix', '--campaign', self.root, '--endpoint', srv.url)
        v = evidence.verdict(self.root, skew_s=0)
        rm = v['gates']['reasoning_matrix']
        self.assertEqual(rm['status'], 'pass', rm)
        self.assertTrue(rm['distinct_renders'])
        for effort in ('max', 'high', 'low'):
            pair = rm['clear_thinking_tokens'][effort]
            self.assertGreater(pair['clear-thinking-with'], pair['clear-thinking-without'])

    def test_clear_thinking_strip_or_effort_collapse_fails(self):
        for label, kwargs in (
            ('stripped', {'strip_reasoning_in_tokenize': True}),
            ('collapsed', {'collapse_efforts': True}),
        ):
            with self.subTest(label=label):
                root = self.tmp / label
                cli('prepare', '--out', root, '--release', self.release, '--profile', self.profile)
                with Loopback(V136FakeEngine(word_scale=1, **kwargs).respond) as srv:
                    cli('reasoning-matrix', '--campaign', root, '--endpoint', srv.url)
                v = evidence.verdict(root, skew_s=0)
                self.assertEqual(v['gates']['reasoning_matrix']['status'], 'fail', v['gates']['reasoning_matrix'])
                self.assertEqual(v['production'], 'rejected')

    def test_reasoning_matrix_rejects_none_reasoning_uses_identical_render_prompt_and_marks_subset_unknown(self):
        # 1. Identical messages across all 4 effort-render probes so ignoring reasoning_effort cannot pass
        plan = evidence.Campaign(self.root).plan
        render_msgs = [gates.reasoning_render_body(plan, eff)['messages'] for eff in gates.REASONING_EFFORTS]
        self.assertEqual(len({json.dumps(m, sort_keys=True) for m in render_msgs}), 1)

        # 2. Effort 'none' emitting reasoning_content must fail
        bad_none = self.tmp / 'bad_none'
        cli('prepare', '--out', bad_none, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1, bad_none_reasoning=True).respond) as srv:
            cli('reasoning-matrix', '--campaign', bad_none, '--endpoint', srv.url)
        v_none = evidence.verdict(bad_none, skew_s=0)
        self.assertEqual(v_none['gates']['reasoning_matrix']['status'], 'fail', v_none['gates']['reasoning_matrix'])
        self.assertEqual(v_none['gates']['reasoning_matrix']['efforts']['none']['status'], 'fail')

        # 3. Missing efforts / missing render receipts leave reasoning-matrix unknown
        subset = self.tmp / 'subset_rm'
        cli('prepare', '--out', subset, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1).respond) as srv:
            cli('reasoning-matrix', '--campaign', subset, '--endpoint', srv.url, '--efforts', 'max,high')
        v_sub = evidence.verdict(subset, skew_s=0)
        self.assertEqual(v_sub['gates']['reasoning_matrix']['status'], 'unknown')
        self.assertEqual(v_sub['gates']['reasoning_matrix']['missing_renders'], ['low', 'none'])
        self.assertEqual(v_sub['gates']['reasoning_matrix']['missing_efforts'], ['low', 'none'])

        # 4. A later clean reasoning-matrix retry cannot erase an earlier collapsed render
        retry_rm = self.tmp / 'retry_rm'
        cli('prepare', '--out', retry_rm, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=1, collapse_efforts=True).respond) as bad_srv:
            cli('reasoning-matrix', '--campaign', retry_rm, '--endpoint', bad_srv.url)
        with Loopback(V136FakeEngine(word_scale=1).respond) as good_srv:
            cli('reasoning-matrix', '--campaign', retry_rm, '--endpoint', good_srv.url)
        v_rrm = evidence.verdict(retry_rm, skew_s=0)
        self.assertEqual(v_rrm['gates']['reasoning_matrix']['status'], 'fail', v_rrm['gates']['reasoning_matrix'])


class CodingReplayAndConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release, self.profile, _, _ = fixtures(self.tmp)
        self.root = self.tmp / 'campaign'
        cli('prepare', '--out', self.root, '--release', self.release, '--profile', self.profile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_coding_replay_40k_262k_and_cold_prefill_concurrent_pass(self):
        fake = V136FakeEngine(word_scale=400, chunk_delay_s=0.02)
        with Loopback(fake.respond) as srv:
            cli('coding-replay', '--campaign', self.root, '--endpoint', srv.url, '--shape', '40k', '--turns', '3')
            cli('coding-replay', '--campaign', self.root, '--endpoint', srv.url, '--shape', '262k', '--turns', '2')
            cli('coding-replay', '--campaign', self.root, '--endpoint', srv.url, '--shape', 'cold-prefill-concurrent')
        v = evidence.verdict(self.root, skew_s=0)
        perf = v['performance']
        self.assertEqual(perf['status'], 'pass', perf)
        self.assertGreaterEqual(perf['coding_replay']['40k']['turns'][0]['measured_prompt_tokens'], 40_960)
        self.assertEqual(len(perf['coding_replay']['40k']['turns']), 3)
        self.assertGreaterEqual(perf['coding_replay']['262k']['turns'][0]['measured_prompt_tokens'], 262_144)
        cold_rep = perf['coding_replay']['cold-prefill-concurrent']
        self.assertFalse(cold_rep['decoder_starved'])
        self.assertTrue(cold_rep['overlapped'])
        self.assertTrue(cold_rep['launched_after_decoder_first_event'])
        self.assertGreater(cold_rep['active_decode_overlap_s'], 0.0)
        self.assertEqual(cold_rep['parts']['concurrent']['timing_basis'], 'sse_event_arrival')

    def test_cold_prefill_concurrent_detects_decoder_starvation(self):
        fake = V136FakeEngine(word_scale=400, stall_s=0.18, stall_on_substring='MOD_DEC_CONCURRENT_')
        with Loopback(fake.respond) as srv:
            cli('coding-replay', '--campaign', self.root, '--endpoint', srv.url,
                '--shape', 'cold-prefill-concurrent', '--starvation-ms', '80')
        v = evidence.verdict(self.root, skew_s=0)
        rep = v['performance']['coding_replay']['cold-prefill-concurrent']
        self.assertTrue(rep['decoder_starved'])
        self.assertGreater(rep['max_stall_ms'], 80.0)
        self.assertEqual(rep['status'], 'fail')
        self.assertEqual(v['production'], 'rejected')

    def test_cold_prefill_detects_ttft_starvation_and_per_sse_event_timings(self):
        # 1. Pre-first-token delay (server holding decoder before first SSE token) must fail via TTFT starvation
        ttft_root = self.tmp / 'ttft_starved'
        cli('prepare', '--out', ttft_root, '--release', self.release, '--profile', self.profile)
        fake_ttft = V136FakeEngine(word_scale=400, first_token_delay_s=0.14, chunk_delay_s=0.02,
                                   stall_on_substring='MOD_DEC_CONCURRENT_')
        with Loopback(fake_ttft.respond) as srv:
            cli('coding-replay', '--campaign', ttft_root, '--endpoint', srv.url,
                '--shape', 'cold-prefill-concurrent', '--starvation-ms', '60')
        v_ttft = evidence.verdict(ttft_root, skew_s=0)
        rep = v_ttft['performance']['coding_replay']['cold-prefill-concurrent']
        self.assertTrue(rep['decoder_starved'])
        self.assertTrue(rep['parts']['concurrent']['ttft_starved'])
        self.assertEqual(rep['status'], 'fail')

        # 2. Multiple SSE events packed in a single socket write record one timestamp per content SSE event
        def packed_sse(h):
            h.send_response(200)
            h.send_header('Content-Type', 'text/event-stream')
            h.end_headers()
            payload = b''.join(sse(item) for item in (
                event({'role': 'assistant'}),
                event({'content': 'a'}),
                event({'content': 'b'}),
                event({'content': 'c'}),
                event({}, 'stop'),
                {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}},
                b'[DONE]',
            ))
            h.wfile.write(payload)
            h.wfile.flush()

        with Loopback(packed_sse) as srv:
            res = transport.exchange(srv.url, 'POST', '/v1/chat/completions', b'{}', 5.0, stream=True)
        self.assertTrue(res['complete'])
        self.assertEqual(len(res['event_times_s']), 3)
        self.assertEqual(res['stream_profile']['timing_basis'], 'sse_event_arrival')
        self.assertIsNotNone(res['first_event_mono_s'])
        self.assertIsNotNone(res['last_event_mono_s'])

        # 3. Offline cold-prefill evaluator rejects missing first-event timestamps, early launch, and post-event teardown overlap
        base_ok = self.tmp / 'cold_overlap_unit'
        cli('prepare', '--out', base_ok, '--release', self.release, '--profile', self.profile)
        with Loopback(V136FakeEngine(word_scale=400, chunk_delay_s=0.02).respond) as srv:
            cli('coding-replay', '--campaign', base_ok, '--endpoint', srv.url, '--shape', 'cold-prefill-concurrent')
        c = evidence.Campaign(base_ok)
        rec = next(r for r in c.receipts() if r.get('kind') == 'coding-replay')
        measures = {}
        for r in c.receipts():
            if r.get('kind') == 'tokenize' and r.get('purpose') == 'measure' and 'request' in r.get('sidecars', {}):
                measures[('measure', transport.sha256(c.sidecar(r, 'request')))] = evidence.tokenize_result(
                    c.sidecar(r, 'response')
                )

        # (a) Prefill launched before decoder first event
        r_early = json.loads(json.dumps(rec))
        r_early['launched_after_decoder_first_event'] = False
        ev_early = coding._eval_cold_concurrent(c, c.plan, r_early, measures, starvation_ms=2000.0)
        self.assertEqual(ev_early['status'], 'fail')

        # (b) Missing first_event_mono_s cannot pass overlap check
        r_missing = json.loads(json.dumps(rec))
        r_missing['concurrent_decoder']['first_event_mono_s'] = None
        ev_missing = coding._eval_cold_concurrent(c, c.plan, r_missing, measures, starvation_ms=2000.0)
        self.assertEqual(ev_missing['status'], 'fail')
        self.assertFalse(ev_missing['overlapped'])

        # (c) Overlap only during post-last-event teardown (after last_event_mono_s) must fail
        r_teardown = json.loads(json.dumps(rec))
        r_teardown['concurrent_decoder']['first_event_mono_s'] = 100.0
        r_teardown['concurrent_decoder']['last_event_mono_s'] = 101.0
        r_teardown['concurrent_decoder']['mono_end_s'] = 105.0
        r_teardown['cold_prefill']['mono_start_s'] = 102.0
        r_teardown['cold_prefill']['mono_end_s'] = 104.0
        ev_teardown = coding._eval_cold_concurrent(c, c.plan, r_teardown, measures, starvation_ms=2000.0)
        self.assertEqual(ev_teardown['status'], 'fail')
        self.assertFalse(ev_teardown['overlapped'])

    def test_concurrency_1_2_3_4_pi_buckets_and_starvation_reporting(self):
        with Loopback(V136FakeEngine(word_scale=1).respond) as srv:
            cli('concurrency', '--campaign', self.root, '--endpoint', srv.url, '--levels', '1,2,3,4', '--chunks', '1')
        v = evidence.verdict(self.root, skew_s=0)
        conc = v['performance']['concurrency']
        self.assertEqual(conc['status'], 'pass', conc)
        self.assertEqual(set(conc['levels']), {'1', '2', '3', '4'})
        self.assertEqual(set(conc['pi_buckets']), {'C1', 'C2-C3', 'C4'})
        self.assertEqual(conc['pi_buckets']['C2-C3']['levels'], [2, 3])
        self.assertTrue(all(b['all_passed'] for b in conc['pi_buckets'].values()))

        # Inject starvation on one of the 4 concurrent streams
        starved_root = self.tmp / 'starved-c4'
        cli('prepare', '--out', starved_root, '--release', self.release, '--profile', self.profile)
        stalled_fake = V136FakeEngine(word_scale=1, stall_s=0.15, stall_on_substring='MOD_CONC_C4_S2_')
        with Loopback(stalled_fake.respond) as srv:
            cli('concurrency', '--campaign', starved_root, '--endpoint', srv.url,
                '--levels', '1,2,3,4', '--chunks', '1', '--starvation-ms', '60')
        vs = evidence.verdict(starved_root, skew_s=0)
        cs = vs['performance']['concurrency']
        self.assertEqual(cs['status'], 'fail')
        self.assertEqual(cs['levels']['4']['starved_streams'], 1)
        self.assertFalse(cs['pi_buckets']['C4']['all_passed'])
        self.assertGreater(cs['pi_buckets']['C4']['worst_itl_max_ms'], 60.0)


if __name__ == '__main__':
    unittest.main()
