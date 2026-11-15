"""Bounded HTTP for qualification probes.

One total deadline covers DNS, connect, request upload, headers and every read, so a server that trickles bytes cannot
stretch it. Nothing retries: a failed exchange is returned as it happened, with whatever bytes arrived, for the caller to
record. Streamed chat replies count as complete only with a `[DONE]` event and a stop or tool_calls finish.
"""
import hashlib
import http.client
import json
import socket
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

BODY_LIMIT = 32 * 1024 * 1024  # the engine's http_body.limit at e52eb42b


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def strict_json(payload):
    """JSON that refuses duplicate keys and NaN/Infinity, which a lenient parser would quietly accept."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'non-finite JSON constant {value}')

    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode('utf-8')
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


class Stream:
    """An OpenAI chat completion SSE stream, parsed strictly as it arrives."""

    def __init__(self):
        self.buffer = b''
        self.data = []
        self.done = False
        self.finish = None
        self.usage = None
        self.message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
        self.calls = {}

    def feed(self, chunk):
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            line = line.removesuffix(b'\r')
            if line == b'':
                if self.data:
                    self.event(b'\n'.join(self.data))
                    self.data = []
            elif line.startswith(b'data:'):
                self.data.append(line[5:].removeprefix(b' '))
            elif line.startswith((b':', b'event:', b'id:', b'retry:')):
                continue
            else:
                raise ValueError('malformed SSE line')

    def event(self, payload):
        if self.done:
            raise ValueError('event after [DONE]')
        if payload == b'[DONE]':
            self.done = True
            return
        data = strict_json(payload)
        if not isinstance(data, dict) or 'error' in data:
            raise ValueError(f'API error or malformed event: {data!r}'[:500])
        if data.get('usage') is not None:
            if self.usage is not None and self.usage != data['usage']:
                raise ValueError('conflicting usage')
            self.usage = data['usage']
        choices = data.get('choices')
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError('expected a choices array with at most one choice')
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get('index')) is not int or choice['index'] != 0:
                raise ValueError('unexpected choice index')
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                raise ValueError('invalid delta')
            if delta.get('role', 'assistant') != 'assistant':
                raise ValueError('unexpected streamed role')
            late = any(v for k, v in delta.items() if k != 'role') or choice.get('finish_reason')
            if self.finish is not None and late:
                raise ValueError('content or a second finish after finish')
            reasoning = delta.get('reasoning_content')
            if reasoning is not None and delta.get('reasoning') is not None and delta['reasoning'] != reasoning:
                raise ValueError('conflicting reasoning aliases')
            for name, value in (('content', delta.get('content')),
                                ('reasoning_content', reasoning if reasoning is not None else delta.get('reasoning'))):
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError('non-text delta')
                    self.message[name] += value
            for part in delta.get('tool_calls') or []:
                index = part.get('index') if isinstance(part, dict) else None
                if type(index) is not int or not 0 <= index < 16:
                    raise ValueError('invalid tool call index')
                if part.get('type', 'function') != 'function':
                    raise ValueError('invalid tool call type')
                call = self.calls.setdefault(index, {'id': '', 'type': 'function',
                                                     'function': {'name': '', 'arguments': ''}})
                call['id'] += part.get('id') or ''
                function = part.get('function') or {}
                for name in ('name', 'arguments'):
                    value = function.get(name)
                    if value is not None and not isinstance(value, str):
                        raise ValueError('non-text tool call field')
                    call['function'][name] += value or ''
            if choice.get('finish_reason') is not None:
                self.finish = choice['finish_reason']

    def result(self):
        """The completed reply; raises unless the stream ended affirmatively."""
        if not self.done or self.buffer or self.data:
            raise ValueError('incomplete stream: no [DONE], or bytes after the last event')
        if self.finish not in ('stop', 'tool_calls'):
            raise ValueError(f'finish_reason {self.finish!r} is not stop or tool_calls')
        usage = self.usage
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0
                                              for k in ('prompt_tokens', 'completion_tokens')):
            raise ValueError('positive prompt and completion usage required')
        if 'total_tokens' in usage and usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
            raise ValueError('inconsistent total_tokens')
        details = usage.get('prompt_tokens_details')
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
        if cached is not None and (type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']):
            raise ValueError('invalid cached_tokens')
        message = dict(self.message)
        if self.calls:
            if sorted(self.calls) != list(range(len(self.calls))):
                raise ValueError('sparse tool call indices')
            calls = [self.calls[i] for i in sorted(self.calls)]
            if any(not c['id'] or not c['function']['name'] for c in calls):
                raise ValueError('tool call without id or name')
            message['tool_calls'] = calls
        if (self.finish == 'tool_calls') != bool(self.calls):
            raise ValueError('finish_reason and tool calls disagree')
        return {'message': message, 'finish_reason': self.finish, 'usage': usage, 'cached_tokens': cached}


def parse_sse(raw):
    """Re-parse a recorded stream from its wire bytes (offline)."""
    stream = Stream()
    stream.feed(raw)
    return stream.result()


def base_url(url):
    """An explicit http(s)://host:port base, optionally ending in /v1; no credentials, query or fragment."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or
            parsed.query or parsed.fragment):
        raise ValueError('endpoint must be an explicit http(s) base URL without credentials, query or fragment')
    path = parsed.path.rstrip('/')
    if path.endswith('/v1'):
        path = path[:-3]
    return parsed, path


def exchange(url, method, path, body, deadline_s, *, stream=False, max_bytes=1 << 20, api_key=None):
    """One request under one total deadline. Never raises for transport or protocol failures; never retries.

    Returns the bytes received (bounded by max_bytes), status, content type, Date header, timings and an error string.
    For streams, reading stops at [DONE]; `complete` is true only for a 200 of the expected type that ended
    affirmatively (SSE: [DONE] after a stop/tool_calls finish; JSON: the whole body, parsed strictly).
    """
    parsed, prefix = base_url(url)
    if body is not None and len(body) > BODY_LIMIT:
        raise ValueError(f'request body of {len(body)} bytes exceeds the engine limit of {BODY_LIMIT}')
    started_at, started = now(), time.monotonic()
    state = {'raw': bytearray(), 'error': None, 'status': None, 'content_type': None, 'date': None,
             'socket': None, 'complete': False, 'timings': {}}
    lock = threading.Lock()
    cancelled = threading.Event()

    def mark(name):
        with lock:
            state['timings'].setdefault(name, round(time.monotonic() - started, 6))

    def worker():
        cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
        connection = cls(parsed.hostname, parsed.port, timeout=deadline_s + 5)  # the total deadline fires first
        try:
            connection.connect()
            with lock:
                state['socket'] = connection.sock
            mark('connected_s')
            if cancelled.is_set():
                return
            headers = {'Accept': 'text/event-stream' if stream else 'application/json'}
            if body is not None:
                headers['Content-Type'] = 'application/json'
            if api_key:
                headers['Authorization'] = 'Bearer ' + api_key
            connection.request(method, prefix + path, body=body, headers=headers)
            response = connection.getresponse()
            mark('headers_s')
            kind = (response.getheader('Content-Type') or '').split(';')[0].strip()
            with lock:
                state['status'], state['content_type'] = response.status, kind
                state['date'] = response.getheader('Date')
            parser = Stream() if stream and response.status == 200 and kind == 'text/event-stream' else None
            while not cancelled.is_set():
                chunk = response.read1(65536)
                if not chunk:
                    break
                mark('first_byte_s')
                with lock:
                    if len(state['raw']) + len(chunk) > max_bytes:
                        raise ValueError(f'response exceeds the {max_bytes} byte bound')
                    state['raw'].extend(chunk)
                if parser:
                    parser.feed(chunk)
                    if parser.message['content'] or parser.message['reasoning_content']:
                        mark('first_token_s')
                    if parser.done:
                        break
            mark('end_s')
            if response.status != 200:
                raise ValueError(f'HTTP {response.status}')
            if stream:
                if parser is None:
                    raise ValueError(f'expected text/event-stream, got {kind!r}')
                parser.result()
            else:
                if kind != 'application/json':
                    raise ValueError(f'expected application/json, got {kind!r}')
                length = response.getheader('Content-Length')
                if length is not None and int(length) != len(state['raw']):
                    raise ValueError('body shorter than Content-Length')
                strict_json(bytes(state['raw']))
            with lock:
                state['complete'] = not cancelled.is_set()
        except Exception as error:  # recorded, never retried
            with lock:
                state['error'] = f'{type(error).__name__}: {error}'
        finally:
            connection.close()

    thread = threading.Thread(target=worker, name='qualification-http', daemon=True)
    thread.start()
    thread.join(max(0.0, deadline_s - (time.monotonic() - started)))
    timed_out = thread.is_alive() or time.monotonic() - started >= deadline_s
    if timed_out:
        cancelled.set()
        with lock:
            sock = state['socket']
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        thread.join(0.2)
    with lock:
        raw = bytes(state['raw'])
        return {'started_at': started_at, 'ended_at': now(), 'elapsed_s': round(time.monotonic() - started, 6),
                'deadline_s': deadline_s, 'deadline_exceeded': timed_out, 'http_status': state['status'],
                'content_type': state['content_type'], 'server_date': state['date'], 'timings': dict(state['timings']),
                'raw': raw, 'response_sha256': sha256(raw), 'complete': state['complete'] and not timed_out,
                'error': 'DeadlineExceeded: total request deadline' if timed_out else state['error']}
