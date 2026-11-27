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

    def __init__(self, on_content_event=None):
        self.buffer = b''
        self.data = []
        self.done = False
        self.finish = None
        self.usage = None
        self.message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
        self.calls = {}
        self.on_content_event = on_content_event

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
            emitted = False
            for name, value in (('content', delta.get('content')),
                                ('reasoning_content', reasoning if reasoning is not None else delta.get('reasoning'))):
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError('non-text delta')
                    if value:
                        emitted = True
                    self.message[name] += value
            if emitted and self.on_content_event is not None:
                self.on_content_event()
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
        if cached is None and 'cached_tokens' in usage:
            cached = usage.get('cached_tokens')
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


def percentile(values, pct):
    """Nearest-rank percentile over a non-empty numeric sequence."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((pct / 100.0) * len(ordered) + 0.999999) - 1))
    return round(float(ordered[index]), 3)


def itl_summary(event_times_s, *, starvation_ms=2000.0, max_gap_override_ms=None):
    """Compute SSE event-arrival inter-event latency tail stats and starvation counts.

    Client-side timings reflect SSE event arrival instants (not internal GPU step timestamps); pair with engine
    telemetry for per-token device latency.
    """
    times = list(event_times_s or ())
    if not times:
        return {
            'timing_basis': 'sse_event_arrival',
            'token_events': 0,
            'ttft_s': None,
            'decode_s': 0.0,
            'itl_ms': {'count': 0, 'p50': None, 'p95': None, 'p99': None, 'max': None},
            'starvation_threshold_ms': float(starvation_ms),
            'starvation_events': 0,
            'max_stall_ms': 0.0,
        }
    gaps_ms = [round(max(0.0, (b - a) * 1000.0), 3) for a, b in zip(times, times[1:])]
    max_gap = round(max(gaps_ms), 3) if gaps_ms else 0.0
    if max_gap_override_ms is not None:
        max_gap = max(max_gap, round(float(max_gap_override_ms), 3))
    starved = sum(1 for gap in gaps_ms if gap > starvation_ms)
    if max_gap_override_ms is not None and max_gap_override_ms > starvation_ms and starved == 0:
        starved = 1
    return {
        'timing_basis': 'sse_event_arrival',
        'token_events': len(times),
        'ttft_s': round(times[0], 6),
        'decode_s': round(max(0.0, times[-1] - times[0]), 6),
        'itl_ms': {
            'count': len(gaps_ms),
            'p50': percentile(gaps_ms, 50),
            'p95': percentile(gaps_ms, 95),
            'p99': percentile(gaps_ms, 99),
            'max': max_gap,
        },
        'starvation_threshold_ms': float(starvation_ms),
        'starvation_events': starved,
        'max_stall_ms': max_gap,
    }


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


def exchange(url, method, path, body, deadline_s, *, stream=False, max_bytes=1 << 20, api_key=None,
             on_first_token=None):
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
             'socket': None, 'complete': False, 'timings': {}, 'event_times': [], 'last_event': None,
             'max_gap_ms': 0.0, 'first_event_mono': None, 'first_cb_fired': False}
    lock = threading.Lock()
    cancelled = threading.Event()

    def mark(name):
        with lock:
            state['timings'].setdefault(name, round(time.monotonic() - started, 6))

    def on_content_event():
        now_m = time.monotonic()
        elapsed = round(now_m - started, 6)
        fire_cb = False
        with lock:
            state['timings'].setdefault('first_token_s', elapsed)
            if state['first_event_mono'] is None:
                state['first_event_mono'] = now_m
            if state['last_event'] is not None:
                gap_ms = max(0.0, (now_m - state['last_event']) * 1000.0)
                if gap_ms > state['max_gap_ms']:
                    state['max_gap_ms'] = round(gap_ms, 3)
            state['last_event'] = now_m
            # Well above REPLY_RESERVATION (32,768) so normal qualification streams never drop late stalls.
            if len(state['event_times']) < 65536:
                state['event_times'].append(elapsed)
            if on_first_token is not None and not state['first_cb_fired']:
                state['first_cb_fired'] = True
                fire_cb = True
        if fire_cb:
            on_first_token()

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
            parser = Stream(on_content_event=on_content_event) if (
                stream and response.status == 200 and kind == 'text/event-stream'
            ) else None
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
    ended_mono = time.monotonic()
    with lock:
        raw = bytes(state['raw'])
        event_times = list(state['event_times'])
        return {'started_at': started_at, 'ended_at': now(), 'elapsed_s': round(ended_mono - started, 6),
                'mono_start_s': round(started, 6), 'mono_end_s': round(ended_mono, 6),
                'first_event_mono_s': round(state['first_event_mono'], 6) if state['first_event_mono'] is not None else None,
                'last_event_mono_s': round(state['last_event'], 6) if state['last_event'] is not None else None,
                'deadline_s': deadline_s, 'deadline_exceeded': timed_out, 'http_status': state['status'],
                'content_type': state['content_type'], 'server_date': state['date'], 'timings': dict(state['timings']),
                'event_times_s': event_times, 'token_times_s': event_times,
                'max_gap_ms_observed': state['max_gap_ms'],
                'stream_profile': itl_summary(event_times, max_gap_override_ms=state['max_gap_ms']),
                'raw': raw, 'response_sha256': sha256(raw), 'complete': state['complete'] and not timed_out,
                'error': 'DeadlineExceeded: total request deadline' if timed_out else state['error']}
