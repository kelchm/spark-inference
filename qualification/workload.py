"""Deterministic qualification sessions: planted facts, color-coded images and a Pi-shaped tool exchange.

Everything here is a pure function of the plan, so the offline verdict can regenerate each user message and its expected
answers instead of trusting what a receipt says it sent. Sizes are counted in chunks; tokens are only ever measured by
the server's /tokenize during an approved run.
"""
import base64
import hashlib
import json
import re
import struct
import zlib

CONTEXT = 1_048_576
REPLY_RESERVATION = 32_768
CEILING = CONTEXT - REPLY_RESERVATION  # the longest prompt the 1M profile admits
SESSIONS = ('ledger', 'evictor', 'image', 'image-fork', 'pi')
BASE_STEPS = {'ledger': ('first', 'grow-1', 'grow-2', 'grow-3', 'grow-4'), 'evictor': ('first',),
              'image': ('first',), 'image-fork': ('first',), 'pi': ('call', 'result')}
CONTINUABLE = ('ledger', 'image', 'image-fork', 'pi')

WORDS = ('amber', 'basin', 'cedar', 'delta', 'ember', 'fjord', 'grove', 'harbor', 'inlet', 'juniper', 'kettle',
         'lantern', 'meadow', 'nectar', 'orchard', 'pebble', 'quarry', 'ridge', 'saddle', 'thicket', 'upland',
         'valley', 'willow', 'yarrow', 'zephyr', 'anchor', 'bramble', 'canopy', 'dune', 'estuary', 'fallow', 'glacier',
         'heath', 'island', 'jetty', 'knoll', 'lagoon', 'marsh', 'notch', 'oxbow', 'prairie', 'quay', 'reef', 'shoal',
         'tundra', 'vale', 'wharf', 'bluff', 'cove', 'drift', 'the', 'a', 'of', 'near', 'beyond', 'under', 'quiet',
         'wide', 'narrow', 'old', 'early', 'late', 'north', 'south', 'east', 'west', 'surveyors', 'noted', 'measured',
         'crossed', 'mapped', 'walked', 'charted', 'recorded', 'and', 'with', 'along', 'past')
CODEWORDS = ('KESTREL', 'MARLIN', 'OSPREY', 'PLOVER', 'HERON', 'CORMORANT', 'TANAGER', 'WARBLER', 'BITTERN', 'GANNET',
             'CURLEW', 'DUNLIN', 'EGRET', 'FULMAR', 'GROUSE', 'JACANA', 'LAPWING', 'MERLIN', 'PIPIT', 'SHRIKE')
PALETTE = {'red': (220, 20, 20), 'green': (20, 160, 40), 'blue': (20, 60, 220), 'yellow': (240, 220, 20),
           'black': (0, 0, 0), 'white': (255, 255, 255), 'orange': (245, 130, 0), 'purple': (130, 40, 170)}
QUADRANTS = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
IMAGE_SIZE = 224
ANSWER = re.compile(r'^\s*\**Q(\d+)\**\s*[=:]\s*(.+?)\s*$', re.M)
SYSTEM = {
    'ledger': 'You are a careful records clerk. Answer only from the records in this conversation.',
    'evictor': 'You are a careful records clerk. Answer only from the records in this conversation.',
    'image': 'You describe images precisely. Name colors using only these words: ' + ', '.join(PALETTE) + '.',
    'pi': 'You are a coding agent working in a terminal. Use the provided tools when a task needs them.',
}
SYSTEM['image-fork'] = SYSTEM['image']
PI_TOOLS = [{'type': 'function', 'function': {
    'name': 'read_ledger', 'description': 'Read one ledger record by its key and return the stored value.',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string', 'description': 'The record key.'}},
                   'required': ['key']}}}]
FORMAT = 'Answer with exactly one line per question in the form Qn=<answer> and nothing else.'


class Det:
    """A byte stream from SHA-256 of (seed, label, counter): stable across Python versions and platforms."""

    def __init__(self, seed, label):
        self.seed, self.label, self.counter, self.pool = seed, label, 0, b''

    def below(self, n):
        while len(self.pool) < 4:
            self.pool += hashlib.sha256(f'{self.seed}/{self.label}/{self.counter}'.encode()).digest()
            self.counter += 1
        value, self.pool = int.from_bytes(self.pool[:4], 'big'), self.pool[4:]
        return value % n


def plan(seed, model, reasoning_effort, max_tokens, *, margin=24_576, near_floor=980_000, evictor_target=600_000,
         ledger_targets=(300_000, 450_000, 600_000, 800_000)):
    if not 0 < max_tokens <= REPLY_RESERVATION:
        raise ValueError(f'max_tokens must be within the {REPLY_RESERVATION}-token reply reservation')
    if not near_floor <= CEILING - margin or ledger_targets[0] < 300_000:
        raise ValueError('targets must leave the margin under the ceiling and start at 300k or more')
    det = Det(seed, 'images')
    images = []
    for _ in range(4):
        names = list(PALETTE)
        images.append([names.pop(det.below(len(names))) for _ in QUADRANTS])
    det = Det(seed, 'pi')
    key = f'LEDGER-{det.below(9000) + 1000}'
    value = f'{CODEWORDS[det.below(len(CODEWORDS))]}-{det.below(90000) + 10000}'
    return {'version': 1, 'seed': seed, 'model': model, 'reasoning_effort': reasoning_effort, 'max_tokens': max_tokens,
            'context': CONTEXT, 'reply_reservation': REPLY_RESERVATION, 'ceiling': CEILING, 'margin': margin,
            'near_floor': near_floor, 'ledger_targets': list(ledger_targets), 'evictor_target': evictor_target,
            'images': images, 'pi': {'key': key, 'value': value}}


# Ledger and evictor: chunks of filler prose, each holding one planted fact.

def fact(seed, session, index):
    det = Det(seed, f'{session}/fact/{index}')
    letters = ''.join(chr(65 + (index // 26 ** k) % 26) for k in (2, 1, 0))  # unique per index by construction
    vault = f'{session[0].upper()}{index:04d}'
    return vault, f'{CODEWORDS[det.below(len(CODEWORDS))]}-{det.below(9000) + 1000}-{letters}'


def chunk(seed, session, index):
    det = Det(seed, f'{session}/chunk/{index}')
    sentences = []
    for _ in range(64):
        words = [WORDS[det.below(len(WORDS))] for _ in range(10 + det.below(8))]
        sentences.append(' '.join(words).capitalize() + '.')
    vault, code = fact(seed, session, index)
    sentences.insert(det.below(len(sentences) + 1), f'Registry note: the archive code for vault {vault} is {code}.')
    return f'Section {session}-{index:04d}.\n' + ' '.join(sentences)


def fact_questions(seed, session, indices):
    questions, expected = [], {}
    for n, index in enumerate(dict.fromkeys(indices), 1):
        vault, code = fact(seed, session, index)
        questions.append(f'Q{n}: What is the archive code for vault {vault}?')
        expected[str(n)] = code
    return questions, expected


def ledger_indices(step, start, count, total):
    """The facts a step asks about: the session's first, and the middle and end of what this step added."""
    if step.startswith('continue-'):
        k = int(step.split('-', 1)[1])
        return [0, (total - 1) * (k % 3 + 1) // 4, total - 1]
    return [0, start + count // 2, start + count - 1]


def ledger_user(p, session, step, start, count):
    total = start + count
    questions, expected = fact_questions(p['seed'], session, ledger_indices(step, start, count, total))
    records = '\n\n'.join(chunk(p['seed'], session, i) for i in range(start, start + count))
    text = (records + '\n\n' if records else '') + 'Questions:\n' + '\n'.join(questions) + '\n' + FORMAT
    return {'role': 'user', 'content': text}, expected


# Images: 2x2 quadrants of named palette colors, so the expected answer is known mechanically.

def png(colors, size=IMAGE_SIZE):
    half = size // 2
    rows = []
    for y in range(size):
        left, right = (colors[0], colors[1]) if y < half else (colors[2], colors[3])
        rows.append(b'\x00' + bytes(PALETTE[left]) * half + bytes(PALETTE[right]) * (size - half))

    def part(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    return (b'\x89PNG\r\n\x1a\n' + part(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)) +
            part(b'IDAT', zlib.compress(b''.join(rows), 9)) + part(b'IEND', b''))


def quadrant_colors(data):
    """The four quadrant colors of an 8-bit RGB PNG with unfiltered rows, read back from its pixels."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG')
    at, idat, header = 8, b'', None
    while at < len(data):
        length, kind = struct.unpack('>I4s', data[at:at + 8])
        body = data[at + 8:at + 8 + length]
        if struct.unpack('>I', data[at + 8 + length:at + 12 + length])[0] != zlib.crc32(kind + body):
            raise ValueError('PNG CRC mismatch')
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', body)
        elif kind == b'IDAT':
            idat += body
        at += 12 + length
    if header is None or header[2:] != (8, 2, 0, 0, 0):
        raise ValueError('only 8-bit RGB non-interlaced PNGs are decoded')
    width, height = header[:2]
    pixels = zlib.decompress(idat)
    stride = 1 + 3 * width
    if len(pixels) != stride * height or any(pixels[y * stride] != 0 for y in range(height)):
        raise ValueError('unexpected PNG row filters or size')
    names = {rgb: name for name, rgb in PALETTE.items()}
    found = []
    for y0, x0 in ((0, 0), (0, width // 2), (height // 2, 0), (height // 2, width // 2)):
        colors = {pixels[y * stride + 1 + 3 * x:y * stride + 4 + 3 * x]
                  for y in range(y0, y0 + height // 2) for x in range(x0, x0 + width // 2)}
        if len(colors) != 1 or tuple(next(iter(colors))) not in names:
            raise ValueError('quadrant is not one palette color')
        found.append(names[tuple(next(iter(colors)))])
    return found


def image_parts(p):
    parts = []
    for n, colors in enumerate(p['images'], 1):
        url = 'data:image/png;base64,' + base64.b64encode(png(colors)).decode()
        parts += [{'type': 'text', 'text': f'Image {n}:'}, {'type': 'image_url', 'image_url': {'url': url}}]
    return parts


def image_plan(p, session):
    # Identical text, dimensions and positions, different pixels: expose image-identity aliasing.
    return dict(p, images=[colors[1:] + colors[:1] for colors in p['images']]) if session == 'image-fork' else p


def image_user(p, step):
    k = 0 if step == 'first' else int(step.split('-', 1)[1])
    questions, expected = [], {}
    for n, colors in enumerate(p['images'], 1):
        q = (n + k) % 4
        questions.append(f'Q{n}: In image {n}, what color is the {QUADRANTS[q]} quadrant?')
        expected[str(n)] = colors[q]
    text = 'Questions:\n' + '\n'.join(questions) + '\n' + FORMAT
    if step == 'first':
        return {'role': 'user', 'content': image_parts(p) + [{'type': 'text', 'text': text}]}, expected
    return {'role': 'user', 'content': text}, expected


# Pi: a tool call whose result carries the answer, replayed with its reasoning and call IDs exactly.

def pi_user(p, step):
    key, value = p['pi']['key'], p['pi']['value']
    if step == 'call':
        return {'role': 'user', 'content': f'Look up ledger key {key} with the read_ledger tool. After you receive the '
                                           f'result, reply with exactly one line Q1=<value>.'}, {'1': value}
    return {'role': 'user', 'content': f'Q1: What value did read_ledger return for key {key}?\nQ2: Which key did you '
                                       f'read?\n{FORMAT}'}, {'1': value, '2': key}


def tool_results(p, message):
    """The tool messages answering each call, keyed by the call's original id."""
    results = []
    for call in message.get('tool_calls') or []:
        try:
            arguments = json.loads(call['function']['arguments'])
        except ValueError:
            arguments = None
        ok = (call['function']['name'] == 'read_ledger' and isinstance(arguments, dict) and
              arguments.get('key') == p['pi']['key'])
        results.append({'role': 'tool', 'tool_call_id': call['id'],
                        'content': p['pi']['value'] if ok else 'error: unknown tool or key'})
    return results


def replay(message):
    """The assistant message as it goes back to the server: content, reasoning and calls exactly as received."""
    out = {'role': 'assistant', 'content': message['content']}
    if message.get('reasoning_content'):
        out['reasoning_content'] = message['reasoning_content']
    if message.get('tool_calls'):
        out['tool_calls'] = [{'id': c['id'], 'type': 'function',
                              'function': {'name': c['function']['name'], 'arguments': c['function']['arguments']}}
                             for c in message['tool_calls']]
    return out


def chat_body(p, session, messages):
    body = {'model': p['model'], 'messages': messages, 'stream': True, 'stream_options': {'include_usage': True},
            'max_tokens': p['max_tokens']}
    if p['reasoning_effort']:
        body['reasoning_effort'] = p['reasoning_effort']
    if session == 'pi':
        body['tools'] = PI_TOOLS
    return body


def tokenize_body(body, add_generation_prompt=True):
    """The /tokenize request that renders exactly what `body` would run (same messages, tools and thinking)."""
    out = {key: body[key] for key in ('model', 'messages', 'tools', 'tool_choice', 'reasoning_effort',
                                      'chat_template_kwargs') if key in body}
    out['add_generation_prompt'] = add_generation_prompt
    return out


def next_turn(p, session, step, history, chunks=None):
    """The next request's body, expected answers and metadata.

    `history` is the session's earlier completed turns as (request body, parsed assistant message); `chunks` is the new
    chunk count for ledger and evictor base steps (fitted by /tokenize during the run).
    """
    if history:
        last_body, last_message = history[-1]
        messages = list(last_body['messages']) + [replay(last_message)]
        if session == 'pi' and step == 'result':
            messages += tool_results(p, last_message)
    else:
        messages = [{'role': 'system', 'content': SYSTEM[session]}]
    meta = {'session': session, 'step': step}
    if session in ('ledger', 'evictor'):
        start = meta_start(history)
        count = 0 if step.startswith('continue-') else chunks
        if count is None or (count < 1 and not step.startswith('continue-')):
            raise ValueError('a base ledger step needs a positive chunk count')
        user, expected = ledger_user(p, session, step, start, count)
        meta.update(chunk_start=start, chunks=count)
        messages.append(user)
    elif session in ('image', 'image-fork'):
        user, expected = image_user(image_plan(p, session), step)
        messages.append(user)
    elif step == 'result':
        expected = {'1': p['pi']['value']}
    else:
        user, expected = pi_user(p, step)
        messages.append(user)
    return chat_body(p, session, messages), expected, meta


def meta_start(history):
    """The next chunk index: one past every chunk already sent, counted from the session's user messages."""
    return sum(len(re.findall(r'^Section \w+-\d{4}\.$', m['content'], re.M))
               for m in (history[-1][0]['messages'] if history else []) if m['role'] == 'user' and
               isinstance(m['content'], str))


def answers(content):
    found = {}
    for number, value in ANSWER.findall(content):
        if number in found:
            return None  # a second answer to one question is ambiguous
        found[number] = value.strip().strip('`*').strip()
    return found


def correct(expected, content):
    found = answers(content)
    return found is not None and all(found.get(k, '').casefold() == v.casefold() for k, v in expected.items())


def fit(count, target, limit, *, reach=True, max_calls=24, hi_cap=8192):
    """The chunk count for a target, from a monotone count(n) of measured prompt tokens.

    reach: the smallest n >= 1 with count(n) >= target, refused if that exceeds limit.
    fill: the largest n >= 1 with count(n) <= limit.
    Returns (n, measured tokens at n, every probe as (n, tokens)).
    """
    memo = {}

    def c(n):
        if n not in memo:
            if len(memo) >= max_calls:
                raise RuntimeError(f'fit used {max_calls} /tokenize calls without converging')
            memo[n] = count(n)
        return memo[n]

    over = (lambda n: c(n) >= target) if reach else (lambda n: c(n) > limit)
    if c(1) > limit:
        raise ValueError(f'one chunk already exceeds the {limit}-token limit')
    if reach and over(1):
        hi = 1
    else:
        per = max(1.0, (c(9) - c(1)) / 8)
        hi = min(hi_cap, max(2, 1 + int(((target if reach else limit) - c(1)) / per)))
        while not over(hi):
            if hi >= hi_cap:
                raise ValueError(f'{hi_cap} chunks do not reach the target')
            hi = min(hi_cap, hi * 2)
        lo = 1
        while hi - lo > 1:  # invariant: not over(lo), over(hi)
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if over(mid) else (mid, hi)
    n = hi if reach else hi - 1
    if reach and c(n) > limit:
        raise ValueError(f'reaching {target} tokens takes {c(n)}, over the {limit}-token limit')
    return n, c(n), sorted(memo.items())
