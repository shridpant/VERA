"""Strict readers for the study's answers: the final anomaly Output field and slot-indexed boxes.

Anomaly label: only the LAST 'Output:' field counts, and what follows it (without code fences,
brackets, quotes or a trailing period) must be exactly 0 or 1. Anything else is missing (None),
never 0. AG answers contain box digits, which corrupt the upstream rule ('0' anywhere after the
last 'Output'); that legacy label is kept for comparison only.

Boxes: one line per frame, 'FrameK: [[x1, y1, x2, y2], ...]' on a 0-1000 grid, '[]' for none
seen, 'unknown' for cannot assess. InternVL <ref>/<box> tags around a list are accepted and noted.
A slot is boxes | empty | unknown | malformed | missing | duplicate | truncated. A slot with any
invalid box is malformed as a whole. A box covering >= 95 % of the frame is flagged whole_image
and kept. Nothing is rescaled or repaired.
"""
import re

SCALE = 1000
WHOLE_IMAGE_AREA = 0.95
OUTPUT = re.compile(r'output\s*:', re.I)
LINE = re.compile(r'^[\s>*_`-]*frame\s*(\d+)\s*[*_`]*\s*[:：]\s*(.*?)\s*$', re.I)
NUMBER = r'-?\d+(?:\.\d+)?'
SPECIAL = re.compile(r'</?s>|<\|im_end\|>|<\|im_start\|>')
LIST = re.compile(r'^\[\s*(?:\[[^\[\]]*\]\s*,?\s*)+\]$')


def final_output_label(text):
    fields = list(OUTPUT.finditer(text))
    if not fields:
        return None
    tail = text[fields[-1].end():].replace('```', ' ').strip(' \t\r\n`[]()"\'*._')
    return int(tail) if tail in ('0', '1') else None


def legacy_label(text):
    """Upstream VERA's rule, verbatim in effect: 0 if any '0' follows the last 'Output', else 1."""
    return 0 if '0' in text.split('Output')[-1] else 1


def split_answer(text):
    """(text before the final Output field, the field) or (text, None) without one."""
    fields = list(OUTPUT.finditer(text))
    return (text[:fields[-1].start()], text[fields[-1].start():]) if fields else (text, None)


def _boxes(content):
    """(status, boxes, notes) for one slot's content."""
    notes = []
    if '<box>' in content or '<ref>' in content:
        notes.append('internvl_tags')
        content = re.sub(r'<ref>.*?</ref>', ' ', content)
        content = content.replace('<box>', ' ').replace('</box>', ' ')
    content = content.strip().strip('`*_ ').rstrip('.,;').strip()
    if content.strip('"\'').lower() == 'unknown':
        return 'unknown', [], notes
    if re.fullmatch(r'\[\s*\]', content):
        return 'empty', [], notes
    if re.fullmatch(r'\[[^\[\]]*\]', content):  # one flat box: unambiguous, noted
        notes.append('flat_box')
        content = '[' + content + ']'
    if not LIST.fullmatch(content):
        return 'malformed', [], notes
    boxes = []
    for group in re.findall(r'\[([^\[\]]*)\]', content):
        if re.sub(NUMBER, '', group).strip(' ,') != '':
            return 'malformed', [], notes
        values = [float(v) for v in re.findall(NUMBER, group)]
        if len(values) != 4 or min(values) < 0 or max(values) > SCALE or values[0] >= values[2] or values[1] >= values[3]:
            return 'malformed', [], notes
        boxes.append(values)
    return 'boxes', boxes, notes


def parse_slots(text, n_slots=8, truncated=False):
    clean = SPECIAL.sub(' ', text)
    lines = {}
    extra = []
    for line in clean.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        k = int(match.group(1))
        if 1 <= k <= n_slots:
            lines.setdefault(k - 1, []).append(match.group(2))
        else:
            extra.append(line.strip())
    slots = []
    for slot in range(n_slots):
        found = lines.get(slot, [])
        row = {'slot': slot, 'raw': found, 'boxes': [], 'whole_image': [], 'notes': []}
        if truncated:
            row['status'] = 'truncated'
        elif not found:
            row['status'] = 'missing'
        elif len(found) > 1:
            row['status'] = 'duplicate'
        else:
            row['status'], row['boxes'], row['notes'] = _boxes(found[0])
            row['whole_image'] = [(b[2] - b[0]) * (b[3] - b[1]) / SCALE ** 2 >= WHOLE_IMAGE_AREA for b in row['boxes']]
        slots.append(row)
    coords = [v for s in slots for b in s['boxes'] for v in b]
    before, field = split_answer(clean)
    return {'response_status': 'truncated' if truncated else ('ok' if lines else 'no_slots'),
            'slots': slots, 'extra_frame_lines': extra,
            'scale_suspect': bool(coords) and max(coords) <= 1,
            'heading': 'firearm boxes:' in clean.lower(),
            'slot_lines_after_output': sum(1 for line in (field or '').splitlines() if LINE.match(line))}
