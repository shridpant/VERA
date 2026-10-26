"""The package's box detector: one frame per call, a foundation VLM, category-free prompt (package plan, Module 1).

Boxes are on the 0-1000 grid of the frame. A box covering at least 95 % of the frame is `whole_image`: it is
kept in the record but never counted as a detection. Labels are the model's own words, mapped to handgun /
long gun / firearm. No confidence is read (none is native), so there is no AP.
"""
import json
import re
from .grounding import FULL_FRAME_AREA

SCALE = 1000
INTERNVL_PROMPT = ('Is there a firearm (a handgun or a long gun) in this image? If yes, give the bounding box of each one as '
                   '<ref>handgun</ref><box>[[x1, y1, x2, y2]]</box> or <ref>long gun</ref><box>[[x1, y1, x2, y2]]</box>. '
                   'If no, answer: none.')
INTERNVL_REC_PROMPT = 'Please provide the bounding box coordinates of the region this sentence describes: <ref>the firearm</ref>'
QWEN_PROMPT = ('Is there a firearm (a handgun or a long gun) in this image? If yes, output the bounding box of each one in JSON: '
               '[{"bbox_2d": [x1, y1, x2, y2], "label": "handgun"}], with label "handgun" or "long gun". If no, output [].')
NUMBER = r'-?\d+(?:\.\d+)?'


def label_of(text):
    t = (text or '').lower()
    if re.search(r'hand ?gun|pistol|revolver', t):
        return 'handgun'
    if re.search(r'long ?gun|rifle|machine ?gun|shotgun|carbine|submachine|assault', t):
        return 'long gun'
    return 'firearm'


def valid(values, scale=SCALE):
    return len(values) == 4 and min(values) >= 0 and max(values) <= scale and values[0] < values[2] and values[1] < values[3]


def parse_internvl(tagged):
    """Boxes from <ref>label</ref><box>[[...], ...]</box> (labels may be absent); 'none' answers give []."""
    found, malformed = [], 0
    for ref, body in re.findall(r'(?:<ref>(.*?)</ref>\s*)?<box>(.*?)(?:</box>|$)', tagged, re.S):
        for group in re.findall(r'\[([^\[\]]*)\]', body):
            values = [float(v) for v in re.findall(NUMBER, group)]
            if valid(values):
                found.append({'box': values, 'label': label_of(ref), 'raw_label': ref.strip()})
            else:
                malformed += 1
    return found, malformed


def parse_qwen(answer):
    """Boxes from a JSON list of {"bbox_2d": [...], "label": ...}; falls back to a regex over bbox_2d entries."""
    text = re.sub(r'```(?:json)?', '', answer)
    found, malformed = [], 0
    items = None
    for match in reversed(list(re.finditer(r'\[.*\]', text, re.S))):
        try:
            items = json.loads(match.group(0))
            break
        except ValueError:
            continue
    if isinstance(items, list) and all(isinstance(i, dict) for i in items):
        pairs = [(i.get('bbox_2d') or i.get('bbox'), i.get('label', '')) for i in items]
    else:
        pairs = []
        for chunk in re.findall(r'\{[^{}]*\}', text):
            box = re.search(r'"bbox(?:_2d)?"\s*:\s*\[([^\]]*)\]', chunk)
            label = re.search(r'"label"\s*:\s*"([^"]*)"', chunk)
            if box:
                pairs.append(([float(v) for v in re.findall(NUMBER, box.group(1))], label.group(1) if label else ''))
    for box, label in pairs:
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            malformed += 1
            continue
        if valid(values):
            found.append({'box': values, 'label': label_of(label), 'raw_label': str(label)})
        else:
            malformed += 1
    return found, malformed


def says_none(answer):
    return bool(re.fullmatch(r'\W*(none|no|\[\s*\])\W*', answer.strip(), re.I)) or answer.strip() == '[]'


def area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1) / SCALE ** 2


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, threshold):
    """Greedy one-to-one matching by IoU; whole-image predictions never match. Returns the matched count."""
    candidates = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) if area(p) < FULL_FRAME_AREA
                         for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, n = set(), set(), 0
    for value, i, j in candidates:
        if value < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j)
            n += 1
    return n


def score_frame(found, gts):
    preds = [f['box'] for f in found]
    return {'n_pred': len(preds), 'n_whole_image': sum(area(p) >= FULL_FRAME_AREA for p in preds), 'n_gt': len(gts),
            'tp50': match(preds, gts, 0.5), 'tp75': match(preds, gts, 0.75),
            'best_iou': max((iou(p, g) for p in preds for g in gts), default=None)}


def parse(family, result):
    """Record fields for one detector call (result from vlm.InternVL.chat / vlm.Qwen.chat)."""
    answer = result['answer']  # the text after any thinking trace: boxes inside the trace are not the answer
    found, malformed = parse_internvl(answer) if family == 'internvl' else parse_qwen(answer)
    status = ('truncated' if result['truncated'] else 'boxes' if found else
              'none' if says_none(answer) else 'malformed' if malformed else 'no_box')
    return {'found': found, 'malformed': malformed, 'status': status}
