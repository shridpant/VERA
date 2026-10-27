"""Parser functions copied unchanged from VERA c239dc0, vera_runner/verbalized.py."""
import re
FIELDS=("category","group","code")
LABELS={"category":r"category","group":r"action\s+group","code":r"action\s+code"}

def normalize(field, value, book):
    v = value.strip().strip('`[]()"\'*.,; \t').strip()
    if field == 'category':
        key = re.sub(r'[\s-]+', '_', v.lower())
        return {c.lower(): c for c in book['categories']}.get(key)
    if field == 'group':
        key = re.sub(r'[\s-]+', '_', v.lower())
        return key if key in book['groups'] else None
    key = v.upper().replace(' ', '')
    return key if key in book['codes'] else None

def parse_answer(text, book):
    """{'category', 'group', 'code'}: the last occurrence of each field, value up to the next field."""
    spans = []
    for field, label in LABELS.items():
        hits = list(re.finditer(label + r'\s*:', text, re.I))
        if hits:
            spans.append((hits[-1].start(), hits[-1].end(), field))
    spans.sort()
    out = {f: None for f in FIELDS}
    for n, (_, end, field) in enumerate(spans):
        stop = spans[n + 1][0] if n + 1 < len(spans) else len(text)
        out[field] = normalize(field, text[end:stop].replace('```', ' '), book)
    return out
