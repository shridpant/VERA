"""Bounding boxes from the answering model itself, read only from what it generates.

Prompts follow the InternVL2 model card and grounding pilot WI-01: one image per call, 100
new tokens, greedy. InternVL2 writes `<ref> phrase </ref> <box> [[x1, y1, x2, y2]] </box>`
with coordinates on a 0-1000 grid of the image it saw. chat() drops those special tokens, so
callers parse the decode that keeps them. The first box is primary; when it covers at least
95 % of the image it is `full_frame`, not a localization (the same rule as the evidence
branch). No box in the text is `no_box`. Nothing is inferred or drawn that the model did not write.
"""
import re

GROUND_PROMPT = 'Please provide the bounding box coordinates of the region this sentence describes: <ref>{}</ref>'
NAME_PROMPT = ('In at most eight words, name the main anomalous object or event in these images. '
               'Reply with the phrase only.')
SCALE = 1000
FULL_FRAME_AREA = 0.95
NUMBER = r'-?\d+(?:\.\d+)?'


def tag_format(tagged):
    if '<box>' in tagged:
        return 'internvl_ref_box'
    if '<|box_start|>' in tagged:
        return 'qwen_box_start'
    return 'none'


def parse_boxes(tagged):
    """Boxes inside <box> tags on the 0-1000 grid, the primary box's area fraction, and a status."""
    result = {'boxes': [], 'status': 'no_box', 'area_fraction': None, 'tag_format': tag_format(tagged)}
    bodies = re.findall(r'<box>(.*?)(?:</box>|$)', tagged, re.S)  # A reply cut at the token limit has no </box>.
    groups = [group for body in bodies for group in re.findall(r'\[([^\[\]]*)\]', body)]
    for group in groups:
        values = [float(value) for value in re.findall(NUMBER, group)]
        if len(values) == 4 and min(values) >= 0 and max(values) <= SCALE and values[0] < values[2] and values[1] < values[3]:
            result['boxes'].append(values)
    if not result['boxes']:
        result['status'] = 'malformed' if bodies else 'no_box'
        return result
    x1, y1, x2, y2 = result['boxes'][0]
    result['area_fraction'] = round((x2 - x1) * (y2 - y1) / SCALE ** 2, 4)
    result['status'] = 'full_frame' if result['area_fraction'] >= FULL_FRAME_AREA else 'box'
    return result


def clean_phrase(text):
    """First line of the naming reply, without quotes or a trailing period; empty if nothing usable."""
    line = text.strip().splitlines()[0] if text.strip() else ''
    return line.strip(' "\'`*.').strip()[:120]


def box_images(backend, items, phrase, max_new_tokens, timeline, **fields):
    """Ground one phrase on each item's image; items carry image, slot, frame_index and image path."""
    rows = []
    for item in items:
        result = backend.ground(item['image'], phrase, max_new_tokens)
        row = {'slot': item['slot'], 'frame_index': item['frame_index'], 'image': item['path'], **result}
        timeline.emit('box', phrase=phrase, **fields, **row)
        rows.append(row)
    return rows


def box_window(backend, images, response, settings, timeline, frame_indices, paths, **fields):
    """For a window labelled 1: name the anomaly once (or use the configured phrase), box every image."""
    if settings['phrase']:
        named = {'prompt': None, 'raw_text': None, 'phrase': settings['phrase'], 'seconds': 0.0, 'source': 'config'}
    else:
        named = {**backend.name_anomaly(images, response), 'source': 'model'}
    timeline.emit('naming', **fields, **named)
    items = [{'image': image, 'slot': slot, 'frame_index': frame_indices[slot], 'path': paths[slot]}
             for slot, image in enumerate(images)]
    frames = box_images(backend, items, named['phrase'], settings['max_new_tokens'], timeline, **fields) if named['phrase'] else []
    return {**fields, 'mode': 'auto', 'phrase': named['phrase'], 'naming': named, 'frames': frames}
