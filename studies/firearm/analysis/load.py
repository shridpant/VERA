"""Everything the tables and figures use, read only from saved artifacts.

videos   one row per scheduled video (inventory metadata, pilot flag, window count)
windows  one row per scheduled window: A and AG labels (strict, runner-strict, legacy), validity,
         truncation, and whether each condition has a complete record
slots    one row per scheduled window slot and grounding condition (G, AG): source frame, its
         audited status, ground-truth boxes (origin from the annotation policy, and the
         sensitivity origin), the slot's parse status and boxes
A unit without a complete record is kept as a row with complete=False; nothing is dropped.
"""
import csv
import json
from pathlib import Path
from vera_runner.config import path
from vera_runner.data import windows
from vera_runner.firearm_annotations import frame_boxes, frame_statuses, load_coco
from vera_runner.ledger import read_jsonl
from vera_runner.study import final_records, reconcile

PILOT_VIDEOS = ('Handgun__PAH7_C1_P1_V1_HB_2', 'Handgun__PAH9_C2_P3_V1_HB_2', 'Machine_Gun__PAW8_C1_P2_V1_HB_2',
                'Machine_Gun__PAW5_C2_P3_V1_HB_2', 'No_Gun__N8_C1_P2_V1_HB_4', 'No_Gun__N7_C2_P3_V1_HB_1')
CATEGORIES = ('Handgun', 'Machine_Gun', 'No_Gun')


class Data:
    pass


def label_fields(record):
    if record is None:
        return {'complete': False, 'label': None, 'strict_label': None, 'upstream_label': None, 'truncated': None,
                'text': None, 'tokens': None, 'seconds': None}
    return {'complete': True, 'label': record['label'], 'strict_label': record['strict_label'],
            'upstream_label': record['upstream_label'], 'truncated': record['truncated'], 'text': record['raw_response'],
            'tokens': record['generated_tokens'], 'seconds': record['seconds']}


def load(study, original_root, refined_dir=None):
    run_dir = path(study['execution']['run_dir'])
    reconcile(study, run_dir)
    final, spec = final_records(study, run_dir)
    inventory = {r['id']: r for r in json.loads(path(study['data']['inventory']).read_text())}
    policy = json.loads(path(study['data']['annotation_policy']).read_text())
    d = Data()
    d.study, d.spec, d.run_dir, d.policy, d.final = study, spec, run_dir, policy, final
    d.reconcile = json.loads((run_dir / 'reconcile.json').read_text())
    d.ledger = read_jsonl(run_dir / 'ledger.jsonl')[0]
    d.sessions = [json.loads(p.read_text()) for p in sorted((run_dir / 'sessions').glob('*/session.json'))]
    d.videos, d.windows, d.slots = [], [], []
    fps_nominal = study['runner']['sampling']['fps']
    for v in spec['videos']:
        row = inventory[v['id']]
        d.videos.append({'id': v['id'], 'category': row['category'], 'camera': row['camera'], 'camera_type': row['camera_type'],
                         'resolution': f"{row['width']}x{row['height']}", 'fps': round(row['fps'], 3), 'frames': row['frames_decoded'],
                         'duration_s': row['duration_s'], 'recording_group': row['recording_group'], 'subject': row['subject'],
                         'location': row['location'], 'action': row['action'], 'pilot': v['id'] in PILOT_VIDEOS,
                         'windows': len(v['anchors']), 'label_file': row['label_file']})
        coco = None
        if row['label_file']:
            coco = load_coco(Path(original_root) / Path(row['source_path']).parent / row['label_file'])
        gts = {o: frame_boxes(coco, row['width'], row['height'], o) if coco else {} for o in (policy['origin'], policy['sensitivity_origin'])}
        status = frame_statuses(row['category'], row['frames_decoded'], coco, policy['origin'])
        # The schedule is deterministic (data.windows); saved records must agree with it.
        schedule = {w['anchor']: w['frame_indices'] for w in windows(row['frames_decoded'], study['runner']['sampling'])}
        for anchor in v['anchors']:
            a, g, ag = (final[c].get((v['id'], anchor)) for c in ('A', 'G', 'AG'))
            frames = schedule[anchor]
            w = {'video': v['id'], 'category': row['category'], 'camera': row['camera'], 'anchor': anchor,
                 'time_s': anchor / row['fps'], 'nominal_time_s': anchor / fps_nominal, 'frame_indices': frames}
            for c, rec in (('A', a), ('AG', ag)):
                w.update({f'{c}_{k}': val for k, val in label_fields(rec).items()})
            w['G_complete'] = g is not None
            for rec in (a, g, ag):
                if rec is not None and rec['frame_indices'] != frames:
                    raise ValueError(f"{v['id']} {anchor}: record frames differ from the schedule")
            d.windows.append(w)
            for c, rec in (('G', g), ('AG', ag)):
                parsed = (rec or {}).get('grounding')
                for slot, frame in enumerate(frames):
                    s = parsed['slots'][slot] if parsed else None
                    d.slots.append({'video': v['id'], 'category': row['category'], 'camera': row['camera'], 'anchor': anchor,
                                    'slot': slot, 'frame': frame, 'condition': c, 'frame_status': status[frame],
                                    'gt': gts[policy['origin']].get(frame, []),
                                    'gt_sensitivity': gts[policy['sensitivity_origin']].get(frame, []),
                                    'complete': rec is not None,
                                    'status': s['status'] if s else 'no_record', 'boxes': s['boxes'] if s else [],
                                    'whole_image': s['whole_image'] if s else [],
                                    'scale_suspect': parsed['scale_suspect'] if parsed else None})
    d.refinement = []
    refined = Path(refined_dir) if refined_dir else run_dir / 'refined'
    if (refined / 'refinement.csv').exists():
        with (refined / 'refinement.csv').open() as f:
            d.refinement = list(csv.DictReader(f))
    d.refined_dir = refined
    d.features = {}
    features_dir = path(study['features']['dir']) if study['features']['dir'] else None
    if features_dir and (features_dir / 'features_manifest.json').exists():
        d.features = json.loads((features_dir / 'features_manifest.json').read_text())
        d.feature_seconds = [json.loads(p.read_text()).get('seconds') for p in features_dir.glob('*.json')
                             if p.name != 'features_manifest.json']
    else:
        d.feature_seconds = []
    return d
