"""T9: one row per scheduled video, with links to its artifacts (paths relative to the run folder)."""
from vera_runner.localization import prf, score_instance
from .tables import refined_rows, table, valid


def t9(d):
    ref = refined_rows(d)
    reasons = {(r['condition'], r['video']): r['reasons'] for r in d.refinement}
    thr = d.study['matching']['iou_thresholds'][0]
    loc = {}
    for s in d.slots:
        if s['frame_status'] in ('positive', 'negative_category'):
            r = score_instance(s['status'], s['boxes'], s['gt'], thr, True)
            acc = loc.setdefault((s['condition'], s['video']), [0, 0, 0])
            acc[0] += r['tp']
            acc[1] += (r['fp'] or 0) if r['answered'] else 0
            acc[2] += r['fn']
    rows = []
    for v in d.videos:
        ws = [w for w in d.windows if w['video'] == v['id']]
        row = {'video': v['id'], 'category': v['category'], 'camera': v['camera'], 'resolution': v['resolution'], 'fps': v['fps'],
               'duration_s': round(v['duration_s'], 2), 'pilot': v['pilot'], 'recording_group': v['recording_group'],
               'scheduled_windows': len(ws)}
        for c in ('A', 'AG'):
            good = [w[f'{c}_label'] for w in ws if valid(w[f'{c}_label'])]
            row.update({f'{c}_complete': sum(w[f'{c}_complete'] for w in ws), f'{c}_anomaly': sum(good),
                        f'{c}_normal': len(good) - sum(good), f'{c}_invalid': len(ws) - len(good),
                        f'{c}_anomaly_fraction': sum(good) / len(good) if good else None,
                        f'{c}_refined_mean': ref.get((c, v['id']), {}).get('mean'), f'{c}_refined_max': ref.get((c, v['id']), {}).get('max'),
                        f'{c}_refinement_exclusion': reasons.get((c, v['id']), '')})
        row['G_complete'] = sum(w['G_complete'] for w in ws)
        for c in ('G', 'AG'):
            tp, fp, fn = loc.get((c, v['id']), (0, 0, 0))
            precision, recall, _ = prf(tp, fp, fn)
            row.update({f'{c}_tp@{thr}': tp, f'{c}_fp@{thr}': fp, f'{c}_fn@{thr}': fn,
                        f'{c}_recall@{thr}': None if v['category'] == 'No_Gun' else recall,
                        f'{c}_precision@{thr}': None if v['category'] == 'No_Gun' else precision})
        row['records'] = 'records/{A,G,AG}.jsonl'
        row['inputs'] = f"videos/{v['id']}/inputs/"
        row['refined'] = f"refined/{{A,AG}}/{v['id']}/" if any((c, v['id']) in ref for c in ('A', 'AG')) else ''
        rows.append(row)
    return table('T9', 'Per-video results', list(rows[0]), rows, [
        'Anomaly fraction = anomaly windows / valid windows. Localization at the first IoU threshold on boxed and No_Gun frames; '
        'No_Gun videos have false positives only. Paths are relative to the run folder.'])
