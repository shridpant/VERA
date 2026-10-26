"""T11 and P8: qualitative cases chosen by a rule fixed before the full run, never by hand.

Rule (seed 20261025):
  change+   the 2 videos with the largest positive AG-minus-A anomaly fraction (paired windows, >= 3 of them)
  change-   the 2 videos with the largest negative difference
  unchanged 2 videos with zero difference, drawn with the seed (one firearm video, one No_Gun video when possible)
  loc_high  the video with the highest AG recall at the first IoU threshold (>= 10 ground-truth objects)
  loc_low   the video with the lowest AG recall under the same condition
Ties break by video ID. In each chosen video the shown window is the first one whose A and AG labels
differ, else the window with the most boxed source frames, else the middle window. Successes and
failures are shown alike; the rule is printed with the cases.
"""
import json
import random
from pathlib import Path
import numpy as np
from vera_runner.localization import score_instance
from .load import CATEGORIES
from .tables import valid

SEED = 20261025
RULE = __doc__


def per_video_change(d):
    per = {}
    for w in d.windows:
        if valid(w['A_label']) and valid(w['AG_label']):
            per.setdefault(w['video'], []).append(w['AG_label'] - w['A_label'])
    return {v: float(np.mean(x)) for v, x in per.items() if len(x) >= 3}


def per_video_recall(d, condition, threshold):
    out = {}
    for s in d.slots:
        if s['condition'] == condition and s['frame_status'] == 'positive':
            r = score_instance(s['status'], s['boxes'], s['gt'], threshold, True)
            tp, gt = out.get(s['video'], (0, 0))
            out[s['video']] = (tp + r['tp'], gt + len(s['gt']))
    return {v: tp / gt for v, (tp, gt) in out.items() if gt >= 10}


def select(d):
    change = per_video_change(d)
    picks = []
    ordered = sorted(change.items(), key=lambda x: (-x[1], x[0]))
    picks += [('change+', v) for v, x in ordered[:2] if x > 0]
    picks += [('change-', v) for v, x in sorted(change.items(), key=lambda x: (x[1], x[0]))[:2] if x < 0]
    rng = random.Random(SEED)
    category = {v['id']: v['category'] for v in d.videos}
    zero = sorted(v for v, x in change.items() if x == 0)
    for want in (lambda v: category[v] != 'No_Gun', lambda v: category[v] == 'No_Gun'):
        pool = [v for v in zero if want(v) and v not in [p for _, p in picks]]
        if pool:
            picks.append(('unchanged', rng.choice(pool)))
    recall = per_video_recall(d, 'AG', d.study['matching']['iou_thresholds'][0])
    if recall:
        high = sorted(recall.items(), key=lambda x: (-x[1], x[0]))[0][0]
        low = sorted(recall.items(), key=lambda x: (x[1], x[0]))[0][0]
        picks.append(('loc_high', high))
        if low != high:
            picks.append(('loc_low', low))
    return picks, change, recall


def window_for(d, video):
    ws = [w for w in d.windows if w['video'] == video]
    for w in ws:
        if valid(w['A_label']) and valid(w['AG_label']) and w['A_label'] != w['AG_label']:
            return w, 'first window whose A and AG labels differ'
    boxed = {}
    for s in d.slots:
        if s['video'] == video and s['condition'] == 'G' and s['frame_status'] == 'positive':
            boxed[s['anchor']] = boxed.get(s['anchor'], 0) + 1
    if boxed:
        best = sorted(boxed.items(), key=lambda x: (-x[1], x[0]))[0][0]
        return next(w for w in ws if w['anchor'] == best), 'window with the most boxed source frames'
    return ws[len(ws) // 2], 'middle window'


def slot_summary(d, video, anchor, condition):
    ss = [s for s in d.slots if s['video'] == video and s['anchor'] == anchor and s['condition'] == condition]
    return ' '.join(f"F{s['slot'] + 1}:{s['status']}" + (f"{s['boxes']}" if s['boxes'] else '') for s in sorted(ss, key=lambda x: x['slot']))


def build(d, inventory_path, original_root, out_dir):
    from studies.firearm.overlays import ground_truth, latest_complete, window_sheet
    plt = __import__('studies.firearm.analysis.plots', fromlist=['setup']).setup()
    out_dir = Path(out_dir)
    picks, change, recall = select(d)
    rows_by_id = {r['id']: r for r in json.loads(Path(inventory_path).read_text())}
    records = {c: latest_complete(d.run_dir, c) for c in ('A', 'G', 'AG')}
    cases, figures = [], []
    for n, (reason, video) in enumerate(picks, 1):
        w, why = window_for(d, video)
        row = rows_by_id[video]
        sheet = window_sheet(d.run_dir, row, w['anchor'], ground_truth(row, original_root, d.policy['origin']), records,
                             out_dir / 'sheets' / f'case{n}_{video}_a{w["anchor"]}.png')
        ws = sorted((x for x in d.windows if x['video'] == video), key=lambda x: x['anchor'])
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={'height_ratios': [1, 3]})
        ax = axes[0]
        for c, offset, color in (('A', 0.0, '#2a78d6'), ('AG', 0.06, '#eb6834')):
            t = [x['time_s'] for x in ws]
            y = [x[f'{c}_label'] + offset if valid(x[f'{c}_label']) else np.nan for x in ws]
            ax.step(t, y, where='post', color=color, linewidth=1.5, label=c)
            bad = [x['time_s'] for x in ws if not valid(x[f'{c}_label'])]
            ax.scatter(bad, [0.5] * len(bad), marker='x', color=color, s=14)
        ax.axvline(w['time_s'], color='#898781', linewidth=0.8)
        ax.set_yticks([0, 1], ['normal', 'anomaly']), ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('window anchor time (s, native frame rate)')
        ax.set_title(f"Case {n} ({reason}): {video}", fontsize=8)
        ax.legend(loc='center right', fontsize=7)
        axes[1].imshow(plt.imread(sheet)), axes[1].axis('off')
        axes[1].set_title('Exact inputs of the marked window: G red, AG blue, existing boxes green', fontsize=7)
        fig.tight_layout()
        target = out_dir / 'figures'
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f'P8_case{n}.pdf', bbox_inches='tight'), fig.savefig(target / f'P8_case{n}.png', bbox_inches='tight')
        plt.close(fig)
        figures.append(f'P8_case{n}')
        cases.append({'case': n, 'reason': reason, 'video': video, 'category': row['category'], 'camera': row['camera'],
                      'anchor': w['anchor'], 'time_s': round(w['time_s'], 2), 'window_choice': why,
                      'video_change_pp': None if video not in change else round(100 * change[video], 1),
                      'video_AG_recall': recall.get(video), 'A_label': w['A_label'], 'AG_label': w['AG_label'],
                      'A_text': w['A_text'], 'AG_text': w['AG_text'],
                      'G_slots': slot_summary(d, video, w['anchor'], 'G'), 'AG_slots': slot_summary(d, video, w['anchor'], 'AG'),
                      'boxed_frames': sum(1 for s in d.slots if s['video'] == video and s['anchor'] == w['anchor']
                                          and s['condition'] == 'G' and s['frame_status'] == 'positive'),
                      'sheet': str(sheet.relative_to(out_dir)) if Path(sheet).is_relative_to(out_dir) else str(sheet)})
    (out_dir / 'qualitative.json').write_text(json.dumps({'rule': RULE, 'cases': cases}, indent=1, ensure_ascii=False) + '\n')
    return cases, figures
