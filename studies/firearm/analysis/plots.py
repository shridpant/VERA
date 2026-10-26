"""P1-P7 and P9 (P8 lives in qualitative.py). Static PDF+PNG figures for the report, light surface.

Colour roles (validated with the dataviz palette checker, adjacent CVD dE >= 9.2):
conditions A #2a78d6, G #1baf7a, AG #eb6834; outcomes are two steps of one blue ramp (normal
light, anomaly dark) plus hatched grey for invalid; counts use a single-hue blue ramp.
Every multi-series panel has a legend, and identity never rests on colour alone.
"""
from pathlib import Path
import numpy as np
from .load import CATEGORIES
from .tables import paired_windows, refined_rows, subset, valid

COND = {'A': '#2a78d6', 'G': '#1baf7a', 'AG': '#eb6834'}
OUTCOME = {'normal': '#b7d3f6', 'anomaly': '#184f95', 'invalid': '#e1e0d9'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
STATUS_COLORS = {'boxes': '#2a78d6', 'empty': '#b7d3f6', 'unknown': '#eda100', 'malformed': '#e87ba4', 'other': '#898781'}


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.6,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'figure.facecolor': 'white',
                         'axes.facecolor': SURFACE, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 200, 'pdf.fonttype': 42})
    return plt


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [str(out / f'{name}.pdf'), str(out / f'{name}.png')]


def label_name(cat):
    return cat.replace('_', ' ')


def p1(d, out):
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        ws = subset(d.windows, cat)
        for i, c in enumerate(('A', 'AG')):
            n = len(ws)
            parts = [('normal', sum(w[f'{c}_label'] == 0 for w in ws)), ('anomaly', sum(w[f'{c}_label'] == 1 for w in ws)),
                     ('invalid', sum(not valid(w[f'{c}_label']) for w in ws))]
            bottom = 0.0
            for name, k in parts:
                h = 100 * k / n if n else 0
                ax.bar(i, h, bottom=bottom, width=0.6, color=OUTCOME[name], edgecolor=SURFACE, linewidth=1.5,
                       hatch='////' if name == 'invalid' else None, label=name if (cat == CATEGORIES[0] and i == 0) else None)
                if h >= 6:
                    ax.text(i, bottom + h / 2, f'{h:.0f}%', ha='center', va='center', fontsize=7,
                            color='white' if name == 'anomaly' else INK)
                bottom += h
        ax.set_xticks([0, 1], ['A', 'AG'])
        ax.set_title(f'{label_name(cat)} (n={len(ws)} windows)', fontsize=8, color=INK)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of scheduled windows')
    fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
    return save(fig, out, 'P1_predictions')


def p2(d, out):
    plt = setup()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for j, cat in enumerate(CATEGORIES):
        per = {}
        for w in paired_windows(d, cat):
            per.setdefault(w['video'], []).append((w['A_label'], w['AG_label']))
        a = np.array([np.mean([x for x, _ in v]) for v in per.values()])
        g = np.array([np.mean([y for _, y in v]) for v in per.values()])
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color='#c3c2b7', linewidth=0.8)
        jitter = np.random.default_rng(0).uniform(-0.01, 0.01, (2, len(a))) if len(a) else np.zeros((2, 0))
        ax.scatter(a + jitter[0], g + jitter[1], s=12, color=COND['A'], edgecolor=SURFACE, linewidth=0.6, alpha=0.8)
        ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{label_name(cat)} ({len(a)} videos)', fontsize=8, color=INK)
        ax.set_xlabel('A anomaly fraction')
        if j == 0:
            ax.set_ylabel('AG anomaly fraction')
        ax = axes[1, j]
        diff = g - a
        ax.hist(diff, bins=np.linspace(-1, 1, 21), color=COND['AG'], edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel('AG − A per-video anomaly fraction')
        if j == 0:
            ax.set_ylabel('videos')
    fig.tight_layout()
    return save(fig, out, 'P2_paired_video_fractions')


def p3(d, out):
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, cat in zip(axes, CATEGORIES):
        ws = paired_windows(d, cat)
        m = np.array([[sum(w['A_label'] == a and w['AG_label'] == b for w in ws) for b in (0, 1)] for a in (0, 1)])
        ax.imshow(m, cmap='Blues', vmin=0, vmax=max(1, m.max()))
        for a in (0, 1):
            for b in (0, 1):
                share = 100 * m[a, b] / m.sum() if m.sum() else 0
                ax.text(b, a, f'{m[a, b]}\n{share:.0f}%', ha='center', va='center', fontsize=7,
                        color='white' if m[a, b] > 0.6 * m.max() else INK)
        ax.set_xticks([0, 1], ['AG normal', 'AG anomaly'])
        ax.set_yticks([0, 1], ['A normal', 'A anomaly'])
        ax.grid(False)
        ax.set_title(f'{label_name(cat)} ({m.sum()} paired)', fontsize=8, color=INK)
    fig.tight_layout()
    return save(fig, out, 'P3_transitions')


def p4(d, out):
    plt = setup()
    ref = refined_rows(d)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    positions, ticks = [], []
    for i, cat in enumerate(CATEGORIES):
        for k, c in enumerate(('A', 'AG')):
            vals = [v['mean'] for (cc, _), v in ref.items() if cc == c and v['category'] == cat]
            x = i * 3 + k
            if vals:
                ax.boxplot([vals], positions=[x], widths=0.6, patch_artist=True, showfliers=False,
                           boxprops={'facecolor': COND[c], 'edgecolor': COND[c], 'alpha': 0.35},
                           medianprops={'color': INK}, whiskerprops={'color': MUTED}, capprops={'color': MUTED})
                ax.scatter(x + np.random.default_rng(i * 2 + k).uniform(-0.18, 0.18, len(vals)), vals, s=6, color=COND[c], zorder=3)
            positions.append(x)
        ticks.append((i * 3 + 0.5, f'{label_name(cat)}'))
    ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks])
    ax.set_ylabel('per-video mean refined frame score')
    ax.scatter([], [], color=COND['A'], label='A'), ax.scatter([], [], color=COND['AG'], label='AG')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax = axes[1]
    for cat, marker in zip(CATEGORIES, ('o', 's', '^')):
        common = [(ref[('A', v)]['mean'], ref[('AG', v)]['mean']) for (c, v) in ref if c == 'A' and ('AG', v) in ref and ref[('A', v)]['category'] == cat]
        if common:
            x, y = zip(*common)
            ax.scatter(x, y, s=12, marker=marker, color=MUTED, label=f'{label_name(cat)} ({len(common)})', edgecolor=SURFACE, linewidth=0.5)
    lim = [0, max([v['mean'] for v in ref.values()] + [0.1]) * 1.05]
    ax.plot(lim, lim, color='#c3c2b7', linewidth=0.8)
    ax.set_xlabel('A per-video mean score'), ax.set_ylabel('AG per-video mean score')
    ax.legend(loc='upper left')
    fig.tight_layout()
    return save(fig, out, 'P4_refined_scores')


def p5(d, out):
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        for k, cam in enumerate(('C1', 'C2')):
            ws = [w for w in subset(d.windows, cat) if w['camera'] == cam]
            res = sorted({v['resolution'] for v in d.videos if v['category'] == cat and v['camera'] == cam})
            for j, c in enumerate(('A', 'AG')):
                good = [w[f'{c}_label'] for w in ws if valid(w[f'{c}_label'])]
                h = 100 * np.mean(good) if good else 0
                x = k * 2.6 + j
                ax.bar(x, h, width=0.9, color=COND[c], edgecolor=SURFACE, linewidth=1.5, label=c if (cat == CATEGORIES[0] and k == 0) else None)
                ax.text(x, h + 1.5, f'n={len(good)}', ha='center', fontsize=6, color=MUTED)
        ax.set_xticks([0.5, 3.1], [f'C1 {res_label(d, cat, "C1")}', f'C2 {res_label(d, cat, "C2")}'])
        ax.set_title(label_name(cat), fontsize=8, color=INK)
        ax.set_ylim(0, 110)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% anomaly (valid windows)')
    fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    return save(fig, out, 'P5_camera_strata')


def res_label(d, cat, cam):
    vs = [v for v in d.videos if v['category'] == cat and v['camera'] == cam]
    return f"{'/'.join(sorted({v['resolution'] for v in vs}))}\n{len(vs)} videos"


def p6(d, out):
    plt = setup()
    groups = ['boxes', 'empty', 'unknown', 'malformed', 'other']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        for i, c in enumerate(('G', 'AG')):
            ss = [s for s in subset(d.slots, cat) if s['condition'] == c]
            n = len(ss)
            bottom = 0.0
            for g in groups:
                k = sum((s['status'] if s['status'] in groups[:-1] else 'other') == g for s in ss)
                h = 100 * k / n if n else 0
                ax.bar(i, h, bottom=bottom, width=0.6, color=STATUS_COLORS[g], edgecolor=SURFACE, linewidth=1.5,
                       label=g if (cat == CATEGORIES[0] and i == 0) else None)
                if h >= 6:
                    ax.text(i, bottom + h / 2, f'{h:.0f}%', ha='center', va='center', fontsize=7, color='white' if g == 'boxes' else INK)
                bottom += h
        ax.set_xticks([0, 1], ['G', 'AG'])
        ax.set_title(f'{label_name(cat)} ({n} slots)', fontsize=8, color=INK)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of scheduled slots')
    fig.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.04, 'other = missing, duplicate, truncated or no complete record', ha='center', fontsize=7, color=MUTED)
    return save(fig, out, 'P6_box_status')


def p7(d, t8_rows, out):
    """Small multiples: rows precision / recall / F1, columns Handgun / Machine_Gun / All; bars by IoU threshold."""
    plt = setup()
    thresholds = d.study['matching']['iou_thresholds']
    rows = [r for r in t8_rows if r['origin'] == 'primary' and not r.get('size')]
    cats = ('Handgun', 'Machine_Gun', 'All')
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharey=True, sharex=True)
    for i, metric in enumerate(('precision', 'recall', 'f1')):
        for j, cat in enumerate(cats):
            ax = axes[i, j]
            for k, thr in enumerate(thresholds):
                for m, c in enumerate(('G', 'AG')):
                    r = next(x for x in rows if x['category'] == cat and x['iou'] == thr and x['condition'] == c)
                    v = r[metric]
                    x = k * 2.6 + m
                    ax.bar(x, v or 0, width=0.9, color=COND[c], edgecolor=SURFACE, linewidth=1.5,
                           label=c if (i == 0 and j == 0 and k == 0) else None)
                    ax.text(x, (v or 0) + 0.03, 'n/a' if v is None else f'{v:.2f}', ha='center', fontsize=6, color=MUTED)
            ax.set_xticks([k * 2.6 + 0.5 for k in range(len(thresholds))], [f'IoU {t}' for t in thresholds])
            ax.set_ylim(0, 1.12)
            ax.grid(axis='x', visible=False)
            if i == 0:
                ax.set_title(label_name(cat), fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(metric)
    fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    saved = save(fig, out, 'P7_localization')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    from vera_runner.localization import score_instance
    ious = {c: [] for c in ('G', 'AG')}
    for s in d.slots:
        if s['frame_status'] == 'positive':
            ious[s['condition']].extend(score_instance(s['status'], s['boxes'], s['gt'], thresholds[0], True)['matched_ious'])
    ax = axes[0]
    for c in ('G', 'AG'):
        if ious[c]:
            ax.hist(ious[c], bins=np.linspace(thresholds[0], 1, 11), color=COND[c], alpha=0.6, label=f'{c} ({len(ious[c])})', edgecolor=SURFACE)
    if not any(ious.values()):
        ax.text(0.5, 0.5, f'no matched boxes at IoU >= {thresholds[0]}', ha='center', va='center', transform=ax.transAxes, color=MUTED)
    ax.set_xlabel('IoU of matched boxes'), ax.set_ylabel('matches')
    if any(ious.values()):
        ax.legend(loc='upper left')
    ax = axes[1]
    sizes = ('small', 'medium', 'large')
    for m, c in enumerate(('G', 'AG')):
        vals = []
        for size in sizes:
            r = next((x for x in t8_rows if x.get('size') == size and x['condition'] == c and x['category'] == 'All'), None)
            vals.append((r or {}).get('recall'))
        for k, v in enumerate(vals):
            ax.bar(k * 2.6 + m, v or 0, width=0.9, color=COND[c], edgecolor=SURFACE, label=c if k == 0 else None)
            ax.text(k * 2.6 + m, (v or 0) + 0.03, 'n/a' if v is None else f'{v:.2f}', ha='center', fontsize=6, color=MUTED)
    ax.set_xticks(np.arange(3) * 2.6 + 0.5, sizes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f'recall at IoU {thresholds[0]} (all firearm videos)')
    ax.set_xlabel('ground-truth size (area tertiles)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    return saved + save(fig, out, 'P7b_iou_and_size')


def p9(d, out):
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, key, xlabel in ((axes[0], 'seconds', 'generation seconds per call'), (axes[1], 'generated_tokens', 'generated tokens per call')):
        data = [[r[key] for r in d.final[c].values()] for c in ('A', 'G', 'AG')]
        parts = ax.boxplot(data, vert=False, patch_artist=True, showfliers=True, widths=0.6,
                           flierprops={'markersize': 2, 'markeredgecolor': MUTED})
        for patch, c in zip(parts['boxes'], ('A', 'G', 'AG')):
            patch.set_facecolor(COND[c]), patch.set_alpha(0.5), patch.set_edgecolor(COND[c])
        for med in parts['medians']:
            med.set_color(INK)
        ax.set_yticks([1, 2, 3], ['A', 'G', 'AG'])
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    return save(fig, out, 'P9_runtime_tokens')
