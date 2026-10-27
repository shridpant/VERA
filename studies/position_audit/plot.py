"""Scientific figures/tables from frozen audit outputs; no synthetic results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('results', type=Path); p.add_argument('out', type=Path); a = p.parse_args()
    s = json.loads((a.results/'summary.json').read_text()); a.out.mkdir(parents=True, exist_ok=True)
    primary = s['cohorts']['ratio_15_all']; x = np.array(primary['offsets'])*100
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for method, color, label, style in [('original', '#0072b2', 'Original weighting', '-o'), ('no_position', '#333333', 'No position weighting', '--s')]:
        axes[0].plot(x, [100*z['auc'] for z in primary['point'][method]], style, color=color, label=label)
    axes[0].set(xlabel='Circular shift (% of each video)', ylabel='Pooled frame ROC-AUC (%)', title='Fixed evidence, changed absolute position', xticks=x)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    for key, color, label in [('ratio_15_all', '#0072b2', 'All 290 videos'), ('ratio_15_exclude5', '#e69f00', 'Exclude five endpoint cases')]:
        b = json.loads((a.results/key/'bootstrap.json').read_text())
        axes[1].hist([100*z['worst_benefit'] for z in b], bins=35, alpha=.45, color=color, density=True, label=label)
    axes[1].axvline(0, color='#444444', lw=1); axes[1].axvline(3, color='#d55e00', linestyle='--', label='Required benefit: 3 pp')
    axes[1].set(xlabel='No-position worst-shift benefit (pp)', ylabel='Bootstrap density', title='Paired whole-video resampling')
    axes[1].legend(fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(a.out/f'position_audit.{ext}', dpi=180)
    plt.close(fig)
    hist = np.asarray(primary['label_position_bins']); centers = (np.arange(20)+.5)/20
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.bar(centers, 100*hist[:, 1]/hist[:, 0], width=.045, color='#0072b2')
    ax.set(xlabel='Relative recording position', ylabel='Frames labelled anomalous (%)', title='Existing benchmark label distribution')
    twin = ax.twinx(); z = np.linspace(0, 1, 200); twin.plot(z, np.exp(-.5*((z-.5)/.5)**2), color='#d55e00')
    twin.set(ylabel='Approximate Gaussian position weight', ylim=(0, 1.05))
    for ext in ('pdf', 'png'):
        fig.savefig(a.out/f'label_positions.{ext}', dpi=180)
    plt.close(fig)
    table = [r'\begin{tabular}{lrrrrr}\toprule', r'Shift & Original AUC & No-pos AUC & Prior AUC & Original AP & No-pos AP\\\midrule']
    for i, offset in enumerate(x):
        values = [primary['point'][m][i][metric]*100 for m, metric in [('original', 'auc'), ('no_position', 'auc'), ('prior_only', 'auc'), ('original', 'ap'), ('no_position', 'ap')]]
        table.append(f'{offset:.0f}'+r'\%'+' & '+' & '.join(f'{v:.3f}' for v in values)+r'\\')
    table.append(r'\bottomrule\end{tabular}'); (a.out/'primary_table.tex').write_text('\n'.join(table)+'\n')
    table = [r'\begin{tabular}{lrrrrl}\toprule', r'Cohort & Spread & Benefit & 95\% CI & Cost & Pass\\\midrule']
    for key, v in s['cohorts'].items():
        g = v['gate']; lo, hi = np.asarray(v['worst_benefit_ci95'])*100
        label = key.replace('ratio_15', '15 pct').replace('ratio_10', '10 pct').replace('_exclude5', ' / 285').replace('_all', ' / 290')
        table.append(f"{label} & {100*g['spread']:.3f} & {100*g['worst_position_benefit']:.3f} & [{lo:.3f}, {hi:.3f}] & {100*g['unshifted_cost']:.3f} & {'yes' if g['pass'] else 'no'}"+r'\\')
    table.append(r'\bottomrule\end{tabular}'); (a.out/'gate_table.tex').write_text('\n'.join(table)+'\n')


if __name__ == '__main__':
    main()
