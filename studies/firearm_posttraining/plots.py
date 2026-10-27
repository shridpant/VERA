"""Render exported training curves with a standard scientific plotting library."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path


def run(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(Path(source).open()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    colors = {'SL': '#2563eb', 'RL': '#7c3aed', 'SG': '#059669', 'RG': '#dc2626'}
    for method, color in colors.items():
        for seed in sorted({r['seed'] for r in rows}):
            selected = [r for r in rows if r['method'] == method and r['seed'] == seed]
            if not selected:
                continue
            for ax, key in zip(axes, ('fit_passes', 'logical_gpu_hours')):
                points = sorted(selected, key=lambda r: float(r[key]))
                ax.plot([float(r[key]) for r in points], [float(r['balanced_category_accuracy']) for r in points],
                        color=color, marker='o', markersize=3, alpha=.65, label=method)
    for ax, title in zip(axes, ('Distinct-video FIT passes', 'Logical training GPU-hours (includes warm-up)')):
        ax.set_xlabel(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('STOP balanced category accuracy')
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False)
    fig.suptitle('Selection-set learning curves; each line is one seed')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'learning-curves.{extension}', dpi=220)
    plt.close(fig)
    (output / 'plot-environment.json').write_text(json.dumps({k: importlib.metadata.version(k) for k in ('matplotlib', 'numpy')}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('output')
    args = p.parse_args()
    run(args.source, args.output)
