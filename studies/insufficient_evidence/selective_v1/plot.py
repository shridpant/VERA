"""Regenerate descriptive scientific figure and tables; never label these accuracy."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument('analysis', type=Path); p.add_argument('out', type=Path); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    r = json.loads(a.analysis.read_text()); keys = ['original', 'manual', 'cited']; c = r['conditions']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for i, (field, color, label) in enumerate([('annotated', '#4c78a8', 'Annotation-included'), ('gap', '#e69f00', 'Unannotated gap')]):
        axes[0].bar([x+(i-.5)*.36 for x in range(3)], [c[k]['alerts_by_view'][field] for k in keys], width=.36, color=color, label=label)
    axes[0].set(xticks=range(3), xticklabels=['Original', 'Manual', 'Cited'], ylabel='Anomaly decisions (out of 12 per view)', ylim=(0, 14), title='Binary decisions; correctness unknown')
    axes[0].legend(fontsize=8)
    audit = json.loads((a.analysis.parent.parent/'output_audit.json').read_text())['run_001']['conditions']
    for i, (field, color, label) in enumerate([('valid_schema', '#009e73', 'Valid JSON schema'), ('empty_observation_lists', '#cc79a7', 'Nonempty observations'), ('responses_with_nonplaceholder_observations', '#d55e00', 'Nonplaceholder observations')]):
        axes[1].bar([x+(i-1)*.24 for x in range(2)], [(audit[k]['valid_schema']-audit[k]['empty_observation_lists']) if field == 'empty_observation_lists' else audit[k][field] for k in ['manual', 'cited']], width=.24, color=color, label=label)
    axes[1].set(xticks=range(2), xticklabels=['Manual', 'Cited'], ylabel='Responses (out of 24)', ylim=(0, 29), title='Empty/template content despite valid JSON')
    axes[1].legend(fontsize=8)
    for ext in ['pdf', 'png']:
        fig.savefig(a.out/f'selective_responses.{ext}', dpi=180)
    table = [r'\begin{tabular}{lrrr}\toprule', r'Measure & Original & Manual & Cited\\\midrule']
    fields = [('Alerts / 24', 'alerts'), ('Agreement with original / 24', 'binary_agreement_with_original'), ('Retained original alerts', 'retained_original_alerts'), ('Valid explanation JSON / 24', 'valid_explanation_schema'), ('Nonempty unresolved / 24', 'nonempty_unresolved'), ('Alert plus unresolved / 24', 'alert_and_unresolved'), ('Observation claims', 'observation_count'), ('Claims with legal citations', 'observations_with_legal_nonempty_citations'), ('At token cap', 'at_token_cap')]
    for label, f in fields:
        table.append(label+' & '+' & '.join('--' if c[k][f] is None else str(c[k][f]) for k in keys)+r'\\')
    table.append('Mean reference-word recall & '+' & '.join(f"{c[k]['positive_reference_word_recall_mean']:.3f}" for k in keys)+r'\\')
    table.append(r'\bottomrule\end{tabular}')
    (a.out/'summary.tex').write_text('\n'.join(table)+'\n')
    table = [r'\begin{longtable}{lrrr}\toprule', r'Source / view & Original & Manual & Cited\\\midrule\endhead']
    for case, values in r['all_cases'].items():
        name = case.replace('_x264', '').replace('Normal_Videos_', 'Normal').replace('|annotated', ' / ann.').replace('|gap', ' / gap').replace('_', r'\_')
        table.append(name+' & '+' & '.join(str(values[k]) for k in keys)+r'\\')
    table.append(r'\bottomrule\end{longtable}')
    (a.out/'all_cases.tex').write_text('\n'.join(table)+'\n')


if __name__ == '__main__':
    main()
