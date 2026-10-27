import json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=Path(__file__).resolve().parents[1];a=json.loads((s/'evidence/temporal_pilot_v1/run_001/analysis.json').read_text())
fig,ax=plt.subplots(figsize=(7,3.3),constrained_layout=True)
keys=list(a['comparisons']);x=range(4)
ax.bar([i-.18 for i in x],[a['comparisons'][k]['label_changes'] for k in keys],width=.36,label='Decision changed')
ax.bar([i+.18 for i in x],[8-a['comparisons'][k]['exact_text_matches'] for k in keys],width=.36,label='Explanation text changed')
ax.set(xticks=list(x),xticklabels=['Identical repeat','Reversed','Shuffled','Single frame ×8'],ylabel='Windows out of 8',ylim=(0,8),title='Exploratory original VERA: fixed eight-window pilot')
ax.legend();fig.savefig(s/'figures/temporal_pilot.pdf');fig.savefig(s/'figures/temporal_pilot.png',dpi=160)
