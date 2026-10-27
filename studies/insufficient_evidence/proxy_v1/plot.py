import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);r=json.loads(a.analysis.read_text());keys=['original_annotated','original_gap','abstain_annotated','abstain_gap'];labels=['Original\nAnnotated','Original\nGap','Abstention\nAnnotated','Abstention\nGap'];fig,ax=plt.subplots(figsize=(7.5,3.7),constrained_layout=True);bottom=[0]*4
 for field,color,label in [('normal','#4c78a8','Normal (0)'),('alert','#e45756','Alert (1)'),('insufficient','#f2cf5b','Insufficient (2)'),('invalid','#777777','Invalid')]:
  vals=[r['counts'][k][field] for k in keys];ax.bar(range(4),vals,bottom=bottom,color=color,label=label);bottom=[x+y for x,y in zip(bottom,vals)]
 ax.set(xticks=range(4),xticklabels=labels,ylabel='Responses (12 sources per condition)',ylim=(0,15),title='Annotation-proxy pilot: response categories, not accuracy');ax.legend(ncol=4,fontsize=8,loc='upper center')
 for ext in ['pdf','png']:fig.savefig(a.out/f'proxy_responses.{ext}',dpi=180)
 table=['\\begin{tabular}{lrrrr}\\toprule','Condition & Normal & Alert & Insufficient & Invalid\\\\\\midrule']
 for k,l in zip(keys,labels):
  c=r['counts'][k];table.append(l.replace('\n',' / ')+' & '+' & '.join(str(c[f]) for f in ['normal','alert','insufficient','invalid'])+r'\\')
 table+=['\\bottomrule\\end{tabular}'];(a.out/'proxy_table.tex').write_text('\n'.join(table)+'\n')
if __name__=='__main__':main()
