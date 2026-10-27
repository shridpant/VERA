"""Descriptive paired proxy outcomes; intentionally no accuracy metric."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();rows=[json.loads(x) for x in (a.run/'results.jsonl').read_text().splitlines()];counts={};pairs={}
 for v in ['original','abstain']:
  for view in ['annotated','gap']:
   r=[x for x in rows if x['variant']==v and x['view']==view];counts[v+'_'+view]={'n':len(r),'normal':sum(x['decision']==0 for x in r),'alert':sum(x['decision']==1 for x in r),'insufficient':sum(x['decision']==2 for x in r),'invalid':sum(x['decision'] is None for x in r),'at_cap':sum((x['result'].get('n_new_tokens') or 0)>=1024 for x in r)}
 for r in rows:pairs.setdefault(r['video_id'],{})[r['variant']+'_'+r['view']]=r['decision']
 complete=(a.run/'completion.json').exists();out={'complete':complete,'counts':counts,'per_video':pairs,'call_count':len(rows),'generation_seconds':sum(x['result']['runtime_s'] for x in rows),'warning':'Annotation coverage proxy only. Insufficient outputs are not verified correct; binary original cannot abstain.'}
 if complete:
  out['gap_minus_annotated_abstentions']=counts['abstain_gap']['insufficient']-counts['abstain_annotated']['insufficient'];out['completion']=json.loads((a.run/'completion.json').read_text())
 (a.run/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
