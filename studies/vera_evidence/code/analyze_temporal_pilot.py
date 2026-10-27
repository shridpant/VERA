"""Summarize complete or stopped pilot, retaining invalid outputs in denominators."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
 rows=[json.loads(x) for x in (a.run/'results.jsonl').read_text().splitlines()]
 groups={}
 for x in rows: groups.setdefault(x['window_uid'],{})[x['condition']]=x['result']
 conditions=['repeat','reverse','shuffle','single_frame'];summary={}
 for c in conditions:
  pairs=[(g['ordered'],g[c]) for g in groups.values() if 'ordered' in g and c in g]
  valid=[(x,y) for x,y in pairs if x['label'] in [0,1] and y['label'] in [0,1]]
  summary[c]={'available_pairs':len(pairs),'valid_label_pairs':len(valid),'label_changes':sum(x['label']!=y['label'] for x,y in valid),'exact_text_matches':sum(x['raw_response']==y['raw_response'] for x,y in pairs)}
 out={'complete':(a.run/'completion.json').exists(),'calls_recorded':len(rows),'windows_started':len(groups),'parse_failures':sum(x['result']['label'] not in [0,1] for x in rows),'at_generation_cap':sum((x['result'].get('n_new_tokens') or 0)>=1024 for x in rows),'generation_seconds':sum(x['result']['runtime_s'] for x in rows),'comparisons':summary,'per_window':{k:{c:v['label'] for c,v in g.items()} for k,g in groups.items()},'limitations':'Eight exploratory UCF windows; no model optimization, no detection-accuracy or temporal-faithfulness claim.'}
 (a.run/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
