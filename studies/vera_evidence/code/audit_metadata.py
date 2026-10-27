"""Read split metadata only; never infer that metadata means videos are staged."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=['UCF_Instruct_train.json','UCF_Instruct_val.json','UCF_Eval.json','xd_train.json','xd_val.json','xd_test_800.json']
sets={};result={'scope':'metadata only; no images/predictions read','splits':{},'intersections':{}}
for name in paths:
 p=root/'Data'/name;x=json.loads(p.read_text());items=x if isinstance(x,list) else list(x.values())
 ids=[]
 for row in items:
  if not isinstance(row,dict):continue
  ident=row.get('video',row.get('video_name',row.get('path')))
  if ident:
   base=Path(ident).name
   for suffix in ('.mp4','.avi','.mkv'):
    if base.lower().endswith(suffix):base=base[:-len(suffix)];break
   ids.append(base)
 sets[name]=set(ids)
 result['splits'][name]={'entries':len(x),'parsed_video_ids':len(ids),'unique_video_ids':len(set(ids)),'sha256':digest(p),'sample_keys':list(items[0]) if isinstance(items[0],dict) else []}
for family in [('UCF_Instruct_train.json','UCF_Instruct_val.json','UCF_Eval.json'),('xd_train.json','xd_val.json','xd_test_800.json')]:
 for i,a in enumerate(family):
  for b in family[i+1:]:result['intersections'][a+' vs '+b]=sorted(sets[a]&sets[b])
result['limitations']=['Filename disjointness is not source-byte or scene disjointness.','Dataset paths embedded in original XD JSON refer to the original authors host.','Training QA can contain time intervals; do not silently introduce stronger temporal supervision into a weak-label comparison.']
out=root/'studies/vera_evidence/evidence/metadata_audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
