"""Read-only reproduction of attempt010 format verdicts; never rescues failures."""
import sys,json,pathlib,hashlib
root=pathlib.Path(sys.argv[1]);release=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(release))
from studies.firearm_posttraining.rewards import parse
from studies.firearm_posttraining.data import digest
book=json.loads((release/'studies/firearm_training/codebook.json').read_text())
out={'attempt':10,'release':str(release),'scoring_rules_changed':False,'arms':{}}
for kind in ('l','g'):
 d=root/f'format-{kind}-010';rows=[json.loads(l) for l in (d/'responses.jsonl').read_text().splitlines()]
 result=json.loads((d/'result.json').read_text());failures=[];box_count=0
 for row in rows:
  parsed=parse(row['text'],book,kind=='g',row['truncated'])
  assert (parsed is not None)==row['valid']
  if parsed is not None and kind=='g':
   for view in parsed['grounding']:
    for b in view['boxes']:
     assert 0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1
     box_count+=1
  if parsed is None:
   failure={'video':row['video'],'sample':row['sample'],'truncated':row['truncated'],'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'raw_text':row['text']}
   try:
    json.loads(row['text']);failure['cause']='strict_schema_or_geometry'
   except json.JSONDecodeError as error:
    failure.update(cause='invalid_json',message=error.msg,offset=error.pos,ends_with_code_fence=row['text'].rstrip().endswith('```'),ends_with_array=row['text'].rstrip().endswith(']'))
   failures.append(failure)
 assert len(rows)==result['responses'] and len(failures)==result['invalid']
 out['arms'][kind]={'result':result,'validated_unit_boxes_in_valid_responses':box_count,'failures':failures,'artifacts':{p.name:digest(p) for p in (d/'result.json',d/'spec.json',d/'responses.jsonl')}}
a=json.loads((root/'pilot-c0l-008/checkpoints/final-000576/state.json').read_text())
b=json.loads((root/'pilot-c0l-010/checkpoints/final-000576/state.json').read_text())
out['fresh_label_adapter_hash']=b['adapter_hash'];out['old_label_adapter_hash']=a['adapter_hash'];out['fresh_label_exact_reproduction']=a['adapter_hash']==b['adapter_hash']
print(json.dumps(out,indent=2,sort_keys=True))
