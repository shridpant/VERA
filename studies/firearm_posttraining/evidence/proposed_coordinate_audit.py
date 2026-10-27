"""CPU-only feasibility preview of the unaccepted representation proposal."""
import sys,json,hashlib,pathlib
root=pathlib.Path(sys.argv[1]); release=pathlib.Path(sys.argv[2]); model=sys.argv[3]
sys.path.insert(0,str(release))
from studies.firearm_posttraining.model import target_parts,token_kinds
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
max_old=max_new=unknown=boxes=rows=0; inventory=[]; max_error=0
for split in ('fit','stop'):
 for row in json.loads((root/'data'/f'{split}.json').read_text()):
  rows+=1
  old,spans=target_parts(row,True)
  chunks=[]; new_spans=[]; cursor=0
  for a,b,kind in spans:
   chunk=old[a:b]
   if kind==2:
    values=json.loads(chunk)
    replacement=[]
    for box in values:
     native=[round(x*1000,2) for x in box]
     assert 0<=native[0]<native[2]<=1000 and 0<=native[1]<native[3]<=1000
     max_error=max(max_error,max(abs(x-y/1000) for x,y in zip(box,native)))
     boxes+=1; replacement.append(native)
    chunk=json.dumps(replacement,separators=(',',':'))
   chunks.append(chunk);new_spans.append((cursor,cursor+len(chunk),kind));cursor+=len(chunk)
  proposed=''.join(chunks)
  old_ids=tok(old,add_special_tokens=False)['input_ids']
  encoded=tok(proposed,add_special_tokens=False,return_offsets_mapping=True)
  kinds=token_kinds(encoded['offset_mapping'],new_spans)
  for (a,b),kind in zip(encoded['offset_mapping'],kinds):
   touches_unknown=any(a<y and b>x and z==0 for x,y,z in new_spans)
   assert (kind==0)==touches_unknown
   unknown+=touches_unknown
  max_old=max(max_old,len(old_ids)+1);max_new=max(max_new,len(kinds)+1)
  inventory.append([split,row['id'],hashlib.sha256(proposed.encode()).hexdigest()])
assert max_new<=512 and max_error<1e-15
print(json.dumps({'status':'cpu_proposal_preview_passed_not_training_authorization','development_videos':rows,'known_boxes_checked':boxes,
 'maximum_old_completion_tokens':max_old,'maximum_proposed_completion_tokens':max_new,
 'maximum_unit_roundtrip_error_relative_to_old_rounded_targets':max_error,'unknown_payload_tokens_masked':unknown,
 'proposed_grounded_target_inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'no_gpu_inference_or_training':True,'no_reporting_split_access':True,
 'limitation':'Read-only preview only; does not verify future implementation or demonstrate scientific feasibility.'},indent=2))
