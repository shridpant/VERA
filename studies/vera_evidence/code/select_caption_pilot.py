"""Freeze ID-only candidate pools. No images, captions, labels or predictions used for ranking."""
import argparse,hashlib,json
from pathlib import Path

def name(x):return x.rsplit('/',1)[-1].removesuffix('.mp4')
def main():
 p=argparse.ArgumentParser();p.add_argument('--uca',type=Path,required=True);p.add_argument('--vera',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out={'status':'candidate metadata manifest; not authorization to infer or evidence of source independence','selection':'sort sha256(caption-positive-v1|split|video_id) within joint splits','splits':{},'source_sha256':{}}
 test={name(x['video']) for x in json.loads((a.vera/'UCF_Eval.json').read_text())}
 for split,n in [('train',16),('val',12)]:
  up=a.uca/'UCF Annotation/json'/f'UCFCrime_{split.title()}.json';vp=a.vera/f'UCF_Instruct_{split}.json';u=json.loads(up.read_text());v={name(x['video']) for x in json.loads(vp.read_text())};pool=set(u)&v
  assert not pool&test
  ids=sorted(pool,key=lambda x:hashlib.sha256(f'caption-positive-v1|{split}|{x}'.encode()).hexdigest())[:n]
  assert len(ids)==n
  out['splits'][split]={'eligible_count':len(pool),'selected_ids':ids}
  for q in [up,vp]:out['source_sha256'][q.name]=hashlib.sha256(q.read_bytes()).hexdigest()
 assert not set(out['splits']['train']['selected_ids'])&set(out['splits']['val']['selected_ids'])
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['splits'],indent=2))
if __name__=='__main__':main()
