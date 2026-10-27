import argparse,hashlib,json
from pathlib import Path

def rank(s): return hashlib.sha256(('proxy-v1|'+s).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--uca',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();u={};hashes={}
 for split in ['Train','Val','Test']:
  path=a.uca/'UCF Annotation/json'/f'UCFCrime_{split}.json';u.update(json.loads(path.read_text()));hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 prior=json.loads(a.prior.read_text())['inputs']; candidates={}
 for x in prior:
  key=x['video_id']
  if key not in u or x['development'] or key in candidates: continue
  d=u[key];ints=sorted(d['timestamps']);merged=[]
  for start,end in ints:
   if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
   else:merged.append([start,end])
  gaps=[];prev=0
  for start,end in merged:
   if start-prev>=5:gaps.append([prev,start])
   prev=end
  if d['duration']-prev>=5:gaps.append([prev,d['duration']])
  eligible=[z for z in ints if z[1]-z[0]>=5]
  if not eligible or not gaps:continue
  ann=min(eligible,key=lambda z:rank(key+'|ann|'+str(z)));gap=min(gaps,key=lambda z:rank(key+'|gap|'+str(z)))
  def view(z):
   center=sum(z)/2;start=center-2.5
   return {'container_seconds':z,'interval_seconds':[start,start+5],'sample_seconds':[start+5*(i+.5)/8 for i in range(8)]}
  candidates[key]={'video_id':key,'category':x['category'],'annotation_duration_seconds':d['duration'],'expected_video_sha256':x['video_sha256'],'manifest_entry':x['manifest_entry'],'manifest_length':x['manifest_length'],'views':{'annotated':view(ann),'gap':view(gap)}}
 normal=[x for x in candidates.values() if 'Normal' in x['category'] or x['video_id'].startswith('Normal')]
 abnormal=[x for x in candidates.values() if x not in normal];chosen=[];cats=set();sha=set()
 for x in sorted(abnormal,key=lambda x:rank(x['video_id'])):
  if x['category'] in cats or x['expected_video_sha256'] in sha:continue
  chosen.append(x);cats.add(x['category']);sha.add(x['expected_video_sha256'])
  if len(chosen)==8:break
 for x in sorted(normal,key=lambda x:rank(x['video_id'])):
  if x['expected_video_sha256'] in sha:continue
  chosen.append(x);sha.add(x['expected_video_sha256'])
  if len(chosen)==12:break
 assert len(chosen)==12 and len(cats)==8
 out={'scope':'previously consulted UCF test; exploratory prompt comparison only; no learning','selection':'hash-ranked one per first eight anomaly categories and four normal sources; five-second views; no response selection','source_hashes':hashes,'prior_inputs_sha256':hashlib.sha256(a.prior.read_bytes()).hexdigest(),'videos':chosen}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print([(x['video_id'],x['category']) for x in chosen])
if __name__=='__main__':main()
