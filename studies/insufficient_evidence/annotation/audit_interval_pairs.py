"""Metadata feasibility only: unannotated gaps are NOT verified event absence."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--uca',type=Path,required=True);p.add_argument('--vera',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data={};hashes={}
 for split in ['Train','Val','Test']:
  path=a.uca/'UCF Annotation/json'/f'UCFCrime_{split}.json';data.update(json.loads(path.read_text()));hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 out={'scope':'metadata-only feasibility; gaps do not establish missing visual evidence','minimum_interval_seconds':2,'sources_sha256':hashes,'splits':{}}
 for split in ['train','val']:
  path=a.vera/f'UCF_Instruct_{split}.json';hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();eligible=[];covered=0
  for row in json.loads(path.read_text()):
   key=row['video'].rsplit('/',1)[-1].removesuffix('.mp4')
   if key not in data:continue
   covered+=1;d=data[key];intervals=sorted(d['timestamps']);merged=[]
   for start,end in intervals:
    if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
    else:merged.append([start,end])
   gaps=[];prev=0
   for start,end in merged:
    if start-prev>=2:gaps.append([prev,start])
    prev=end
   if d['duration']-prev>=2:gaps.append([prev,d['duration']])
   candidates=[v for v in intervals if v[1]-v[0]>=2]
   if candidates and gaps:eligible.append({'video_id':key,'annotated_intervals':len(candidates),'unannotated_gaps':len(gaps)})
  out['splits'][split]={'uca_covered':covered,'has_2s_annotation_and_gap':len(eligible),'eligible':eligible}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{x:y for x,y in v.items() if x!='eligible'} for k,v in out['splits'].items()},indent=2))
if __name__=='__main__':main()
