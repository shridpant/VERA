"""Metadata-only UCA/VERA crosswalk; does not read caption text for model selection."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def ident(x):
 x=x.rsplit('/',1)[-1]
 for ext in ('.mp4','.avi','.mkv'):
  if x.lower().endswith(ext):x=x[:-len(ext)];break
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--uca',type=Path,required=True);p.add_argument('--vera',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 u={};v={};files={};stats={}
 for split in ['Train','Val','Test']:
  path=a.uca/'UCF Annotation/json'/f'UCFCrime_{split}.json';raw=path.read_bytes();data=json.loads(raw);u[split]={ident(x) for x in data};files[path.name]=hashlib.sha256(raw).hexdigest()
  problems=[]
  for key,d in data.items():
   if len(d['timestamps'])!=len(d['sentences']):problems.append([key,'count_mismatch'])
   for j,(start,end) in enumerate(d['timestamps']):
    if not 0<=start<end<=d['duration']:problems.append([key,j,'interval_outside_duration_or_nonpositive',start,end,d['duration']])
  stats[split]={'videos':len(data),'sentences':sum(len(d['sentences']) for d in data.values()),'interval_problems':problems}
 for split,name in [('Train','UCF_Instruct_train.json'),('Val','UCF_Instruct_val.json'),('Test','UCF_Eval.json')]:
  path=a.vera/name;raw=path.read_bytes();v[split]={ident(x['video']) for x in json.loads(raw)};files[name]=hashlib.sha256(raw).hexdigest()
 matrix={i:{j:len(u[i]&v[j]) for j in v} for i in u}
 out={'uca_upstream':'https://github.com/Xuange923/Surveillance-Video-Understanding','uca_commit':subprocess.check_output(['git','-C',str(a.uca),'rev-parse','HEAD'],text=True).strip(),'file_sha256':files,'uca_statistics':stats,'uca_rows_vera_columns':matrix,'uca_internal_split_overlap':{i+'_'+j:len(u[i]&u[j]) for i,j in [('Train','Val'),('Train','Test'),('Val','Test')]},'normalization':'basename; remove only known video extension; no fuzzy name mapping','warning':'Different split conventions. Exclude every VERA test ID from any learning regardless of UCA split. Caption absence is not negative evidence. ID checks do not establish scene independence.'}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
