"""One accepted native-coordinate candidate; fail closed and preserve all old runs."""
import concurrent.futures, fcntl, hashlib, json, os, pathlib, subprocess, sys, time
root=pathlib.Path('/local/datasets/sp2117/firearm_visilab/posttraining1')
release=pathlib.Path('/local/datasets/sp2117/firearm_visilab/code/posttraining-releases/8d9232c')
python='/local/datasets/sp2117/envs/firearm-posttrain/bin/python'
os.chdir(release)
sys.path.insert(0,str(release))
from studies.firearm_posttraining.data import digest,write_json
from studies.firearm_posttraining.execute import ALLOWED
from studies.firearm_posttraining.runtime import append
lock=(root/'native-coordinate-driver-v1.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.time()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
audit=json.loads((root/'data/coordinate-audit-v1.json').read_text())
assert audit['status']=='passed' and audit['code_commit']==commit
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
for name,expected in audit['code_hashes'].items():assert digest(release/'studies/firearm_posttraining'/name)==expected
for name,expected in audit['data_hashes'].items():assert digest(root/'data'/name)==expected
assert not pathlib.Path('/proc/293624').exists()
assert not (root/'protocol.json').exists()
model=json.loads((root/'data/preflight.json').read_text())['model']
def status(stage,**kw):
 write_json(root/'native-coordinate-driver-v1-status.json',{'stage':stage,'pid':os.getpid(),'started_unix':started,'updated_unix':time.time(),'code_commit':commit,**kw})
try:
 status('fresh_engineering_checks')
 def smoke(index,kind):
  name=f'smoke-{kind}-006'
  child=[python,'-m','studies.firearm_posttraining','smoke','--model',model,'--data',str(root/'data'),'--output',str(root/name)]
  if kind=='grounded':child.append('--grounded')
  cmd=[python,'-m','studies.firearm_posttraining.execute','--root',str(root),'--gpu',ALLOWED[index],'--name',name,'--stage','pilot','--seconds','600','--',*child]
  with (root/f'{name}-launch.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  r=json.loads((root/name/'result.json').read_text());assert r['status']=='engineering_passed'
  return r
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(smoke,i,k) for i,k in enumerate(('labels','grounded'))]
  results=[f.result() for f in futures]
 status('calibration_and_conditional_pipeline',smoke_statuses=[x['status'] for x in results])
 cmd=[python,'-m','studies.firearm_posttraining.calibrate','--root',str(root),'--original-data','/local/datasets/sp2117/firearm_visilab/training1/data','--frames-root','/local/datasets/sp2117/firearm_visilab/training1/frames','--control-name','calibration-native-coordinates-v1','--attempt-offset','8','--warmup-grid','3','--label-smoke',str(root/'smoke-labels-006/result.json'),'--grounded-smoke',str(root/'smoke-grounded-006/result.json'),'--sequential-panels']
 append(root/'operator-actions.jsonl',{'action':'accepted_native_coordinate_candidate_launch','authorization_chat':'01a17dce-3275-78b0-b005-0f3a408d5cbe','command':cmd,'code_commit':commit,'driver_sha256':digest(__file__),'scoring_rules_changed':False,'fresh_warmups':True})
 with (root/'calibration-native-coordinates-v1-controller.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 final=json.loads((root/'calibration-native-coordinates-v1/status.json').read_text())
 status(final['stage'],controller=final)
except Exception as exc:
 status('failed',error_type=type(exc).__name__,error=str(exc))
 raise
