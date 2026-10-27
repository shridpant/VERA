"""Original-VERA exploratory diagnostic, separate output and read-only dependencies."""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['VERA_PROJECT_ROOT']=str(a.base)
    sys.path.insert(0,str(a.base/'experiments/experiment03B/code'))
    import config3b
    sys.path.insert(0,str(a.base/'experiments/experiment02/code'))
    import engine
    cfgpath=a.base/'experiments/experiment03B/runs/full_v1/frozen/config.json'
    cfg=config3b.exp02_compat(json.loads(cfgpath.read_text()))
    inp=json.loads(a.manifest.read_text())['inputs']
    em=a.out/'eval_manifest.json';em.write_text(json.dumps([{'video':x['manifest_entry'],'length':x['manifest_length']} for x in inp]))
    dependencies=[cfgpath,a.base/'experiments/experiment02/code/engine.py',a.base/'experiments/experiment02/code/attr_lib.py',a.base/'experiments/experiment03B/code/config3b.py',a.base/'setup/source/VERA/generate_initial_score.py']
    provenance={'start_unix':time.time(),'python':platform.python_version(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'runner_sha256':sha(__file__),'manifest_sha256':sha(a.manifest),'dependency_sha256':{str(x):sha(x) for x in dependencies},'generation':cfg['generation'],'kind':'exploratory original VERA; not LRPO reproduction; no learning'}
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    eng=engine.Engine(cfg,em)
    orders={'ordered':list(range(8)),'repeat':list(range(8)),'reverse':list(reversed(range(8))),'shuffle':[3,0,6,1,7,4,2,5],'single_frame':[3]*8}
    records=[]
    with (a.out/'results.jsonl').open('x') as f:
        for x in inp:
            baseline=None
            for condition,order in orders.items():
                case={'case_id':x['window_uid']+'|'+condition,'video_id':x['video_id'],'anchor':x['anchor'],'frames_dir':x['frames_dir'],'video_sha256':x['video_sha256'],'frame_indices':[x['frame_indices'][i] for i in order],'frame_sha256':[x['frame_sha256'][i] for i in order]}
                result=eng.generate('original',case,(),{'family':'none'})
                hashes=result['slot_tensor_sha256']
                if baseline is None: baseline=hashes
                if hashes != [baseline[i] for i in order]: raise RuntimeError('Input permutation tensor mismatch')
                rec={'window_uid':x['window_uid'],'video_sha256':x['video_sha256'],'strata':x['strata'],'condition':condition,'order':order,'result':result}
                records.append(rec);f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps({'window':x['window_uid'],'condition':condition,'label':result['label'],'runtime_s':result['runtime_s']}),flush=True)
    summary={'n_calls':len(records),'wall_seconds':time.time()-provenance['start_unix'],'engine':eng.summary(),'all_dependency_hashes_unchanged':all(sha(x)==provenance['dependency_sha256'][str(x)] for x in dependencies)}
    assert summary['all_dependency_hashes_unchanged']
    (a.out/'completion.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
