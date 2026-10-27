"""Frozen original versus manual-abstention comparison; no learning."""
import argparse,hashlib,json,os,re,sys,time
from pathlib import Path
FLAG="Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
OLD="You are designed to do binary classification. The input is a sequence of video frames for identifying whether there is an anomaly in the video; you need to output the class label, i.e., an integer in the set {0, 1}. 0 represents normal video, and 1 represents abnormal video. Please answer the prompt questions."
NEW="Assess the supplied video frames. Output 1 when the frames support an anomaly, 0 when they support a normal scene, or 2 when the supplied visual evidence is insufficient to decide. Missing evidence is not evidence of normality. Do not assume unseen events, sounds, intent or causes. Please answer the prompt questions."
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(text,ternary):
 fields=re.split(r'(?i)\bOutput\s*:',text)
 if len(fields)!=2:return None
 tail=fields[1].strip().strip('`').strip()
 return int(tail) if tail in (('0','1','2') if ternary else ('0','1')) else None

def manual_instruction(original,questions):
 assert original.count(OLD)==1 and original.count(FLAG)==1
 assert '[ONLY the integer class label; make necessary assumptions if needed]' in original
 return original.replace(OLD,NEW).replace(FLAG,questions+"Based only on the supplied visual evidence, conclude with 0 (normal), 1 (anomaly), or 2 (insufficient evidence). If evidence does not resolve the question, explicitly state the uncertainty rather than guessing.").replace('[ONLY the integer class label; make necessary assumptions if needed]','[ONLY one integer: 0, 1, or 2; do not make assumptions to fill missing evidence]')

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 os.environ['VERA_PROJECT_ROOT']=str(a.base);sys.path.insert(0,str(a.base/'experiments/experiment03B/code'));import config3b
 sys.path.insert(0,str(a.base/'experiments/experiment02/code'));import engine
 cfgpath=a.base/'experiments/experiment03B/runs/full_v1/frozen/config.json';cfg=config3b.exp02_compat(json.loads(cfgpath.read_text()));data=json.loads(a.inputs.read_text());cases=data['cases'];assert len(cases)==24
 start=time.time();dependencies=[cfgpath,a.base/'experiments/experiment02/code/engine.py',a.base/'experiments/experiment02/code/attr_lib.py',a.base/'setup/source/VERA/generate_initial_score.py',a.base/'setup/source/VERA/VERA_learner_instruct.txt']
 provenance={'start_unix':start,'runner_sha256':sha(__file__),'inputs_sha256':sha(a.inputs),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'dependency_sha256':{str(x):sha(x) for x in dependencies},'model':cfg['model'],'generation':cfg['generation'],'call_limit':48,'seconds_limit':3600,'scope':'previously consulted UCFtest exploratory proxy; no optimization'}
 (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 em=a.out/'eval_manifest.json';seen={x['manifest_entry']:x['manifest_length'] for x in cases};em.write_text(json.dumps([{'video':k,'length':v} for k,v in seen.items()]))
 eng=engine.Engine(cfg,em);original,qs=eng.prompt_parts['original'];ins=manual_instruction(original,qs)
 eng.prompt_parts['abstain']=(ins,'');eng.wrappers['abstain']=eng.gis.VideoAnomalyDetectionModelInference(model=eng.model,tokenizer=eng.tok,optimizer_instruct=None,model_instruct=ins,new_qs='',generation_config=dict(cfg['generation'])).eval().cuda()
 prompts={v:eng.effective_question(v) for v in ['original','abstain']}
 for v,text in prompts.items():(a.out/f'prompt_{v}.txt').write_text(text)
 records=[];hashes={};videos=list(dict.fromkeys(x['video_id'] for x in cases))
 with (a.out/'results.jsonl').open('x') as f:
  for vi,video in enumerate(videos):
   vc=[x for x in cases if x['video_id']==video]
   if vi%2:vc.reverse()
   variants=['original','abstain'] if vi%2==0 else ['abstain','original']
   for case in vc:
    for variant in variants:
     if len(records)>=48 or time.time()-start>=3500:raise RuntimeError('Budget stop')
     result=eng.generate(variant,case,(),{'family':'none'})
     assert result['question_sha256']==hashlib.sha256(prompts[variant].encode()).hexdigest()
     if case['case_id'] in hashes:assert hashes[case['case_id']]==result['slot_tensor_sha256']
     hashes[case['case_id']]=result['slot_tensor_sha256']
     decision=parse(result['raw_response'],variant=='abstain')
     rec={'video_id':video,'category':case['category'],'view':case['condition'],'variant':variant,'decision':decision,'result':result};records.append(rec);f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
     print(json.dumps({'video':video,'view':case['condition'],'variant':variant,'decision':decision,'runtime':result['runtime_s']}),flush=True)
     if decision is None:raise RuntimeError('Strict output parse failure; preserve partial run')
 assert len(records)==48
 summary={'calls':len(records),'wall_seconds':time.time()-start,'generation_seconds':sum(x['result']['runtime_s'] for x in records),'engine':eng.summary(),'dependencies_unchanged':all(sha(p)==h for p,h in provenance['dependency_sha256'].items()),'paired_tensor_checks':len(hashes)};assert summary['dependencies_unchanged']
 (a.out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
