#!/usr/bin/env python3
"""python run_vera.py --config configs/baseline.json --dataset ucf_crime"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from vera_runner.config import ROOT, load, path, prepare_environment
from vera_runner.timeline import Timeline
from vera_runner.data import resolve, sha256, preprocess, windows


def write(p,value):
    tmp=p.with_suffix(f'.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def allocate(root):
    root.mkdir(parents=True,exist_ok=True)
    n=1
    while True:
        p=root/f'run {n}'
        try:p.mkdir();return p
        except FileExistsError:n+=1


def run(cfg,run_dir,state,timeline):
    from PIL import Image
    from vera_runner.backends import BACKENDS,intervene,strict_label,prompt
    from vera_runner.grounding import box_window
    output=run_dir/'output'
    state['stage']='resolving_dataset'
    items=resolve(cfg);write(output/'dataset_manifest.json',items)
    timeline.emit('run',stages=cfg['stages'],videos=[x['id'] for x in items],model=cfg['model']['path'],boxes=cfg['boxes']['mode'])
    backend=None;all_gt=[];all_scores=[];summary=[]
    modes=['baseline'] if cfg['intervention']['mode']=='baseline' else ([ 'baseline',cfg['intervention']['mode']] if cfg['intervention']['probe'] else [cfg['intervention']['mode']])
    for item in items:
        dest=output/item['id'];dest.mkdir();state.update(video=item['id'],stage='preprocess')
        info=preprocess(item,dest/'frames');write(dest/'video.json',info)
        anchors=list(windows(info['decoded_frames'],cfg['sampling']))
        limit=cfg['selection']['max_windows_per_video']
        chosen=anchors[:limit] if limit else anchors
        write(dest/'coverage.json',{'total_windows':len(anchors),'selected_anchors':[x['anchor'] for x in chosen],
            'partial':len(chosen)!=len(anchors),'actual_fps':info['fps_actual'],'window_fps':cfg['sampling']['fps']})
        timeline.emit('video',video=item['id'],frames=info['decoded_frames'],fps=info['fps_actual'],size=info['native_size'],windows=len(chosen))
        if 'infer' not in cfg['stages']:continue
        if backend is None:
            state['stage']='loading_model';timeline.emit('model_loading',path=cfg['model']['path'])
            loading=time.monotonic();backend=BACKENDS[cfg['model']['backend']](cfg)
            state['device']=backend.device
            def progress(stage,details):
                state.update(model_stage=stage,model_progress=details)
                record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**details}
                with (output/'model_progress.jsonl').open('a') as telemetry:telemetry.write(json.dumps(record)+'\n')
                print('MODEL',json.dumps(record),flush=True)
            backend.progress=progress
            identity=getattr(backend,'model_identity',{'synthetic':True})
            write(output/'model_identity.json',identity)
            (output/'prompt.txt').write_text(prompt(cfg))
            timeline.emit('model_ready',device=backend.device,dtype=identity.get('dtype'),flash_attention=identity.get('flash_attention'),seconds=time.monotonic()-loading)
            timeline.emit('prompt',text=prompt(cfg),generation=cfg['generation'])
        records=[]
        with (dest/'raw_responses.jsonl').open('x') as raw:
            for number,window in enumerate(chosen):
                images=[]
                for index in window['frame_indices']:
                    with Image.open(dest/'frames'/f'{index:06d}.jpg') as im:
                        # Exactly the upstream grid_size=1 PIL resize. No aspect preservation or crop.
                        images.append(im.convert('RGB').resize((cfg['preprocessing']['image_size'],)*2))
                for mode in modes:
                    state.update(stage='inference',anchor=window['anchor'],condition=mode)
                    spec={**cfg['intervention'],'mode':mode}
                    altered=[intervene(im,spec) if mode!='baseline' and i in spec['slots'] else im for i,im in enumerate(images)]
                    inputs=dest/'inputs'/str(window['anchor'])/mode;inputs.mkdir(parents=True)
                    hashes=[]
                    for i,im in enumerate(altered):
                        im.save(inputs/f'{i:02d}.png');hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
                    paths=[f"{item['id']}/inputs/{window['anchor']}/{mode}/{i:02d}.png" for i in range(len(altered))]
                    where=dict(video=item['id'],anchor=window['anchor'],condition=mode)
                    timeline.emit('window',**where,index=number,total=len(chosen),start=window['start'],end=window['end'],frame_indices=window['frame_indices'],images=paths)
                    begin=time.monotonic();prediction=backend.predict(altered,on_text=timeline.streamer(**where))
                    text=prediction['raw_response']
                    record={**window,'condition':mode,'slots':spec['slots'] if mode!='baseline' else [],
                        **prediction,'strict_label':strict_label(text),'upstream_label':0 if '0' in text.split('Output')[-1] else 1,
                        'rgb_sha256':hashes,'seconds':time.monotonic()-begin,'synthetic':cfg['model']['backend']=='mock'}
                    raw.write(json.dumps(record)+'\n');raw.flush();records.append(record)
                    state['completed_requests']=state.get('completed_requests',0)+1
                    if not (output/'model_input.txt').exists():
                        rendered=backend.rendered_input();(output/'model_input.txt').write_text(rendered);timeline.emit('model_input',text=rendered)
                    timeline.emit('answer',**where,text=text,label=record['strict_label'],seconds=record['seconds'])
                    if cfg['boxes']['mode']=='alerts' and mode=='baseline' and record['strict_label']==1:
                        state['stage']='boxes'
                        boxed=box_window(backend,altered,text,cfg['boxes'],timeline,window['frame_indices'],paths,**where)
                        with (dest/'boxes.jsonl').open('a') as found:found.write(json.dumps(boxed)+'\n')
                    print(item['id'],window['anchor'],mode,record['strict_label'],flush=True)
        write(dest/'segment_scores.json',records)
        flips=[]
        if cfg['intervention']['probe']:
            for anchor in [x['anchor'] for x in chosen]:
                pair=[r for r in records if r['anchor']==anchor]
                flips.append({'anchor':anchor,'baseline_label':pair[0]['strict_label'],'intervened_label':pair[1]['strict_label'],
                    'flipped':None if any(r['strict_label'] is None for r in pair) else pair[0]['strict_label']!=pair[1]['strict_label']})
            write(dest/'probe_comparison.json',flips)
        if 'refine' in cfg['stages']:
            import numpy as np
            from vera_runner.refine import refine,ground_truth
            state['stage']='refinement'
            feature_path=Path(item['features']);write(dest/'feature_identity.json',{'path':str(feature_path),'sha256':sha256(feature_path),
                'alignment_requirement':'Rows correspond to ordered anchors in this run. Semantic feature extraction compatibility is the user responsibility.'})
            refined=refine(records,np.load(feature_path,allow_pickle=False),info['decoded_frames'],cfg['sampling']['stride'],cfg['refinement'])
            for key,values in refined.items():np.save(dest/(key+'.npy'),values)
            timeline.emit('refine',video=item['id'],arrays=[f"{item['id']}/{key}.npy" for key in refined])
            if 'evaluate' in cfg['stages']:
                all_gt.extend(ground_truth(item['annotation'],info['decoded_frames']).tolist());all_scores.extend(refined['frame_scores'].tolist())
        summary.append({'video':item['id'],'requests':len(records),'unparsed':sum(r['strict_label'] is None for r in records),'partial':len(chosen)!=len(anchors)})
    if 'evaluate' in cfg['stages']:
        from sklearn.metrics import roc_auc_score,average_precision_score
        if len(set(all_gt))<2:raise ValueError('ROC-AUC requires both classes in selected ground truth frames')
        write(output/'metrics.json',{'synthetic':cfg['model']['backend']=='mock','frame_roc_auc':roc_auc_score(all_gt,all_scores),'frame_average_precision':average_precision_score(all_gt,all_scores),
              'note':'Selected-set frame metrics; not full-dataset benchmark unless coverage matches the benchmark. This AP is not the XD-specific published AP protocol.'})
        timeline.emit('metrics',**{k:v for k,v in json.loads((output/'metrics.json').read_text()).items() if k!='note'})
    write(output/'summary.json',{'synthetic':cfg['model']['backend']=='mock','videos':summary})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--dataset');ap.add_argument('--split');ap.add_argument('--check',action='store_true')
    args=ap.parse_args();cfg=load(args.config,{'dataset':args.dataset,'split':args.split})
    root=path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=allocate(root/ 'checks' if args.check else root);out=folder/'output';out.mkdir()
    write(folder/'command.json',cfg);write(out/'invocation.json',{'argv':sys.argv,'cwd':os.getcwd(),'python':sys.executable})
    # Set before importing torch; no side effects on global user configuration.
    prepare_environment(cfg)
    state={'status':'checking','pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'inference_completed':False,'synthetic':cfg['model']['backend']=='mock'}
    write(out/'status.json',state);print('Run directory:',folder,flush=True)
    timeline=Timeline(out/'timeline.jsonl')
    stop=threading.Event()
    def beat():
        while not stop.wait(5):write(out/'status.json',{**state,'heartbeat_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    worker=threading.Thread(target=beat,daemon=True);worker.start()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    # An explicit per-run wall limit prevents unbounded silent model calls.
    def deadline():
        state['status']='timed_out';state['inference_completed']=False;write(out/'status.json',state)
        with (out/'error.txt').open('a') as f:f.write('\nRun exceeded runtime.timeout_seconds; terminated.\n')
        with (out/'diagnostic.log').open('w') as combined:
            for name in ('status.json','inference.log','error.txt'):
                p=out/name
                if p.exists():combined.write('\n=== '+name+' ===\n'+p.read_text())
        os._exit(124)
    timer=threading.Timer(cfg['runtime']['timeout_seconds'],deadline);timer.daemon=True;timer.start()
    try:
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        source=[p for p in ROOT.glob('vera_runner/*.py')]+[ROOT/'run_vera.py',ROOT/'generate_initial_score.py',path(cfg['prompt']['instruction_file'])]
        write(out/'source.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'working_diff':subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
            'sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in source}})
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for p in source:
            target=snapshot/(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        resolve(cfg)
        if args.check:
            if cfg['model']['backend']=='internvl2':
                import torch,transformers,torchvision,cv2,accelerate,einops,timm,sentencepiece
                state['mps_available']=torch.backends.mps.is_available();state['cuda_available']=torch.cuda.is_available()
                if cfg['model']['device']=='mps' and not state['mps_available']:raise RuntimeError('MPS unavailable to this process')
                from vera_runner.backends import resolve_checkpoint
                state['checkpoint']=resolve_checkpoint(cfg['model'])
            state['status']='check_passed'
        else:
            state['status']='running'
            with (out/'inference.log').open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):run(cfg,folder,state,timeline)
            state.update(status='complete',inference_completed='infer' in cfg['stages'] and cfg['model']['backend']!='mock')
    except BaseException as exc:
        state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';(out/'error.txt').write_text(traceback.format_exc())
        timeline.emit('error',message=f'{type(exc).__name__}: {exc}')
    finally:
        timer.cancel();stop.set();worker.join();state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(out/'status.json',state)
        timeline.emit('done',status=state['status'])
        with (out/'diagnostic.log').open('w') as combined:
            for name in ('status.json','inference.log','error.txt'):
                p=out/name
                if p.exists():combined.write('\n=== '+name+' ===\n'+p.read_text())
    print(json.dumps(state,indent=2));print('Diagnostics:',out/'diagnostic.log')
    return 0 if state['status'] in ('complete','check_passed') else 1

if __name__=='__main__':sys.exit(main())
