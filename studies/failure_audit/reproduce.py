"""CPU-only verification of the claims in main.tex. No model calls.
Usage: python reproduce.py --evidence E --annotations A --out verification.json
E/A are directories obtained by extracting the two supplied tar.gz archives.
"""
import argparse, collections, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent / 'code'))
from training_stats import cluster_bootstrap
from ag_stats import cluster_bootstrap_sums
from slots import parse_slots, final_output_label
from firearm_annotations import frame_boxes, load_coco
from localization import score_instance
from training_parser import parse_answer

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
checks=[]
def check(name,actual,expected,tol=1e-12):
    ok=(math.isclose(actual,expected,rel_tol=tol,abs_tol=tol) if isinstance(actual,(float,np.floating)) and isinstance(expected,(int,float)) else actual==expected)
    checks.append({'claim':name,'actual':actual,'expected':expected,'passed':bool(ok)})
    if not ok: raise AssertionError(checks[-1])
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def table(p):return list(csv.DictReader(p.open()))
for root in [a.evidence,a.annotations]:
    manifest=load(root/'manifest.json')
    for f in manifest['files']:
        b=(root/f['relative']).read_bytes()
        check('sha256:'+f['relative'],hashlib.sha256(b).hexdigest(),f['sha256'])
b=a.evidence/'local/datasets/sp2117/firearm_visilab'
ab=a.annotations/'local/datasets/sp2117/firearm_visilab'
splits={s:load(b/f'training1/data/{s}.json') for s in ['train','val','test']}
book={'categories':sorted({r['category'] for rs in splits.values() for r in rs}),'groups':sorted({r['action_group'] for rs in splits.values() for r in rs}),'codes':sorted({r['action'] for rs in splits.values() for r in rs})}
for s,n in [('train',240),('val',62),('test',96)]:check('split '+s,len(splits[s]),n)
for s,t in [('train','val'),('train','test'),('val','test')]:
    check('recording-group disjoint '+s+'/'+t,bool({r['recording_group'] for r in splits[s]} & {r['recording_group'] for r in splits[t]}),False)

manifest=load(b/'training1/compare/compare_manifest.json')
c1=table(b/'training1/compare/tables/C1.csv'); c2=table(b/'training1/compare/tables/C2.csv')
training=[]; reference=None
for path,row,delta in zip(manifest['runs'],c1,c2):
    rd=b/'training1/runs'/Path(path).name
    spec=load(rd/'spec.json'); tests={x['name']:x for x in lines(rd/'test.jsonl')}
    ss=tests['seed']['predictions']; learned=tests['learned']['predictions']
    truth={r['id']:[r['category'],r['action_group'],r['action']] for r in splits['test']}
    for r in ss+learned:
        if r['target']!=truth[r['video']] or parse_answer(r['response'],book)!=r['pred']:raise AssertionError('training target/parser mismatch')
    check(row['run']+' test n',len(learned),96)
    if reference is None:reference={x['video']:x['response'] for x in ss}
    check(row['run']+' identical seed responses',sum(reference[x['video']] != x['response'] for x in ss),0)
    vals=[x for x in lines(rd/'validations.jsonl') if not x['reference_only']]
    selected=max(vals,key=lambda v:(v['metrics']['category_accuracy'],v['metrics']['group_accuracy'],-v['step']))
    for v in vals:
        check(row['run']+' validation accuracy step '+str(v['step']),sum(x['pred']['category']==x['target'][0] for x in v['predictions'])/len(v['predictions']),v['metrics']['category_accuracy'])
    check(row['run']+' selected step',selected['step'],int(row['selected_step']))
    for k,f in enumerate(['category','group','code']):
        correct=lambda x:x['pred'][f]==x['target'][k]
        accuracy=lambda u:sum(correct(x) for x in u)/len(u)
        units=[{**x,'group_key':x['recording_group']} for x in learned]
        est,lo,hi,n=cluster_bootstrap(units,'group_key','category',accuracy,2000,20261026,f'S4 learned {f}')
        for key,v in [('acc',est),('low',lo),('high',hi)]:check(row['run']+' '+f+' '+key,v,float(row[f'test_{f}_{key}']))
        if k==0:
            sm={x['video']:x for x in ss}; lm={x['video']:x for x in learned}
            paired=[{**sm[v],'other':lm[v],'g':sm[v]['recording_group']} for v in sorted(sm)]
            stat=lambda u:sum(correct(x['other'])-correct(x) for x in u)/len(u)
            d,dl,dh,_=cluster_bootstrap(paired,'g','category',stat,2000,20261026,'S4 diff category')
            for key,v in [('',d),('_low',dl),('_high',dh)]:check(row['run']+' paired category'+key,v,float(delta['vs_seed_category'+key]))
            trainrow={'run':row['run'],'seed':spec['training']['seed'],'method':spec['training'].get('method','vera'),'correct':sum(correct(x) for x in learned),'n':96,'accuracy':est,'ci':[lo,hi],'difference':d,'difference_ci':[dl,dh],'selected_step':selected['step'],'validation_accuracy':selected['metrics']['category_accuracy']}
    check(row['run']+' invalid',sum(x['pred']['category'] is None for x in learned),int(row['test_invalid']))
    check(row['run']+' capped',sum(x['generated_tokens']>=1024 for x in learned),int(row['test_at_cap']))
    trainrow['group_correct']=sum(x['pred']['group']==x['target'][1] for x in learned)
    trainrow['code_correct']=sum(x['pred']['code']==x['target'][2] for x in learned)
    trainrow['invalid']=sum(x['pred']['category'] is None for x in learned)
    trainrow['capped']=sum(x['generated_tokens']>=1024 for x in learned)
    training.append(trainrow)

g=lines(b/'grounding1/phase1_seed_measurements.jsonl'); gate=load(b/'grounding1/gate.json')
check('grounding records',len(g),398);check('grounding unique videos',len({r['video'] for r in g}),398)
ev=[r for r in g if r.get('lift') is not None];check('grounding evaluable',len(ev),276)
v1=[r for r in ev if r.get('lift_shared') is not None and r.get('lift_shuffled') is not None]
u=[{'g':r['recording_group'],'category':r['category'],'d':r['lift_shared']-r['lift_shuffled']} for r in v1]
v1res=cluster_bootstrap(u,'g','category',lambda z:float(np.mean([r['d'] for r in z])),2000,20261026,'V1')
v2=[r for r in ev if r.get('reliance') is not None]
def rho(z):return float(spearmanr([r['lift'] for r in z],[r['reliance'] for r in z]).statistic)
u=[{**r,'g':r['recording_group']} for r in v2]
v2res=cluster_bootstrap(u,'g','category',rho,2000,20261026,'V2')
for name,res,key in [('V1',v1res,'mean_difference'),('V2',v2res,'rho')]:
    check(name+' estimate',res[0],gate[name][key]);check(name+' lower',res[1],gate[name]['ci'][0]);check(name+' upper',res[2],gate[name]['ci'][1])
check('grounding failed gate',v1res[1]>0 and v2res[1]>0,False)
correct=[r for r in ev if r['pred']['category']==r['category']]
check('correct firearm decisions',len(correct),112)
check('correct box-mask flips',sum(r['occlusion']['box']['pred']['category']!=r['category'] for r in correct),112)
check('correct control-mask flips',sum(r['occlusion']['control']['pred']['category']!=r['category'] for r in correct),5)

inventory={r['id']:r for r in load(ab/'inventory.json')}; agdir=b/'runs/firearm_a_g_ag_v1'
records={c:lines(agdir/f'records/{c}.jsonl') for c in ['A','G','AG']}
for c,rs in records.items():
    check(c+' complete records',len(rs),6375);check(c+' distinct windows',len({(r['video'],r['anchor']) for r in rs}),6375)
    check(c+' non-synthetic complete',all(not r['synthetic'] and r['status']=='complete' for r in rs),True)
    for r in rs:
        if c in ['A','AG']:
            if final_output_label(r['raw_response']) != r['label']:raise AssertionError('label reparse mismatch')
        if c in ['G','AG']:
            if parse_slots(r['raw_response'],truncated=r['truncated']) != r['grounding']:raise AssertionError('slot reparse mismatch')
    checks.append({'claim':c+' all raw responses reparse identically','actual':len(rs),'expected':6375,'passed':True})
am={(r['video'],r['anchor']):r for r in records['A']}
paired=[]
for r in records['AG']:
    ar=am[(r['video'],r['anchor'])]
    check_same=ar['frame_indices']==r['frame_indices'] and ar['rgb_sha256']==r['rgb_sha256']
    if not check_same:raise AssertionError('unpaired pixels')
    if ar['label'] in (0,1) and r['label'] in (0,1):paired.append({'video':r['video'],'category':inventory[r['video']]['category'],'d':r['label']-ar['label']})
check('A/AG valid pairs',len(paired),6370)
d,lo,hi,_=cluster_bootstrap_sums(paired,'video','category',lambda r:(r['d'],1),lambda t:t[:,0]/t[:,1],2000,20261025,'T4 pooled All')
t4=next(r for r in table(b/'analysis/tables/T4.csv') if r['category']=='All')
for name,v in [('diff_pp',d),('diff_pp_ci_low',lo),('diff_pp_ci_high',hi)]:check('A/AG '+name,100*v,float(t4[name]))
annotations={}
for vid,row in inventory.items():
    if row.get('label_file'):
        cp=ab/'original/Gun_Action_Recognition_Dataset'/Path(row['source_path']).parent/row['label_file']
        annotations[vid]=frame_boxes(load_coco(cp),row['width'],row['height'],1)
boxresults={}
for c in ['G','AG']:
    slots=[s for r in records[c] for s in r['grounding']['slots']]
    t7=next(r for r in table(b/'analysis/tables/T7.csv') if r['condition']==c and r['category']=='All')
    empty=sum(s['status']=='empty' for s in slots); emitted=sum(len(s['boxes']) for s in slots); whole=sum(sum(s['whole_image']) for s in slots)
    for name,v in [('empty',empty),('boxes_emitted',emitted),('whole_image_boxes',whole)]:check(c+' '+name,v,int(t7[name]))
    counts={thr:{'tp':0,'fn':0,'gt':0} for thr in [0.5,0.75]}
    for r in records[c]:
        for s in r['grounding']['slots']:
            gt=annotations.get(r['video'],{}).get(r['frame_indices'][s['slot']],[])
            for thr,v in counts.items():
                sc=score_instance(s['status'],s['boxes'],gt,thr,True)
                v['tp']+=sc['tp'];v['fn']+=sc['fn'];v['gt']+=len(gt)
    for thr,v in counts.items():
        check(c+' gt '+str(thr),v['gt'],23841);check(c+' tp '+str(thr),v['tp'],0);check(c+' fn '+str(thr),v['fn'],23841)
    boxresults[c]={'slots':len(slots),'empty':empty,'emitted':emitted,'whole_image':whole,'counts':counts}

arates={}
for cat in ['Handgun','Machine_Gun','No_Gun','All']:
    rs=[r for r in records['A'] if cat=='All' or inventory[r['video']]['category']==cat]
    v=sum(r['label']==1 for r in rs)/len(rs)
    t=next(r for r in table(b/'analysis/tables/T3.csv') if r['category']==cat)
    check('A anomaly fraction '+cat,100*v,float(t['anomaly_pct']))
    arates[cat]={'alerts':sum(r['label']==1 for r in rs),'n':len(rs),'percent':100*v}

support=Path(__file__).parent/'supporting'
for f in load(support/'manifest.json'):
    check('supporting sha256:'+f['relative'],hashlib.sha256((support/f['relative']).read_bytes()).hexdigest(),f['sha256'])
complete=lines(support/'selective_complete.jsonl');partial=lines(support/'selective_partial.jsonl')
check('selective completed calls',len(complete),72);check('selective repair calls',len(partial),6)
manual=[r for r in complete if r['variant']=='manual' and r['explanation'] is not None]
check('manual valid schemas',len(manual),18);check('manual empty observation lists',sum(not r['explanation']['observations'] for r in manual),18)
check('partial repair missing final label',partial[-1]['decision'],None)
ps=load(support/'position_summary.json')['cohorts']['ratio_15_all'];pg=ps['gate'];po=ps['point']
spread=max(r['auc'] for r in po['original'])-min(r['auc'] for r in po['original'])
benefit=min(r['auc'] for r in po['no_position'])-min(r['auc'] for r in po['original'])
cost=po['original'][0]['auc']-po['no_position'][0]['auc']
for key,v in [('spread',spread),('worst_position_benefit',benefit),('unshifted_cost',cost)]:check('position '+key,v,pg[key])
check('position primary gate',(spread>=.03 and benefit>=.03 and ps['worst_benefit_ci95'][0]>0 and cost<=.015),False)
result={'status':'PASS','checks':len(checks),'failed':0,'training':training,'grounding':{'n':len(g),'evaluable':len(ev),'V1':list(v1res),'V2':list(v2res),'gate_passed':False,'correct_box_mask_flips':[112,112],'correct_control_mask_flips':[5,112]},'AG_minus_A':{'n':6370,'difference_pp':100*d,'ci_pp':[100*lo,100*hi]},'boxes':boxresults,'other':{'selective_complete_calls':72,'selective_repair_calls':6,'manual_valid_empty':[18,18],'position':{'spread_pp':100*spread,'benefit_pp':100*benefit,'cost_pp':100*cost,'benefit_ci_pp':[100*x for x in ps['worst_benefit_ci95']],'gate_passed':False}},'check_ledger':checks,'limits':['Verifies stored inference records and recorded annotation JSON, not fresh model inference.','Does not establish binary anomaly transfer failure after retraining.','Historical all-split grounding gate is not a held-out test effect.','Position bootstrap interval is source-checked; AUC effect sizes are recomputed from archived aggregate scores, not rerun from all frame-level assets.']}
result['A_anomaly_rates']=arates
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','failed']},indent=2))
