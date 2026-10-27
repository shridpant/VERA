"""Read-only FIT format triage; run with the immutable gate release on sys.path."""
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); release=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(release))
from studies.firearm_posttraining.rewards import parse,parse_labels
book=json.loads((release/'studies/firearm_training/codebook.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
failures=[]
trace=root/'format-g-009/responses.jsonl'
for index,line in enumerate(trace.read_text().splitlines()):
    answer=json.loads(line)
    valid=parse(answer['text'],book,True,answer['truncated']) is not None
    assert valid == answer['valid'], 'saved parser result disagrees with immutable release'
    if valid: continue
    value=json.loads(answer['text'])
    boxes=[(f['frame_id'],b) for f in value['grounding'] for b in f['boxes']]
    scale=[{'frame_id':k,'box':b} for k,b in boxes if any(x<0 or x>1 for x in b)]
    degenerate=[{'frame_id':k,'box':b} for k,b in boxes if b[0]>=b[2] or b[1]>=b[3]]
    failures.append({'response_index_zero_based':index,'video':answer['video'],'sample':answer['sample'],
        'text':answer['text'],'truncated':answer['truncated'],
        'labels_valid':parse_labels(answer['text'],book,answer['truncated']) is not None,
        'unit_range_violations':scale,'degenerate_boxes':degenerate,
        'all_coordinates_in_native_range':all(0<=x<=1000 for _,b in boxes for x in b)})
ledger=json.loads((root/'budget.json').read_text())
settled=sum(v.get('elapsed_seconds',0) for v in ledger.values())
paths=[root/'format-l-009/result.json',root/'format-g-009/result.json',trace,root/'budget.json',root/'calibration-probability-v3/status.json']
result={'status':'failed_format_gate_preserved','release_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=release,text=True).strip(),
 'source_artifacts':{str(p):sha(p) for p in paths},
 'gate_results':{k:json.loads((root/f'format-{k}-009/result.json').read_text()) for k in ('l','g')},
 'controller_status':json.loads((root/'calibration-probability-v3/status.json').read_text()),
 'controller_alive':pathlib.Path('/proc/293624').exists(),
 'pilot_settled_gpu_seconds':settled,'pilot_limit_gpu_seconds':21600,'pilot_remaining_gpu_seconds':21600-settled,
 'active_budget_reservations':{k:v for k,v in ledger.items() if v['status']!='finished'},
 'protocol_exists':(root/'protocol.json').exists(),'selection_exists':(root/'selection.json').exists(),
 'failed_grounded_responses':failures,
 'triage_summary':{'invalid':len(failures),'valid_json_untruncated':sum(not f['truncated'] for f in failures),
 'unit_range_violations':sum(bool(f['unit_range_violations']) for f in failures),
 'degenerate_geometry':sum(bool(f['degenerate_boxes']) for f in failures)},
 'interpretation':'FIT-only diagnostic, not a rescued gate or evidence that changing representation will pass. Partial 5/65 is not a full-panel error estimate.'}
print(json.dumps(result,indent=2))
