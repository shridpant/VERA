#!/usr/bin/env python3
"""Post-data integrity amendment R1. Never writes into the source run.

Recompute all original checks, then apply explicit revised checks to four known
contract errors. Raw scores, labels, rankings and original validation are immutable.
"""
import argparse, hashlib, json, math, platform, statistics, sys
from collections import defaultdict
from pathlib import Path
import validate3b as v


def spacing(x):
    return 0.0 if x == 0 else 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def readout_ok(hi, native):
    return (isinstance(hi, (float, int)) and isinstance(native, (float, int))
            and math.isfinite(hi) and math.isfinite(native)
            and v.bf16(native) == native
            and abs(hi-native) <= spacing(native)/2 + 1e-3)


def content_identity(inp, rec):
    # IDs are names, not pixel identities. Require both source and frame identities.
    return (inp['video_sha256'], tuple(inp['frame_indices']), tuple(inp['frame_sha256']),
            rec['family'], tuple(rec['mask']), tuple(rec['slot_tensor_sha256']))


def hashes(root, portable=True):
    return {str(p.relative_to(root)):v.sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file() and (not portable or (p.name != 'run.lock' and 'example_inputs' not in p.parts))}


def prove_noops(run, frames):
    from PIL import Image, ImageFilter
    refs,_=v._unmasked_reference(run)
    proofs={}
    for r in run.lines:
        if not v.is_success(r) or r['family'] != 'blur': continue
        for s in r['mask']:
            if r['slot_tensor_sha256'][s] != refs[r['window_uid']][0][s]: continue
            key=(r['window_uid'],s)
            if key in proofs: continue
            inp=run.inputs_by_uid[key[0]]
            path=frames/inp['video_id']/('%06d.jpg'%inp['frame_indices'][s])
            if v.sha256_file(path) != inp['frame_sha256'][s]: raise ValueError('Source frame hash mismatch')
            im=Image.open(path).convert('RGB')
            # Narrow proof: all-black images remain all black through the original
            # resize/black letterbox path and Gaussian blur. No general no-op waiver.
            if im.getextrema() != ((0,0),(0,0),(0,0)): raise ValueError('Unproven non-black no-op')
            resized=im.resize((448,448)); blurred=resized.filter(ImageFilter.GaussianBlur(17.92))
            if resized.tobytes()!=blurred.tobytes(): raise ValueError('Blur no-op not reproduced')
            proofs[key]={'window_uid':key[0],'slot':s,'frame_sha256':inp['frame_sha256'][s],
                         'extrema':im.getextrema(),'blur_radius':17.92,'size':im.size}
    return proofs


def revised_checks(run, proofs):
    score=v.Check('score_rounding','R1: finite representable BF16 readouts within half-step + historical 1e-3 tolerance; preserve exact-rounding mismatch count')
    present=v.Check('slot_hashes_present','R1: eight SHA256 hashes; legacy flag must equal strict identity recomputation')
    identity=v.Check('slot_identity','R1: unchanged unmasked slots; masked change except source-hash-verified black blur no-op; replacement consistency')
    cache=v.Check('mask_hash_consistency','R1: shared sequence key requires identical source, frames, transform and tensor hashes')
    refs,conflicts=v._unmasked_reference(run)
    for conflict in conflicts: identity.fail(problem='Conflicting baseline hashes',detail=conflict)
    exact_mismatch=[]; noop_lines=[]; replacements={}; groups={}; owners={}; aliases=[]
    for r in run.lines:
        if not v.is_success(r): continue
        uid=r['window_uid']; h=r['slot_tensor_sha256']; mask=r['mask']; fam=r['family']
        present.tick(); identity.tick(); cache.tick()
        if len(h)!=8 or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in h):
            present.fail(src=r['_src'],problem='Invalid hashes');continue
        ref=refs[uid][0]
        strict=all((h[s]!=ref[s])==(s in mask) for s in range(8))
        if r['slot_identity_ok'] is not strict: present.fail(src=r['_src'],problem='Incorrect legacy flag')
        noops=[]
        for s in range(8):
            if s not in mask and h[s]!=ref[s]: identity.fail(src=r['_src'],problem='Unmasked slot changed',slot=s)
            if s in mask:
                if h[s]==ref[s]:
                    if fam!='blur' or (uid,s) not in proofs: identity.fail(src=r['_src'],problem='Unproved no-op',slot=s)
                    else: noops.append(s)
                k=(uid,fam,s)
                if k in replacements and replacements[k]!=h[s]: identity.fail(src=r['_src'],problem='Accumulated or inconsistent replacement')
                replacements[k]=h[s]
        if noops: noop_lines.append({'src':r['_src'],'window_uid':uid,'slots':noops})
        k=(uid,fam,tuple(mask)); value=(tuple(h),r['sequence_key'])
        if k in groups and groups[k]!=value: cache.fail(src=r['_src'],problem='One input has different hashes or keys')
        groups[k]=value
        ident=content_identity(run.inputs_by_uid[uid],r); sk=r['sequence_key']
        if sk in owners:
            if owners[sk][0]!=ident: cache.fail(src=r['_src'],problem='Sequence collision across distinct source/pixels')
            elif owners[sk][1]!=uid: aliases.append(sk)
        else: owners[sk]=(ident,uid)
        if r['target']==v.SCORE:
            score.tick()
            if r.get('finite') is not True or r.get('S_hi_rounds_to_model_logits') is not True: score.fail(src=r['_src'],problem='False finite/legacy tolerance flag')
            for n in (0,1):
                if not readout_ok(r['hi_logit_'+str(n)],r['logit_'+str(n)]): score.fail(src=r['_src'],problem='Readout outside historical tolerance')
            if not math.isfinite(r['S']): score.fail(src=r['_src'],problem='Nonfinite S')
            delta=r['hi_logit_1']-r['hi_logit_0']
            if r['S_hi'] not in (delta,v.f32(delta)): score.fail(src=r['_src'],problem='Incorrect S_hi')
            if any(v.bf16(r['hi_logit_'+str(n)])!=r['logit_'+str(n)] for n in (0,1)):
                exact_mismatch.append({'src':r['_src'],'window_uid':uid,'max_half_step_excess':max(abs(r['hi_logit_'+str(n)]-r['logit_'+str(n)])-spacing(r['logit_'+str(n)])/2 for n in (0,1))})
    score.notes={'exact_rounding_mismatches':exact_mismatch,'historical_slack':1e-3}
    identity.notes={'verified_noop_records':noop_lines,'source_proofs':list(proofs.values())}
    cache.notes={'shared_sequence_keys_across_video_ids':len(set(aliases))}
    return [score,present,identity,cache]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frames-root',type=Path,required=True);ap.add_argument('--exp02-run-dir',type=Path,required=True)
    a=ap.parse_args();src=a.run_dir.resolve();out=a.out.resolve()
    if out==src or src in out.parents: raise ValueError('Audit output must be outside frozen run')
    out.mkdir(parents=True,exist_ok=False)
    before=hashes(src); full_before=hashes(src, portable=False)
    old=v.validate(str(src),str(a.exp02_run_dir))
    run=v.load_run(str(src));proofs=prove_noops(run,a.frames_root)
    revised=[x.to_json() for x in revised_checks(run,proofs)];ids={x['id'] for x in revised}
    checks=[x for x in old['checks'] if x['id'] not in ids]+revised
    after=hashes(src)
    if before!=after or full_before!=hashes(src, portable=False): raise RuntimeError('Frozen run changed during audit')
    result={'amendment':'R1 post-data integrity-only','original_summary':old['summary'],
            'summary':{s:sum(c['status']==s for c in checks) for s in ('passed','failed','skipped')},
            'passed':all(c['status']=='passed' for c in checks),'checks':checks,
            'original_run_unchanged':True,'recomputed_primary':old['recomputed']['primary'],
            'environment':{'python':sys.version,'platform':platform.platform()},
            'audit_code_sha256':v.sha256_file(__file__),'original_validator_sha256':v.sha256_file(v.__file__)}
    for name,obj in [('validation_r1.json',result),('original_validation_recomputed.json',old),('source_sha256.json',before),('source_allfiles_sha256.json',full_before)]:
        (out/name).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'summary':result['summary'],'source_files':len(before)}))
    return 0 if result['passed'] else 1
if __name__=='__main__': sys.exit(main())
