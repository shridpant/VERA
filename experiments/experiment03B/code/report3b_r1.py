#!/usr/bin/env python3
"""Generate an R1 report and source-deduplicated descriptive sensitivity, CPU only."""
import argparse,csv,json,random,statistics,sys
from collections import defaultdict
from pathlib import Path
import audit3b_r1 as audit

def read(p):return json.loads(p.read_text())
def bootstrap(vals):
    rng=random.Random(0);n=len(vals)
    draws=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    def q(p):
        z=(len(draws)-1)*p;i=int(z);return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(z-i)
    return [q(.025),q(.975)]
def means(rows,key,group):
    d=defaultdict(list)
    for r in rows:
        if r.get(key) not in ('',None):d[group[r['video_id']]].append(float(r[key]))
    return [statistics.mean(x) for _,x in sorted(d.items())]
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--audit-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rd=a.run_dir;out=a.out;out.mkdir(parents=True,exist_ok=True)
    val=read(a.audit_dir/'validation_r1.json')
    if not val['passed'] or audit.hashes(rd)!=read(a.audit_dir/'source_sha256.json'):raise RuntimeError('Audit not passed or raw source changed')
    p=read(rd/'results/primary.json');sec=read(rd/'results/secondary.json');table=read(rd/'results/table_a1.json')
    inp=read(rd/'inputs.json')['inputs'];identity={r['video_id']:r['video_sha256'] for r in inp}
    vids=defaultdict(set)
    for vid,sha in identity.items():vids[sha].add(vid)
    duplicate_groups=[sorted(x) for x in vids.values() if len(x)>1]
    rows=[r for r in csv.DictReader((rd/'results/inputs_metrics.csv').open()) if r['development']=='False']
    sensitivity={'rule':'mean windows within source SHA256, then equal-weight sources; duplicate copies have identical windows/results','duplicate_groups':duplicate_groups,'metrics':{}}
    original={vid:vid for vid in identity}
    for k in ['d_w','e_w','overlap_blur_grey','tau_b_blur_grey','overlap_native_hi','tau_b_native_hi','nosound_changed','normal_0to1_grey_pairs','reciprocal_blur_effect_of_grey_top2']:
        old=means(rows,k,original);new=means(rows,k,identity)
        sensitivity['metrics'][k]={'original':statistics.mean(old),'deduplicated':statistics.mean(new),'n_original':len(old),'n_sources':len(new)}
        if k in ('d_w','e_w'):sensitivity['metrics'][k]['deduplicated_ci95']=bootstrap(new)
    trajectory=list(csv.DictReader((rd/'results/trajectories.csv').open()));uids={r['window_uid'] for r in rows}
    sensitivity['prefixes']={}
    for mode in ('fixed','adaptive'):
        for k in range(1,5):
            rr=[r for r in trajectory if r['window_uid'] in uids and r['mode']==mode and int(r['k'])==k]
            x=means(rr,'flip',identity)
            sensitivity['prefixes'][f'{mode}_k{k}']={'original':sec['2_fixed_vs_adaptive_grey_prefixes'][f'{mode}_k{k}']['label_change_rate_video_weighted'],'deduplicated':statistics.mean(x),'n_sources':len(x)}
    (a.audit_dir/'duplicate_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    e=p['estimate_pp'];lo,hi=p['ci95_pp'];ax[0].errorbar(e,0,xerr=[[e-lo],[hi-e]],fmt='o',capsize=6);ax[0].axvline(0,color='grey',ls='--');ax[0].set_yticks([]);ax[0].set_xlim(-2,25);ax[0].set_xlabel('Ranked-pair advantage (percentage points)');ax[0].set_title('Original primary result: unchanged')
    for mode in ('fixed','adaptive'):
        vals=[100*sec['2_fixed_vs_adaptive_grey_prefixes'][f'{mode}_k{k}']['label_change_rate_video_weighted'] for k in range(1,5)]
        ax[1].plot(range(1,5),vals,'o-',label=mode)
    ax[1].set_xticks(range(1,5));ax[1].set_xlabel('Frames replaced');ax[1].set_ylabel('Label-change rate (%)');ax[1].set_title('Secondary: adaptive uses extra calls');ax[1].legend()
    fig.savefig(out/'results.pdf');fig.savefig(out/'results.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.3),layout='constrained');ax.hist(list(p['video_means_pp'].values()),bins=18,color='#386b8d',edgecolor='white');ax.axvline(0,color='grey',ls='--');ax.set_xlabel('Video-level ranked-pair advantage (percentage points)');ax.set_ylabel('Videos');fig.savefig(out/'heterogeneity.pdf');plt.close(fig)
    rows_tex=[]
    for r in table['rows']:
        if not isinstance(r['paired_advantage_pp'],(int,float)):adv='NA'
        else:adv=f"{r['paired_advantage_pp']:.1f}"
        def fmt(x):return f'{x:.1f}' if isinstance(x,(int,float)) else 'NA'
        rows_tex.append(f"{esc(r['population'])} & {r['n_videos']} & {r['n_windows']} & {r['n_alerts']} & {fmt(r['top2_flip_pct'])} & {fmt(r['random_pair_flip_pct'])} & {adv}"+r' \\')
    failures={c['id']:c for c in val['checks']};mismatches=failures['score_rounding']['notes']['exact_rounding_mismatches'];excess=max(r['max_half_step_excess'] for r in mismatches)
    text=r'''\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,longtable,graphicx,hyperref}
\title{Experiment03B: Frame-Ranking Validation and Integrity Amendment R1}
\author{Research report for Shrid Pant}\date{October 26, 2026}
\begin{document}\maketitle
\begin{abstract}
Across the original VERA alert population, blur-ranked frame pairs removed alerts under grey replacement more often than the exhaustive pair reference. The original result is preserved. A separately versioned, post-data integrity audit resolves four validator-contract defects without rerunning inference or altering any original score, label, rank or result. This establishes the declared perturbation effect, not semantic faithfulness or improved detection.
\end{abstract}
\section{Design and coverage}
The frozen InternVL2-8B model answers the original VERA explanation-first prompt on eight sampled frames per window. A separate label-only scorer ranks frames by singleton blur effects. Evaluation replaces the top two with grey and compares the original VERA alert flip with the exact mean over all 28 pairs. Blur and grey are distribution-shifting interventions, not natural missingness.

The manifest covers 290 UCF-Crime test videos, 452 window records and 433 unique available inputs. Nineteen negative-window records are unavailable. Excluding the two development videos leaves 430 inputs from 288 video IDs. The primary population contains 199 alerts across 138 videos, with no missing primary effects. Event-centred windows use benchmark interval information and are oracle-localized. This is not a benchmark-wide detection evaluation.
\section{Primary result}
'''
    text+=f"The original video-weighted advantage is \\textbf{{{e:.3f} percentage points}}, 95\\% interval $[{lo:.3f},{hi:.3f}]$. The video-cluster bootstrap uses 10,000 draws and seed 0. Bootstrap stability seeds are numerical checks, not independent experiments.\n"
    text+=r'''\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{results.pdf}\caption{Original primary result and secondary prefix behavior. Adaptive ranking costs more queries; this is not an equal-compute comparison.}\end{figure}
\section{Prespecified descriptive table}
\small\begin{longtable}{p{5.3cm}rrrrrr}\toprule Population & Videos & Windows & Alerts & Top-2 (\%) & Random (\%) & Gain (pp)\\\midrule\endhead
'''+ '\n'.join(rows_tex)+r'''
\bottomrule\end{longtable}\normalsize
Category and stratum rows are descriptive, not separate confirmatory tests. Gains refer to alert removal, not accuracy improvements.
\section{Integrity amendment: original 16/20, R1 20/20}
The original validator and its failed artifact remain unchanged. R1 reruns the original validator and replaces only four checks with explicitly documented contracts; the other sixteen pass unchanged. All 36 local source-run files (31 portable record files) are SHA256-identical before and after the audit.
\begin{enumerate}
\item \textbf{Numerical readouts.} The old validator demanded exact BF16 round-to-nearest, while the inference engine's existing contract used half a BF16 step plus $10^{-3}$ slack. R1 requires finite, representable BF16 values, verifies that pre-existing bound and the exact stored score difference, and retains every exact-rounding mismatch. This does not claim exact rounding or change rankings.
\item \textbf{Slot-hash flags.} The legacy flag means exactly the masked slots changed. R1 verifies that flag against recorded baseline tensors, rather than requiring it always be true.
\item \textbf{No-op blur.} Two hash-verified source frames are all black. Resizing and applying Gaussian blur with radius 17.92 leaves their pixels unchanged. Twenty records therefore contain valid no-op replacements. R1 permits only source-verified black-frame blur no-ops; unchanged grey replacements or unexplained unchanged masks fail. Unmasked slots must remain identical and repeated replacements must have consistent hashes.
\item \textbf{Cache aliases.} Different video names may identify identical content. Shared keys are allowed only when source hash, frame indices, frame hashes, family, mask and tensor hashes agree. Archived Normal Videos 936 and 937 independently hash to the same SHA256. Eighty-eight shared sequence keys are explained this way; distinct content sharing a key still fails.
\end{enumerate}
'''
    text+=f"There are {len(mismatches)} exact-rounding mismatch records; the largest excess beyond half a step is {excess:.9f}, below the historical 0.001 slack.\n"
    text+=r'''\section{Duplicate-source sensitivity (post-data)}
The original analysis is retained. The sensitivity groups by source SHA256 and averages within source before averaging across sources. The duplicate pair is absent from the primary alert population: the primary point estimate is unchanged. Secondary population size falls from 288 IDs to 287 sources (430 rows contain 429 distinct window contents).
\begin{center}\begin{tabular}{lrrr}\toprule Metric & Original & Source-deduplicated & Sources\\\midrule
'''
    for k,label,scale in [('d_w','Primary advantage (pp)',100),('e_w','Numeric margin (nats)',1),('overlap_blur_grey','Blur/grey top-two overlap',1),('nosound_changed','No-sound changes (\%)',100)]:
        d=sensitivity['metrics'][k];text+=f"{label} & {d['original']*scale:.4f} & {d['deduplicated']*scale:.4f} & {d['n_sources']}"+r' \\'+'\n'
    text+=r'''\bottomrule\end{tabular}\end{center}
All stored sensitivity metrics and prefix rates are in duplicate\_sensitivity.json. These are disclosed post-data checks, not replacements for the prespecified estimate.
\begin{figure}[ht]\centering\includegraphics[width=.8\linewidth]{heterogeneity.pdf}\caption{Heterogeneity across the 138 primary video-level effects. A positive average does not imply each video's ranking is informative.}\end{figure}
\section{Scientific interpretation and limitations}
The result supports ranking utility for this local perturbation task. It does not establish that textual explanations identify human-valid evidence. The scorer and original VERA use different prompts; agreement must be measured. False alerts can also be fragile under deletion. The no-sound diagnostic changes labels but does not itself prove improvement. Heatmaps from prior feasibility work are not validated explanations. Source duplication is only one form of dependence; a source-hash audit cannot detect all related scenes or near-duplicate recordings.

This study targets local eight-frame judgments, not the ImageBind-refined timeline. Existing UCF test results have informed research decisions; future question learning needs disjoint development material and honest disclosure of evaluation reuse. No new question-learning experiment was launched.
\section{Reproduction and review}
The original frozen run is runs/full\_v1. The additive audit is audits/full\_v1\_r1; no model or GPU is needed. Run code/audit3b\_r1.py with the source run, a fresh audit directory, the original sparse frames, and experiment02 run\_002 for the precision-erratum check. Then run code/report3b\_r1.py against that audit. Exact commands, software versions, tests, hashes and amendment rationale are recorded in R1\_AMENDMENT.md and the audit JSON. Dataset images remain outside Git.\par
Review: \url{https://github.com/shridpant/VERA/pull/5}. The next experiment remains a decision for the researcher after reviewing this result.
\end{document}
'''
    (out/'main.tex').write_text(text)
    (out/'report_inputs_sha256.json').write_text(json.dumps({str(x):audit.v.sha256_file(x) for x in [rd/'results/primary.json',rd/'results/secondary.json',rd/'results/table_a1.json',a.audit_dir/'validation_r1.json',a.audit_dir/'duplicate_sensitivity.json']},indent=2)+'\n')
    print(json.dumps(sensitivity['metrics']['e_w']))
if __name__=='__main__':main()
