"""Build every table, figure and cited number from a finished run: `run_firearm_study.py analyze`."""
import json
from pathlib import Path
from . import plots, qualitative
from .latex import write_csv, write_numbers, write_tex
from .load import CATEGORIES, load
from .per_video import t9
from .tables import t1, t2, t2b, t3, t4, t5, t6, t7, t8, t10, table
from .views import views

T9_TEX = ['video', 'category', 'camera', 'duration_s', 'scheduled_windows', 'A_anomaly_fraction', 'AG_anomaly_fraction',
          'A_invalid', 'AG_invalid', 'A_refined_mean', 'AG_refined_mean']
T11_COLUMNS = ['case', 'reason', 'video', 'category', 'camera', 'anchor', 'time_s', 'window_choice', 'video_change_pp',
               'video_AG_recall', 'A_label', 'AG_label', 'A_text', 'AG_text', 'G_slots', 'AG_slots', 'boxed_frames']


def key(*parts):
    return '_'.join(str(p).replace('_', '').replace('.', 'p') for p in parts)


def numbers_from(tables, d, extra):
    n = {'videos': len(d.videos), 'windows': len(d.windows), 'units': d.reconcile['expected_units']}
    for subject in sorted({v['subject'] for v in d.videos}):
        n[key('subject', subject, 'videos')] = sum(v['subject'] == subject for v in d.videos)
    for disposition, count in d.reconcile['counts'].items():
        n[key('units', disposition)] = count
    for r in tables['T1']['rows']:
        if r['camera'] == 'all':
            n[key('T1', r['category'], 'videos')] = r['videos']
            n[key('T1', r['category'], 'windows')] = r['windows']
            n[key('T1', r['category'], 'duration_mean_s')] = r['duration_mean_s']
    for r in tables['T2']['rows']:
        for col in ('complete', 'exhausted', 'pending', 'preprocess_failed', 'valid_labels', 'unparsed_labels', 'truncated'):
            if col in r:
                n[key('T2', r['condition'], r['category'], col)] = r[col]
    for r in tables['T2b']['rows']:
        for col in ('answers', 'final_valid', 'final_vs_runner_differ', 'final_vs_legacy_differ_where_final_valid',
                    'legacy_0_where_final_missing', 'legacy_1_where_final_missing'):
            n[key('T2b', r['condition'], r['category'], col)] = r[col]
    for r in tables['T3']['rows']:
        for col in ('valid_A_windows', 'anomaly_pct', 'normal_pct', 'not_valid', 'per_video_anomaly_mean_pct', 'per_video_anomaly_median_pct'):
            n[key('T3', r['category'], col)] = r[col]
    for r in tables['T4']['rows']:
        for col in ('paired_windows', 'paired_videos', 'A_anomaly_pct', 'AG_anomaly_pct', 'diff_pp', 'diff_pp_ci_low', 'diff_pp_ci_high',
                    'diff_pp_ci_group_low', 'diff_pp_ci_group_high', 'equal_video_diff_pp', 'equal_video_diff_pp_ci_low',
                    'equal_video_diff_pp_ci_high', 'A_full_anomaly_pct', 'AG_full_anomaly_pct'):
            n[key('T4', r['category'], col)] = r[col]
    for r in tables['T5']['rows']:
        for col in ('normal_anomaly_pct', 'anomaly_normal_pct', 'A_only_unparsed', 'AG_only_unparsed', 'both_unparsed'):
            n[key('T5', r['category'], col)] = r[col]
    for r in tables['T6']['rows']:
        if 'eligible' in r:
            n[key('T6', r['condition'], r['category'], 'eligible')] = r['eligible']
            n[key('T6', r['condition'], r['category'], 'too_few')] = r['excluded_too_few_windows']
            if r['condition'] == 'AG':
                for col in ('paired_mean_diff', 'paired_ci_low', 'paired_ci_high', 'common_videos'):
                    n[key('T6', r['category'], col)] = r[col]
    for r in tables['T7']['rows']:
        for col in ('submitted_slots', 'boxes', 'empty', 'malformed', 'unknown', 'missing', 'duplicate', 'truncated', 'whole_image_boxes',
                    'boxes_emitted', 'responses', 'responses_with_boxes', 'responses_same_boxes_every_boxed_slot', 'responses_only_whole_image'):
            n[key('T7', r['condition'], r['category'], col)] = r[col]
    for r in tables['T8']['rows']:
        if not r.get('size'):
            for col in ('recall', 'precision', 'f1', 'tp', 'fp_on_boxed_frames', 'gt_objects', 'negative_slots', 'negative_box_rate',
                        'answered_recall', 'matched_iou_median', 'precision_incl_no_gun_frames', 'paired_executed_slots',
                        'recall_diff_AG_minus_G', 'recall_diff_ci_low', 'recall_diff_ci_high', 'precision_diff_AG_minus_G',
                        'negative_box_rate_diff', 'negative_box_rate_diff_ci_low', 'negative_box_rate_diff_ci_high'):
                if r.get(col) is not None or col in r:
                    n[key('T8', r['origin'], r['condition'], r['category'], 'iou' + str(int(r['iou'] * 100)), col)] = r.get(col)
    for r in tables['T10']['rows']:
        n[key('T10', r['stage'].split()[-1] if r['stage'].startswith('inference') else r['stage'].split()[0], 'hours')] = r.get('total_hours')
        if r['stage'].startswith('inference'):
            n[key('T10', r['stage'].split()[-1], 'median_s')] = r['median_s']
            n[key('T10', r['stage'].split()[-1], 'truncated')] = r['truncated']
    n.update(extra)
    return n


def analyze(study, inventory_path, original_root, out_dir, features_check=None):
    out = Path(out_dir)
    for sub in ('tables', 'figures'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    d = load(study, original_root)
    (out / 'prompts').mkdir(exist_ok=True)
    for c in ('A', 'G', 'AG'):
        (out / 'prompts' / f'{c}.txt').write_text((d.run_dir / 'prompts' / f'{c}.txt').read_text())
    tables = {t['id']: t for t in (t1(d), t2(d), t2b(d), t3(d), t4(d), t5(d), t6(d), t7(d), t8(d), t9(d), t10(d))}
    cases, case_figures = qualitative.build(d, inventory_path, original_root, out)
    tables['T11'] = table('T11', 'Qualitative cases (selection rule in the text)', T11_COLUMNS, cases,
                          ['Texts are the complete model answers; slot summaries list each frame\'s parse status and boxes.'])
    for t in tables.values():
        write_csv(t, out / 'tables')
    write_tex(tables['T9'], out / 'tables', columns=T9_TEX, longtable=True)
    for v in views(tables):  # readable subsets for the PDF; the CSVs keep every column
        write_tex(v, out / 'tables', raw=True)
    figures = [plots.p1(d, out / 'figures'), plots.p2(d, out / 'figures'), plots.p3(d, out / 'figures'),
               plots.p4(d, out / 'figures'), plots.p5(d, out / 'figures'), plots.p6(d, out / 'figures'),
               plots.p7(d, tables['T8']['rows'], out / 'figures'), plots.p9(d, out / 'figures')]
    extra = {'cases': len(cases)}
    if features_check and Path(features_check).exists():
        fc = json.loads(Path(features_check).read_text())
        extra.update({'fc_cosine_median': fc.get('cosine_median'), 'fc_cosine_min': fc.get('cosine_min'),
                      'fc_refined_identical': fc.get('refined_frame_scores_identical'), 'fc_refined_maxdiff': fc.get('refined_frame_scores_max_abs_diff')})
    write_numbers(numbers_from(tables, d, extra), out)
    (out / 'qualitative_cases.tex').write_text(qualitative_tex(cases))
    manifest = {'tables': sorted(tables), 'figures': [f for pair in figures for f in pair] + case_figures,
                'run_dir': str(d.run_dir), 'spec_sha256': json.loads((d.run_dir / 'spec.json').read_text())['spec_sha256'],
                'model_sha256': d.reconcile['model_sha256'], 'categories': list(CATEGORIES)}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def qualitative_tex(cases):
    from .latex import esc
    parts = []
    for c in cases:
        parts.append(f"\\paragraph{{Case {c['case']} ({esc(c['reason'])}): {esc(c['video'])}, window at {c['time_s']} s}}\n"
                     f"Window choice: {esc(c['window_choice'])}. A label {esc(c['A_label'])}, AG label {esc(c['AG_label'])}; "
                     f"boxed source frames {c['boxed_frames']}.\\\\\n"
                     f"\\textit{{A:}} {esc(c['A_text'] or '(no complete answer)')}\\\\\n"
                     f"\\textit{{AG:}} {esc(c['AG_text'] or '(no complete answer)')}\\\\\n"
                     f"\\textit{{G slots:}} {{\\scriptsize {esc(c['G_slots'])}}}\\\\\n"
                     f"\\textit{{AG slots:}} {{\\scriptsize {esc(c['AG_slots'])}}}\n\n"
                     f"\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\textwidth]{{figures/P8_case{c['case']}.pdf}}"
                     f"\\caption{{Case {c['case']} ({esc(c['reason'])}): A/AG timeline and the exact inputs of the marked window.}}"
                     f"\\label{{fig:P8_case{c['case']}}}\\end{{figure}}\n")
    return '\n'.join(parts) if parts else 'No case met the selection rule.\n'
