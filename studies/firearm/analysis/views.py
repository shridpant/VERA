"""Report views: readable subsets of the full tables for the PDF. The CSVs keep every column."""
from .latex import esc, fmt
from .tables import table

CAT = {'Handgun': 'Handgun', 'Machine_Gun': 'Machine gun', 'No_Gun': 'No gun', 'All': 'All'}


def ci(lo, hi, column='x_pp'):
    return '--' if lo is None or hi is None else f'[{fmt(lo, column)}, {fmt(hi, column)}]'


def view(t, vid, title, spec, rows=None, notes=None):
    """spec: [(header, function(row) -> value)]."""
    rows = t['rows'] if rows is None else rows
    out = [{h: f(r) for h, f in spec} for r in rows]
    return table(vid, title, [h for h, _ in spec], out, notes if notes is not None else t['notes'])


def get(col, kind=None):
    return lambda r: fmt(r.get(col), kind or col)


def views(tables):
    v = []
    t = tables['T2']
    v.append(view(t, 'T2', t['title'], [
        ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('Sched.', get('scheduled')),
        ('Complete', get('complete')), ('Exhausted', get('exhausted')), ('Decode fail', get('preprocess_failed')),
        ('Pending', get('pending')), ('Failed att.', get('failed_attempts')), ('Timeouts', get('timeouts')),
        ('Valid labels', get('valid_labels')), ('Unparsed', get('unparsed_labels')), ('Truncated', get('truncated')),
        ('All slots answered', get('all_slots_answered')), ('Any malformed', get('any_malformed')),
        ('Refined videos', get('refined_videos')), ('Refinement excl.', get('refinement_excluded'))]))
    t = tables['T2b']
    v.append(view(t, 'T2b', t['title'], [
        ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('Answers', get('answers')),
        ('Final valid', get('final_valid')), ('Runner-strict valid', get('runner_strict_valid')),
        ('Final $\\neq$ runner', get('final_vs_runner_differ')), ('Final $\\neq$ legacy', get('final_vs_legacy_differ_where_final_valid')),
        ('Legacy 0 if final missing', get('legacy_0_where_final_missing')), ('Legacy 1 if final missing', get('legacy_1_where_final_missing'))]))
    t = tables['T3']
    v.append(view(t, 'T3', t['title'], [
        ('Category', lambda r: CAT[r['category']]), ('Scheduled', get('scheduled_windows')), ('Valid', get('valid_A_windows')),
        ('Anomaly', get('anomaly')), ('Anomaly %', get('anomaly_pct')), ('Normal', get('normal')), ('Normal %', get('normal_pct')),
        ('Not valid', get('not_valid')), ('Not valid % sched.', get('not_valid_pct_of_scheduled')),
        ('Videos', get('videos_with_valid')), ('Per-video mean %', get('per_video_anomaly_mean_pct')),
        ('Per-video median %', get('per_video_anomaly_median_pct'))]))
    t = tables['T4']
    v.append(view(t, 'T4', t['title'], [
        ('Category', lambda r: CAT[r['category']]), ('Paired windows', get('paired_windows')), ('Videos', get('paired_videos')),
        ('A %', get('A_anomaly_pct')), ('AG %', get('AG_anomaly_pct')), ('AG$-$A (pp)', get('diff_pp')),
        ('95\\% CI (videos)', lambda r: ci(r['diff_pp_ci_low'], r['diff_pp_ci_high'])),
        ('95\\% CI (groups)', lambda r: ci(r['diff_pp_ci_group_low'], r['diff_pp_ci_group_high'])),
        ('Equal-video (pp)', get('equal_video_diff_pp')), ('95\\% CI', lambda r: ci(r['equal_video_diff_pp_ci_low'], r['equal_video_diff_pp_ci_high'])),
        ('A % (all valid)', get('A_full_anomaly_pct')), ('AG % (all valid)', get('AG_full_anomaly_pct'))]))
    t = tables['T5']
    v.append(view(t, 'T5', t['title'], [
        ('Category', lambda r: CAT[r['category']]), ('Paired', get('paired')),
        ('N$\\to$N', lambda r: f"{fmt(r['normal_normal'])} ({fmt(r['normal_normal_pct'], 'pct')}\\%)"),
        ('N$\\to$A', lambda r: f"{fmt(r['normal_anomaly'])} ({fmt(r['normal_anomaly_pct'], 'pct')}\\%)"),
        ('A$\\to$N', lambda r: f"{fmt(r['anomaly_normal'])} ({fmt(r['anomaly_normal_pct'], 'pct')}\\%)"),
        ('A$\\to$A', lambda r: f"{fmt(r['anomaly_anomaly'])} ({fmt(r['anomaly_anomaly_pct'], 'pct')}\\%)"),
        ('Only A unparsed', get('A_only_unparsed')), ('Only AG unparsed', get('AG_only_unparsed')), ('Both', get('both_unparsed'))]))
    t = tables['T6']
    if 'eligible' in t['columns']:
        v.append(view(t, 'T6', t['title'], [
            ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('Videos', get('videos')), ('Eligible', get('eligible')),
            ('$<$15 windows', get('excluded_too_few_windows')), ('Invalid label', get('excluded_invalid_labels')),
            ('No features', get('excluded_missing_features')), ('Incomplete', get('excluded_incomplete')),
            ('Mean of means', get('mean_of_video_means')), ('Median', get('median_of_video_means')),
            ('IQR', lambda r: ci(r['iqr_low'], r['iqr_high'], 'x')), ('Median max', get('median_of_video_max')),
            ('Paired videos', lambda r: fmt(r.get('common_videos'))), ('AG$-$A mean', lambda r: fmt(r.get('paired_mean_diff'), 'x')),
            ('95\\% CI', lambda r: ci(r.get('paired_ci_low'), r.get('paired_ci_high'), 'x'))]))
    else:
        v.append(tables['T6'])
    t = tables['T7']
    v.append(view(t, 'T7', t['title'], [
        ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('Submitted', get('submitted_slots')),
        ('Boxes', get('boxes')), ('Empty', get('empty')), ('Unknown', get('unknown')), ('Malformed', get('malformed')),
        ('Missing', get('missing')), ('Duplicate', get('duplicate')), ('Truncated', get('truncated')),
        ('Boxes emitted', get('boxes_emitted')), ('Whole-image', get('whole_image_boxes')),
        ('Answers with boxes', get('responses_with_boxes')), ('Same box every slot', get('responses_same_boxes_every_boxed_slot')),
        ('Only whole-image', get('responses_only_whole_image')), ('Boxed frames', get('frames_positive')),
        ('Listed, no box', get('frames_listed_no_box')), ('No-gun frames', get('frames_negative_category'))]))
    t = tables['T8']
    main = [r for r in t['rows'] if r['origin'] == 'primary' and not r.get('size')]
    v.append(view(t, 'T8', t['title'], [
        ('IoU', get('iou', 'x')), ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]),
        ('Boxed slots', get('positive_slots')), ('GT', get('gt_objects')), ('No record', get('no_record_slots')),
        ('TP', get('tp')), ('FP (boxed)', get('fp_on_boxed_frames')), ('FN', get('fn')), ('Precision', get('precision')),
        ('Recall', get('recall')), ('F1', get('f1')), ('Recall (answered)', get('answered_recall')),
        ('Matched IoU (median)', get('matched_iou_median')), ('No-gun slots', get('negative_slots')),
        ('No-gun box rate', get('negative_box_rate'))], rows=main))
    diffs = [r for r in main if r['condition'] == 'AG']
    v.append(view(t, 'T8d', 'Paired AG minus G localization differences on slots executed in both conditions', [
        ('IoU', get('iou', 'x')), ('Category', lambda r: CAT[r['category']]), ('Paired slots', get('paired_executed_slots')),
        ('Recall diff.', lambda r: fmt(r.get('recall_diff_AG_minus_G'), 'x')),
        ('95\\% CI', lambda r: ci(r.get('recall_diff_ci_low'), r.get('recall_diff_ci_high'), 'x')),
        ('Precision diff.', lambda r: fmt(r.get('precision_diff_AG_minus_G'), 'x')),
        ('95\\% CI ', lambda r: ci(r.get('precision_diff_ci_low'), r.get('precision_diff_ci_high'), 'x')),
        ('No-gun box-rate diff.', lambda r: fmt(r.get('negative_box_rate_diff'), 'x')),
        ('95\\% CI  ', lambda r: ci(r.get('negative_box_rate_diff_ci_low'), r.get('negative_box_rate_diff_ci_high'), 'x'))], rows=diffs,
        notes=['Video-cluster paired bootstrap (2,000 draws, seed 20261025). Precision uses boxed frames of firearm videos; '
               'the No-gun box rate is the share of answered No_Gun slots with at least one box.']))
    sizes = [r for r in t['rows'] if r.get('size')]
    v.append(view(t, 'T8s', 'Recall at the first IoU threshold by ground-truth size (area tertiles)', [
        ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('Size', get('size')),
        ('GT', get('gt_objects')), ('TP', get('tp')), ('Recall', get('recall'))], rows=sizes, notes=['Size bins are tertiles of all annotated box areas.']))
    sens = [r for r in t['rows'] if r['origin'] == 'sensitivity' and not r.get('size')]
    v.append(view(t, 'T8o', 'Localization with ground truth re-mapped under origin 0 (sensitivity)', [
        ('IoU', get('iou', 'x')), ('Cond.', get('condition')), ('Category', lambda r: CAT[r['category']]), ('GT', get('gt_objects')),
        ('TP', get('tp')), ('Precision', get('precision')), ('Recall', get('recall')), ('F1', get('f1'))], rows=sens,
        notes=['image_id k mapped to decoded frame k instead of k-1; everything else identical.']))
    t = tables['T10']
    v.append(view(t, 'T10', t['title'], [
        ('Stage', get('stage')), ('Units', get('units')), ('Failed/interrupted', get('failed_or_interrupted_attempts')),
        ('Hours', get('total_hours')), ('Median s', get('median_s')), ('p95 s', get('p95_s')), ('Tokens', get('tokens_total')),
        ('Median tokens', get('tokens_median', 'x_pp')), ('p95 tokens', get('tokens_p95', 'x_pp')), ('Truncated', get('truncated')),
        ('Peak GPU GiB', get('peak_gpu_gib'))]))
    t = tables['T9']
    v.append(view(t, 'T9', t['title'], [
        ('Video', lambda r: esc(r['video'])), ('Cam.', get('camera')), ('Dur.\\ s', get('duration_s')), ('Windows', get('scheduled_windows')),
        ('A anomaly fraction', lambda r: fmt(r['A_anomaly_fraction'], 'x')), ('AG anomaly fraction', lambda r: fmt(r['AG_anomaly_fraction'], 'x')),
        ('A invalid', get('A_invalid')), ('AG invalid', get('AG_invalid')),
        ('A refined mean', lambda r: fmt(r['A_refined_mean'], 'x')), ('AG refined mean', lambda r: fmt(r['AG_refined_mean'], 'x'))]))
    t = tables['T1']
    v.append(view(t, 'T1', t['title'], [
        ('Category', lambda r: CAT[r['category']]), ('Camera', get('camera')), ('Resolution', get('resolution')), ('FPS', get('fps')),
        ('Videos', get('videos')), ('Minutes', get('duration_total_min')), ('Mean s', get('duration_mean_s')),
        ('Frames', get('frames')), ('Windows', get('windows'))]))
    return v
