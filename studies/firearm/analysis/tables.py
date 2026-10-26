"""T1-T10 as row lists (T11 lives in qualitative.py). Each table carries its own definitions as notes.

Denominators (handoff section 7): anomaly percentages use VALID windows; invalid rates use all
SCHEDULED windows; paired A-vs-AG statistics use windows valid in both. Pooled-window and
equal-video summaries are reported separately. Categories are dataset classes, not anomaly
truth: no accuracy, AUC or AP is computed anywhere.
"""
import numpy as np
from vera_runner.localization import prf, score_instance, size_bin
from .load import CATEGORIES
from .stats import cluster_bootstrap_sums, pct, quantiles, share

CATS = CATEGORIES + ('All',)
STATUSES = ('boxes', 'empty', 'unknown', 'malformed', 'missing', 'duplicate', 'truncated', 'no_record')


def subset(rows, cat):
    return rows if cat == 'All' else [r for r in rows if r['category'] == cat]


def valid(label):
    return label in (0, 1)


def table(tid, title, columns, rows, notes):
    return {'id': tid, 'title': title, 'columns': columns, 'rows': rows, 'notes': notes}


def t1(d):
    rows = []
    for cat in CATS:
        for cam in ('C1', 'C2', 'all'):
            vs = [v for v in subset(d.videos, cat) if cam == 'all' or v['camera'] == cam]
            if not vs:
                continue
            rows.append({'category': cat, 'camera': cam, 'resolution': '/'.join(sorted({v['resolution'] for v in vs})),
                         'fps': '/'.join(f'{x:g}' for x in sorted({v['fps'] for v in vs})), 'videos': len(vs),
                         'duration_total_min': sum(v['duration_s'] for v in vs) / 60, 'duration_mean_s': share(v['duration_s'] for v in vs),
                         'frames': sum(v['frames'] for v in vs), 'windows': sum(v['windows'] for v in vs)})
    return table('T1', 'Dataset composition', list(rows[0]), rows, [
        'C1: IP camera (CCTV); C2: phone. Durations use each video\'s own frame rate; windows = ceil(frames/16).'])


def t2(d):
    units = d.reconcile['units']
    ends, starts = {}, {}
    for e in d.ledger:
        if e['event'] == 'end':
            ends.setdefault(e['unit'].rsplit('|', 1)[1], []).append(e)
        elif e['event'] == 'start':
            starts.setdefault(e['unit'].rsplit('|', 1)[1], []).append(e)
    rows = []
    for c in ('A', 'G', 'AG'):
        for cat in CATS:
            us = [u for u in subset(units, cat) if u['condition'] == c]
            ws = subset(d.windows, cat)
            vids = {v['id'] for v in subset(d.videos, cat)}
            ev = [e for e in ends.get(c, []) if e['unit'].split('|')[0] in vids]
            st = [e for e in starts.get(c, []) if e['unit'].split('|')[0] in vids]
            ended = {(e['unit'], e['attempt']) for e in ev}
            row = {'condition': c, 'category': cat, 'scheduled': len(us),
                   'complete': sum(u['disposition'] == 'complete' for u in us),
                   'exhausted': sum(u['disposition'] == 'exhausted' for u in us),
                   'preprocess_failed': sum(u['disposition'] == 'preprocess_failed' for u in us),
                   'pending': sum(u['disposition'] == 'pending' for u in us),
                   'videos_complete': sum(all(u['disposition'] == 'complete' for u in us if u['video'] == v) for v in vids),
                   'failed_attempts': sum(e['status'] == 'failed' for e in ev),
                   'timeouts': sum(e['status'] == 'timeout' for e in ev),
                   'interrupted': sum(e['status'] == 'interrupted' for e in ev),
                   'decode_failures': sum(e['status'] == 'preprocess_failed' for e in ev),
                   'starts_without_end': sum((e['unit'], e['attempt']) not in ended for e in st)}
            if c in ('A', 'AG'):
                row.update(valid_labels=sum(valid(w[f'{c}_label']) for w in ws),
                           unparsed_labels=sum(w[f'{c}_complete'] and not w[f'{c}_truncated'] and not valid(w[f'{c}_label']) for w in ws),
                           truncated=sum(bool(w[f'{c}_truncated']) for w in ws))
                ref = [r for r in d.refinement if r['condition'] == c and (cat == 'All' or r['category'] == cat)]
                row.update(refined_videos=sum(r['refined'] == 'True' for r in ref) if ref else None,
                           refinement_excluded=sum(r['refined'] != 'True' for r in ref) if ref else None)
            if c in ('G', 'AG'):
                ss = [s for s in subset(d.slots, cat) if s['condition'] == c and s['complete']]
                by_window = {}
                for s in ss:
                    by_window.setdefault((s['video'], s['anchor']), []).append(s['status'])
                row.update(all_slots_answered=sum(all(x in ('boxes', 'empty') for x in v) for v in by_window.values()),
                           any_malformed=sum('malformed' in v for v in by_window.values()),
                           unknown_slots=sum(s['status'] == 'unknown' for s in ss))
                if c == 'G':
                    row['truncated'] = sum(set(v) == {'truncated'} for v in by_window.values())
            rows.append(row)
    columns = ['condition', 'category', 'scheduled', 'complete', 'exhausted', 'preprocess_failed', 'pending', 'videos_complete',
               'failed_attempts', 'timeouts', 'interrupted', 'decode_failures', 'starts_without_end', 'valid_labels', 'unparsed_labels',
               'truncated', 'all_slots_answered', 'any_malformed', 'unknown_slots', 'refined_videos', 'refinement_excluded']
    return table('T2', 'Coverage by condition and category', columns, rows, [
        'Units are windows. complete/exhausted/preprocess_failed/pending are reconcile dispositions; the attempt columns count ledger end events '
        '(starts_without_end: attempts killed before any end event).',
        'unparsed_labels: complete, not truncated, but no strict 0/1 in the final Output field. Truncated answers are counted separately; '
        'T3 not_valid = unparsed + truncated + no complete record.',
        'all_slots_answered: every FrameK line parsed to boxes or []; any_malformed: at least one malformed line.'])


def t2b(d):
    """Parser agreement on complete, non-truncated answers (handoff section 5)."""
    rows = []
    for c in ('A', 'AG'):
        for cat in CATS:
            ws = [w for w in subset(d.windows, cat) if w[f'{c}_complete'] and not w[f'{c}_truncated']]
            final, runner, legacy = ([w[f'{c}_{k}'] for w in ws] for k in ('label', 'strict_label', 'upstream_label'))
            rows.append({'condition': c, 'category': cat, 'answers': len(ws), 'final_valid': sum(valid(x) for x in final),
                         'runner_strict_valid': sum(valid(x) for x in runner),
                         'final_vs_runner_differ': sum(a != b for a, b in zip(final, runner)),
                         'final_vs_legacy_differ_where_final_valid': sum(valid(a) and a != b for a, b in zip(final, legacy)),
                         'legacy_0_where_final_missing': sum(not valid(a) and b == 0 for a, b in zip(final, legacy)),
                         'legacy_1_where_final_missing': sum(not valid(a) and b == 1 for a, b in zip(final, legacy))})
    return table('T2b', 'Agreement of the strict, runner-strict and original (legacy) label parsers', list(rows[0]), rows, [
        'final: the last Output field must be exactly 0/1 (used everywhere); runner strict: exactly one Output field; legacy: VERA\'s original '
        'rule (0 if any 0 follows the last Output). Truncated answers and missing records are excluded here.',
        'legacy_*_where_final_missing: what the original parser would have reported for answers the strict parser rejects.'])


def t3(d):
    rows = []
    for cat in CATS:
        ws = subset(d.windows, cat)
        good = [w for w in ws if valid(w['A_label'])]
        per_video = {}
        for w in good:
            per_video.setdefault(w['video'], []).append(w['A_label'])
        fractions = [share(v) for v in per_video.values()]
        n1 = sum(w['A_label'] == 1 for w in good)
        rows.append({'category': cat, 'scheduled_windows': len(ws), 'valid_A_windows': len(good), 'anomaly': n1,
                     'anomaly_pct': pct(n1 / len(good)) if good else None, 'normal': len(good) - n1,
                     'normal_pct': pct((len(good) - n1) / len(good)) if good else None, 'not_valid': len(ws) - len(good),
                     'not_valid_pct_of_scheduled': pct((len(ws) - len(good)) / len(ws)) if ws else None,
                     'unparsed': sum(w['A_complete'] and not w['A_truncated'] and not valid(w['A_label']) for w in ws),
                     'truncated': sum(bool(w['A_truncated']) for w in ws), 'no_record': sum(not w['A_complete'] for w in ws),
                     'videos_with_valid': len(per_video), 'per_video_anomaly_mean_pct': pct(share(fractions)),
                     'per_video_anomaly_median_pct': pct(float(np.median(fractions))) if fractions else None})
    return table('T3', 'Original VERA (A) anomaly judgments by dataset category', list(rows[0]), rows, [
        'Not a confusion matrix: categories are dataset classes, not anomaly ground truth.',
        'Anomaly/normal percentages use valid A windows; the not-valid percentage uses all scheduled windows (unparsed + truncated + no record).',
        'Per-video fractions: anomaly windows / valid windows of each video with at least one valid window.'])


def paired_windows(d, cat):
    return [w for w in subset(d.windows, cat) if valid(w['A_label']) and valid(w['AG_label'])]


def pooled_diff(ws):
    return (sum(w['AG_label'] for w in ws) - sum(w['A_label'] for w in ws)) / len(ws) if ws else None


def equal_video_diff(ws):
    per = {}
    for w in ws:
        per.setdefault(w['video'], []).append(w['AG_label'] - w['A_label'])
    return share(share(v) for v in per.values()) if per else None


def t4(d):
    B, seed, level = d.study['analysis']['bootstrap_resamples'], d.study['analysis']['bootstrap_seed'], d.study['analysis']['ci_level']
    groups = {v['id']: v['recording_group'] for v in d.videos}
    rows = []
    for cat in CATS:
        ws = [{**w, 'group': groups[w['video']]} for w in paired_windows(d, cat)]
        diff = lambda w: (w['AG_label'] - w['A_label'], 1)
        ratio = lambda t: t[:, 0] / t[:, 1]
        est, lo, hi, _ = cluster_bootstrap_sums(ws, 'video', 'category', diff, ratio, B, seed, f'T4 pooled {cat}', level)
        _, glo, ghi, _ = cluster_bootstrap_sums(ws, 'group', 'category', diff, ratio, B, seed, f'T4 pooled group {cat}', level)
        per_video = {}
        for w in ws:
            per_video.setdefault((w['video'], w['category']), []).append(w['AG_label'] - w['A_label'])
        videos = [{'video': v, 'category': c, 'mean': share(x)} for (v, c), x in per_video.items()]
        eq, elo, ehi, _ = cluster_bootstrap_sums(videos, 'video', 'category', lambda u: (u['mean'], 1), ratio, B, seed, f'T4 equal {cat}', level)
        all_ws = subset(d.windows, cat)
        a_valid = [w['A_label'] for w in all_ws if valid(w['A_label'])]
        ag_valid = [w['AG_label'] for w in all_ws if valid(w['AG_label'])]
        rows.append({'category': cat, 'paired_windows': len(ws), 'paired_videos': len({w['video'] for w in ws}),
                     'A_anomaly_pct': pct(share(w['A_label'] for w in ws)), 'AG_anomaly_pct': pct(share(w['AG_label'] for w in ws)),
                     'diff_pp': pct(est), 'diff_pp_ci_low': pct(lo), 'diff_pp_ci_high': pct(hi),
                     'diff_pp_ci_group_low': pct(glo), 'diff_pp_ci_group_high': pct(ghi),
                     'equal_video_diff_pp': pct(eq), 'equal_video_diff_pp_ci_low': pct(elo), 'equal_video_diff_pp_ci_high': pct(ehi),
                     'A_valid_windows': len(a_valid), 'A_full_anomaly_pct': pct(share(a_valid)),
                     'AG_valid_windows': len(ag_valid), 'AG_full_anomaly_pct': pct(share(ag_valid))})
    return table('T4', 'A versus AG anomaly rates on identical paired windows', list(rows[0]), rows, [
        f'Paired support: windows with a valid strict label in both A and AG. Differences are AG minus A in percentage points.',
        f'{int(level * 100)} % percentile intervals from {B} paired bootstrap draws resampling source videos within category '
        f'(seed {seed}); diff_pp_ci_group resamples recording groups (repeated takes) instead.',
        'equal_video: mean over videos of the per-video difference on paired windows. The full-coverage columns use each condition\'s own valid windows.',
        'A change in anomaly rate is not evidence of improvement: there is no anomaly ground truth.'])


def t5(d):
    rows = []
    for cat in CATS:
        ws = subset(d.windows, cat)
        paired = [w for w in ws if valid(w['A_label']) and valid(w['AG_label'])]
        n = len(paired)
        cell = lambda a, b: sum(w['A_label'] == a and w['AG_label'] == b for w in paired)
        row = {'category': cat, 'paired': n}
        for name, a, b in (('normal_normal', 0, 0), ('normal_anomaly', 0, 1), ('anomaly_normal', 1, 0), ('anomaly_anomaly', 1, 1)):
            row[name] = cell(a, b)
            row[name + '_pct'] = pct(cell(a, b) / n) if n else None
        row.update(A_only_unparsed=sum(not valid(w['A_label']) and valid(w['AG_label']) for w in ws),
                   AG_only_unparsed=sum(valid(w['A_label']) and not valid(w['AG_label']) for w in ws),
                   both_unparsed=sum(not valid(w['A_label']) and not valid(w['AG_label']) for w in ws))
        rows.append(row)
    return table('T5', 'Window-level prediction transitions from A to AG', list(rows[0]), rows, [
        'Rows read A to AG (e.g. normal_anomaly: A normal, AG anomaly). Percentages are of paired valid windows.',
        'Unparsed pairs (no valid label in A, AG or both, including truncation and missing records) are counted separately.'])


def refined_rows(d):
    out = {}
    for r in d.refinement:
        if r['refined'] == 'True':
            out[(r['condition'], r['video'])] = {'category': r['category'], 'mean': float(r['mean_frame_score']),
                                                 'max': float(r['max_frame_score'])}
    return out


def t6(d):
    B, seed, level = d.study['analysis']['bootstrap_resamples'], d.study['analysis']['bootstrap_seed'], d.study['analysis']['ci_level']
    if not d.refinement:
        return table('T6', 'Refined scores', ['note'], [{'note': 'not estimable: no refinement output'}], [])
    ref = refined_rows(d)
    rows = []
    for c in ('A', 'AG'):
        for cat in CATS:
            rs = [r for r in d.refinement if r['condition'] == c and (cat == 'All' or r['category'] == cat)]
            means = [ref[(c, r['video'])]['mean'] for r in rs if (c, r['video']) in ref]
            maxes = [ref[(c, r['video'])]['max'] for r in rs if (c, r['video']) in ref]
            q25, med, q75 = quantiles(means)
            reasons = {}
            for r in rs:
                for reason in filter(None, r['reasons'].split(';')):
                    reasons[reason] = reasons.get(reason, 0) + 1
            row = {'condition': c, 'category': cat, 'videos': len(rs), 'eligible': len(means),
                   'excluded_too_few_windows': reasons.get('too_few_windows', 0), 'excluded_invalid_labels': reasons.get('invalid_labels', 0),
                   'excluded_missing_features': reasons.get('missing_features', 0), 'excluded_incomplete': reasons.get('incomplete_inference', 0),
                   'mean_of_video_means': share(means), 'median_of_video_means': med, 'iqr_low': q25, 'iqr_high': q75,
                   'median_of_video_max': float(np.median(maxes)) if maxes else None}
            if c == 'AG':
                common = [{'video': r['video'], 'category': r['category'], 'diff': ref[('AG', r['video'])]['mean'] - ref[('A', r['video'])]['mean']}
                          for r in rs if ('AG', r['video']) in ref and ('A', r['video']) in ref]
                est, lo, hi, _ = cluster_bootstrap_sums(common, 'video', 'category', lambda u: (u['diff'], 1), lambda t: t[:, 0] / t[:, 1],
                                                        B, seed, f'T6 {cat}', level)
                row.update(common_videos=len(common), paired_mean_diff=est, paired_ci_low=lo, paired_ci_high=hi,
                           paired_median_diff=float(np.median([x['diff'] for x in common])) if common else None)
            rows.append(row)
    columns = list(rows[-1])
    return table('T6', 'Refined frame scores (VERA refinement, identical ImageBind features for A and AG)', columns, rows, [
        'Per-video mean and maximum of the refined frame scores (scene + temporal refinement and positional weighting as in compute_auc.py).',
        'Videos are excluded, never padded, when they have fewer than 15 windows, any invalid label, missing features or missing records; a video can have several reasons.',
        'Paired columns (AG rows): AG minus A per-video mean frame score on videos eligible in both conditions, with a video bootstrap interval.'])


def t7(d):
    rows = []
    for c in ('G', 'AG'):
        for cat in CATS:
            ss = [s for s in subset(d.slots, cat) if s['condition'] == c]
            done = [s for s in ss if s['complete']]
            row = {'condition': c, 'category': cat, 'scheduled_slots': len(ss), 'submitted_slots': len(done)}
            for st in STATUSES[:-1]:
                row[st] = sum(s['status'] == st for s in done)
            responses = {}
            for s in done:
                responses.setdefault((s['video'], s['anchor']), []).append(s)
            boxed = [[s for s in r if s['status'] == 'boxes'] for r in responses.values()]
            row.update(boxes_emitted=sum(len(s['boxes']) for s in done), whole_image_boxes=sum(sum(s['whole_image']) for s in done),
                       unit_scale_slots=sum(bool(s['scale_suspect']) and s['status'] == 'boxes' for s in done),
                       responses=len(responses), responses_with_boxes=sum(bool(b) for b in boxed),
                       responses_same_boxes_every_boxed_slot=sum(len(b) >= 2 and all(s['boxes'] == b[0]['boxes'] for s in b) for b in boxed),
                       responses_only_whole_image=sum(bool(b) and all(all(s['whole_image']) for s in b) for b in boxed))
            for fs in ('positive', 'listed_no_box', 'negative_category', 'unknown'):
                row['frames_' + fs] = sum(s['frame_status'] == fs for s in ss)
            rows.append(row)
    return table('T7', 'Grounding outcomes per window slot', list(rows[0]), rows, [
        'Submitted slots are the eight slots of every complete response. Status counts partition them.',
        'whole_image_boxes: boxes covering at least 95 % of the frame (kept in the metrics, reported here). '
        'responses_same_boxes_every_boxed_slot and responses_only_whole_image flag answers that do not vary by frame '
        '(a sign that slot references or per-frame coordinates are not being used).',
        'frames_*: the existing annotation coverage of the scheduled slots\' source frames (audit statuses).'])


def instances(d, origin_key='gt'):
    """{(video, anchor, slot): {'G': slot, 'AG': slot}} for slots whose frame supports evaluation."""
    out = {}
    for s in d.slots:
        positive = bool(s[origin_key]) if origin_key != 'gt' else s['frame_status'] == 'positive'
        negative = s['frame_status'] == 'negative_category'
        if positive or negative:
            out.setdefault((s['video'], s['anchor'], s['slot']), {})[s['condition']] = {**s, 'eval_gt': s[origin_key] if positive else []}
    return out


def _sum(us, c, field, positive=None):
    return sum((u[c][field] or 0) for u in us if (positive is None or u['positive'] == positive) and (field != 'fp' or u[c]['answered']))


def precision_of(us, c):
    """Precision on boxed (positive) frames only; None where no prediction was made there."""
    tp, fp = _sum(us, c, 'tp', True), _sum(us, c, 'fp', True)
    return tp / (tp + fp) if tp + fp else None


def negative_box_rate(us, c):
    neg = [u for u in us if not u['positive'] and u[c]['answered']]
    return sum(bool(u[c]['fp']) for u in neg) / len(neg) if neg else None


def t8(d):
    B, seed, level = d.study['analysis']['bootstrap_resamples'], d.study['analysis']['bootstrap_seed'], d.study['analysis']['ci_level']
    edges = d.policy['size_bins_area_fraction']
    rows = []
    for origin_name, key in (('primary', 'gt'), ('sensitivity', 'gt_sensitivity')):
        inst = instances(d, key)
        for thr in d.study['matching']['iou_thresholds']:
            units = []
            for (video, anchor, slot), pair in inst.items():
                u = {'video': video, 'category': pair['G']['category'], 'positive': bool(pair['G']['eval_gt']),
                     'executed_both': pair['G']['complete'] and pair['AG']['complete']}
                for c in ('G', 'AG'):
                    s = pair[c]
                    u[c] = score_instance(s['status'], s['boxes'], s['eval_gt'], thr, True)
                    u[c + '_no_record'] = not s['complete']
                    u[c + '_matched_sizes'] = [size_bin(s['eval_gt'][j], edges) for j in range(len(s['eval_gt'])) if j not in u[c]['missed_gt']]
                u['gt'] = len(pair['G']['eval_gt'])
                u['sizes'] = [size_bin(b, edges) for b in pair['G']['eval_gt']]
                units.append(u)
            for cat in CATS:
                us = subset(units, cat)
                firearm = [u for u in us if u['category'] != 'No_Gun']
                paired = [u for u in us if u['executed_both']]
                gt = sum(u['gt'] for u in us)
                no_truth = cat == 'No_Gun'
                for c in ('G', 'AG'):
                    tp, fn = _sum(us, c, 'tp'), _sum(us, c, 'fn')
                    fp_pos = _sum(us, c, 'fp', True)
                    recall = tp / gt if gt else None
                    precision = precision_of(firearm, c)
                    f1 = (2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall
                          else (0.0 if precision is not None and recall is not None else None))
                    neg_tp_fp = _sum(us, c, 'fp', False)
                    with_neg = (tp / (tp + fp_pos + neg_tp_fp)) if tp + fp_pos + neg_tp_fp else None
                    ious = [v for u in us for v in u[c]['matched_ious']]
                    answered_pos = [u for u in us if u['positive'] and u[c]['answered']]
                    row = {'origin': origin_name, 'iou': thr, 'condition': c, 'category': cat,
                           'evaluable_slots': len(us), 'positive_slots': sum(u['positive'] for u in us), 'gt_objects': gt,
                           'no_record_slots': sum(u[c + '_no_record'] for u in us), 'tp': tp, 'fp_on_boxed_frames': fp_pos, 'fn': fn,
                           'precision': None if no_truth else precision, 'recall': None if no_truth else recall, 'f1': None if no_truth else f1,
                           'precision_incl_no_gun_frames': with_neg if cat == 'All' else None,
                           'answered_recall': (sum(u[c]['tp'] for u in answered_pos) / sum(u['gt'] for u in answered_pos)) if answered_pos else None,
                           'matched_iou_mean': share(ious), 'matched_iou_median': float(np.median(ious)) if ious else None,
                           'negative_slots': sum(not u['positive'] for u in us), 'negative_box_rate': negative_box_rate(us, c),
                           'na_reason': 'no ground-truth firearms: precision and recall undefined; boxes on No_Gun frames are false positives' if no_truth else ''}
                    if c == 'AG':
                        row['paired_executed_slots'] = len(paired)
                        name = f'T8 {origin_name} {thr} {cat}'
                        firearm_paired = [u for u in paired if u['category'] != 'No_Gun']
                        if not no_truth:
                            recall_v = lambda u: (u['AG']['tp'] - u['G']['tp'], u['gt'])
                            prec_v = lambda u: ((u['AG']['tp'], u['AG']['fp'] or 0, u['G']['tp'], u['G']['fp'] or 0) if u['positive'] else (0, 0, 0, 0))
                            for label, vals, stat in (('recall', recall_v, lambda t: t[:, 0] / t[:, 1]),
                                                      ('precision', prec_v, lambda t: t[:, 0] / (t[:, 0] + t[:, 1]) - t[:, 2] / (t[:, 2] + t[:, 3]))):
                                est, lo, hi, _ = cluster_bootstrap_sums(firearm_paired, 'video', 'category', vals, stat, B, seed, f'{name} {label}', level)
                                row.update({f'{label}_diff_AG_minus_G': est, f'{label}_diff_ci_low': lo, f'{label}_diff_ci_high': hi})
                        if cat in ('No_Gun', 'All'):
                            neg_v = lambda u: ((bool(u['AG']['fp']), 1, bool(u['G']['fp']), 1) if not u['positive'] and u['AG']['answered'] and u['G']['answered']
                                               else (0, 0, 0, 0))
                            est, lo, hi, _ = cluster_bootstrap_sums(paired, 'video', 'category', neg_v, lambda t: t[:, 0] / t[:, 1] - t[:, 2] / t[:, 3],
                                                                    B, seed, f'{name} negative', level)
                            row.update(negative_box_rate_diff=est, negative_box_rate_diff_ci_low=lo, negative_box_rate_diff_ci_high=hi)
                    rows.append(row)
                    if origin_name == 'primary' and thr == d.study['matching']['iou_thresholds'][0] and not no_truth:
                        for size in ('small', 'medium', 'large'):
                            n = sum(u['sizes'].count(size) for u in us)
                            hit = sum(u[c + '_matched_sizes'].count(size) for u in us)
                            rows.append({'origin': origin_name, 'iou': thr, 'condition': c, 'category': cat, 'size': size,
                                         'gt_objects': n, 'tp': hit, 'recall': hit / n if n else None,
                                         'na_reason': '' if n else 'no objects of this size'})
    columns = ['origin', 'iou', 'condition', 'category', 'size', 'evaluable_slots', 'positive_slots', 'gt_objects', 'no_record_slots', 'tp',
               'fp_on_boxed_frames', 'fn', 'precision', 'recall', 'f1', 'precision_incl_no_gun_frames', 'answered_recall', 'matched_iou_mean',
               'matched_iou_median', 'negative_slots', 'negative_box_rate', 'paired_executed_slots', 'recall_diff_AG_minus_G', 'recall_diff_ci_low',
               'recall_diff_ci_high', 'precision_diff_AG_minus_G', 'precision_diff_ci_low', 'precision_diff_ci_high', 'negative_box_rate_diff',
               'negative_box_rate_diff_ci_low', 'negative_box_rate_diff_ci_high', 'na_reason']
    return table('T8', 'Localization of firearms against the existing boxes (fixed-output operating point)', columns, rows, [
        'Evaluable slots: boxed frames (recall and precision; one annotated firearm per frame) and No_Gun frames (false positives only). '
        'Frames listed without a box are excluded (audit: subjects sometimes visible without a box). G and AG use the identical slot set.',
        'One-to-one matching maximising the number of matches, then total IoU. Recall is end to end: a slot without a usable answer or without '
        'a complete record misses its firearm (no_record_slots counts the latter). answered_recall restricts recall to answered slots.',
        'precision uses boxed frames of firearm videos only; precision_incl_no_gun_frames (All rows) also counts boxes on No_Gun frames as false '
        'positives. negative_box_rate: answered No_Gun slots with at least one box.',
        'AG-minus-G differences use slots executed in both conditions, with paired video-bootstrap intervals. No confidence scores exist, so no '
        'ranked mAP. origin sensitivity: ground truth re-mapped with image_id k = frame k. Size rows: GT-area tertiles fixed from the annotations.'])


def t10(d):
    rows = []
    for c in ('A', 'G', 'AG'):
        recs = list(d.final[c].values())
        secs = [r['seconds'] for r in recs]
        toks = [r['generated_tokens'] for r in recs]
        failed = sum(1 for e in d.ledger if e['event'] == 'end' and e['unit'].endswith('|' + c) and e['status'] != 'complete')
        rows.append({'stage': f'inference {c}', 'units': len(recs), 'failed_or_interrupted_attempts': failed,
                     'total_hours': sum(secs) / 3600 if secs else None, 'median_s': float(np.median(secs)) if secs else None,
                     'p95_s': float(np.quantile(secs, 0.95)) if secs else None, 'tokens_total': sum(toks),
                     'tokens_median': float(np.median(toks)) if toks else None, 'tokens_p95': float(np.quantile(toks, 0.95)) if toks else None,
                     'truncated': sum(bool(r['truncated']) for r in recs),
                     'peak_gpu_gib': max((r['peak_gpu_bytes'] or 0) for r in recs) / 2 ** 30 if recs else None})
    loads = [s.get('model_load_seconds') for s in d.sessions if s.get('model_load_seconds')]
    rows.append({'stage': 'model loads (sessions)', 'units': len(loads), 'total_hours': sum(loads) / 3600 if loads else None,
                 'median_s': float(np.median(loads)) if loads else None})
    fs = [x for x in d.feature_seconds if x]
    rows.append({'stage': 'ImageBind features (separate)', 'units': len(fs), 'total_hours': sum(fs) / 3600 if fs else None,
                 'median_s': float(np.median(fs)) if fs else None, 'p95_s': float(np.quantile(fs, 0.95)) if fs else None})
    columns = ['stage', 'units', 'failed_or_interrupted_attempts', 'total_hours', 'median_s', 'p95_s', 'tokens_total', 'tokens_median',
               'tokens_p95', 'truncated', 'peak_gpu_gib']
    return table('T10', 'Computation', columns, rows, [
        'Inference: generation seconds per call (complete records), one RTX 6000 Ada, BF16, FlashAttention, greedy, 1024-token cap. '
        'Feature units are videos. Decoding and PNG writing are not timed separately.'])
