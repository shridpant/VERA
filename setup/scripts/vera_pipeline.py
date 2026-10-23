"""Shared, GPU-free pieces of the VERA one-video pipeline.

Everything here mirrors the pinned VERA source (commit 15b8bcb8) line for line where it computes numbers:
- window_for_anchor(): the anchor/window arithmetic of generate_initial_score.py's main loop;
- vera_refine(): the per-video body of compute_auc.py (scene-context refinement, temporal smoothing,
  segment-to-frame expansion and position weighting), returning every intermediate array;
- gt_frame_mask(): compute_auc.py's ground-truth construction (used for diagnostics only).
validate_run.py checks vera_refine() against the unmodified compute_auc.py executed in a sandbox.

The output parser is new: upstream's parsers turn malformed text into a label silently.
"""
import re

import numpy as np

# ---------------------------------------------------------------- output parsing

_OUTPUT_FIELD = re.compile(r"output\s*:", re.IGNORECASE)
_STRIP = " \t\r\n`[]()\"'*._"


def strict_parse_label(text):
    """Return (label, status). label is 0, 1 or None; status is 'ok' or a reason.

    The learner instruction asks for 'Output:' followed by ONLY the integer label. Accepted: exactly one
    'Output:' field (case-insensitive), whose remaining text is a single 0 or 1 after removing whitespace,
    code fences, brackets, quotes, asterisks and a trailing period. Anything else is not given a label.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    fields = list(_OUTPUT_FIELD.finditer(text))
    if not fields:
        return None, "missing_output_field"
    if len(fields) > 1:
        return None, "multiple_output_fields"
    tail = text[fields[0].end():].replace("```", " ").strip(_STRIP)
    if tail in ("0", "1"):
        return int(tail), "ok"
    if not tail:
        return None, "empty_output_field"
    return None, "output_field_not_a_single_label"


def upstream_root_parse(text):
    """generate_initial_score.py (root) at 15b8bcb8: label 0 iff a '0' follows the last 'Output', else 1."""
    return 0 if "0" in text.split("Output")[-1] else 1


def upstream_inference_variant_parse(text):
    """Inference/generate_intial_score.py at 15b8bcb8: label 1 iff the last character is '1', else 0."""
    return 1 if text and text[-1] == "1" else 0


SOUND_WORDS = re.compile(r"\b(sound|sounds|noise|noises|noisy|audio|hear|heard|hearing|loud|silent|silence|siren|"
                         r"sirens|honk|honking|scream|screaming|bang|crash sound)\b", re.IGNORECASE)


def mentions_sound(text):
    return bool(SOUND_WORDS.search(text or ""))


# ---------------------------------------------------------------- windows (generate_initial_score.py main loop)

def anchors(n_frames, sampling_rate=16):
    start, end = 0, n_frames
    return [i for i in range(start, end, sampling_rate)]


def window_for_anchor(ind, n_frames, fps=30, t_window=10, snippet_len=8):
    """Exactly the upstream expressions; returns (snippet_start, snippet_end, snippet_indices)."""
    start, end = 0, n_frames
    snippet_start = max(ind - 1/2*fps*t_window, start)
    snippet_end = min(ind+1/2*fps*t_window, end)
    snippet_indices = np.arange(snippet_start, snippet_end, (snippet_end-snippet_start)/snippet_len).astype(int).tolist()
    return snippet_start, snippet_end, snippet_indices


# ---------------------------------------------------------------- refinement (compute_auc.py per-video body)

def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_smoothing(data, sigma):
    n = len(data)
    smoothed_data = np.zeros(n)
    x = np.arange(n)

    centroid_index = int(n/2)

    kernel_values = gaussian_kernel_original(x, centroid_index, sigma)
    smoothed_data = kernel_values * data

    return smoothed_data


def gaussian_kernel(size, sigma):
    kernel = np.exp(-np.linspace(-size//2, size//2, size)**2 / (2*sigma**2))
    return kernel / kernel.sum()


def gaussian_smooth_1d(data, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data


def softmax(row_vector):
    exp_row = np.exp(row_vector - np.max(row_vector))
    return exp_row / np.sum(exp_row)


def vera_refine(score_dict, v_array_raw, sampling_rate=16):
    """compute_auc.py's per-video computation for one video, with intermediates.

    score_dict: {anchor_str: {'start', 'end', 'score'}} in anchor order (the Step 1 JSON format).
    v_array_raw: (num_segment, D) segment features. Raises on a row/anchor mismatch instead of
    letting upstream index past the end or silently use fewer rows.
    """
    start_index = [key for key in score_dict.keys()]
    if v_array_raw.ndim != 2 or v_array_raw.shape[0] != len(start_index):
        raise ValueError(f"feature rows {v_array_raw.shape} do not match {len(start_index)} scored anchors")

    v_norms = np.linalg.norm(v_array_raw, axis=1, keepdims=True)
    normalized_v = v_array_raw / v_norms
    v_array = np.dot(normalized_v, normalized_v.T)

    num_segment = v_array.shape[0]
    top_n = int(0.15*num_segment)

    v_indices = np.argsort(v_array, axis=1)[:, -top_n:]
    v_top_values = np.take_along_axis(v_array, v_indices, axis=1)

    all_score = []
    for i, start in enumerate(start_index):
        all_score.append(score_dict[start]['score'])

    if max(all_score) == 1.0:  # upstream no-op (array round trip), kept for fidelity
        all_score = np.asarray(all_score)
        one_indices = np.where(all_score != 0)[0]
        all_score = all_score.tolist()

    score_all = []
    neighbor_weights = []
    for i, start in enumerate(start_index):
        neighbor_score = []
        for neighbor in v_indices[i].tolist():
            neighbor_score += [all_score[neighbor]]
        neighbor_score = np.array(neighbor_score)
        v_softmax_values = softmax(v_top_values[i]*10)
        neighbor_weights.append(v_softmax_values)

        v_segment_score = np.dot(v_softmax_values, neighbor_score)

        score_all.append(v_segment_score)
    scene_refined = np.array(score_all, dtype=float)

    smoothed_data = gaussian_smooth_1d(np.array(score_all), 15, 10)

    pred_score = []
    rounded = []
    for i, start in enumerate(start_index):
        v_segment_score = smoothed_data[i]

        segment_score = round(v_segment_score, 1)
        rounded.append(segment_score)
        if i != len(start_index)-1:
            num_ele = sampling_rate
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)
        else:
            num_ele = int(score_dict[start]['end']-int(start))
            seg_list = [segment_score]*num_ele
            pred_score.extend(seg_list)

    before_position = np.array(pred_score, dtype=float)
    sigma = int(len(pred_score)/2)
    final = gaussian_smoothing(np.array(pred_score), sigma)
    position_weights = gaussian_kernel_original(np.arange(len(pred_score)), int(len(pred_score)/2), sigma)

    return {
        "anchors": [int(s) for s in start_index],
        "raw_labels": np.array(all_score, dtype=float),
        "cosine_similarity": v_array,
        "top_n": top_n,
        "neighbor_indices": v_indices,
        "neighbor_weights": np.array(neighbor_weights),
        "scene_refined": scene_refined,
        "temporal_smoothed": np.asarray(smoothed_data, dtype=float),
        "temporal_rounded": np.array(rounded, dtype=float),
        "frame_before_position_weighting": before_position,
        "position_sigma": sigma,
        "position_weights": position_weights,
        "frame_final": np.asarray(final, dtype=float),
    }


def gt_frame_mask(length, temporal_label):
    """compute_auc.py's ground truth: frames [s, e) set to 1 for each (s, e) pair that is not -1."""
    gt = [0.0]*length
    for anno_i in range(0, len(temporal_label), 2):
        if temporal_label[anno_i] != -1:
            anno_s = temporal_label[anno_i]
            if temporal_label[anno_i+1] > length:
                anno_e = length
            else:
                anno_e = temporal_label[anno_i+1]
            gt[anno_s:anno_e] = [1.0]*(anno_e-anno_s)
    return np.array(gt)
