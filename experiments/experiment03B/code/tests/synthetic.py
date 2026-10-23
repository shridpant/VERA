"""Synthetic fixtures for experiment03B tests: an analytic mock scorer spec and a matching mock-engine config.

Every expected value below is worked out by hand from the weights (mock score F = offset + sum w_i v_i + interactions,
v_i = 1 if slot i is visible; original label = 1 iff F > 0; native S = F rounded to the 0.25 grid).
"""
import copy
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
EXP = CODE.parent


def w(*ws):
    return list(ws)


INPUTS = [
    # uid, video, category, stratum, development, blur spec, extra
    ("SynthArson_001@160", "SynthArson_001", "Arson", "event", False,
     {"w": w(10, 5, 1, 0, 0, 0, 0, 0), "inter": [[0, 2, 5]], "offset": -8}, {}),          # brief's interaction fixture
    ("SynthArson_001@896", "SynthArson_001", "Arson", "negative", False,
     {"w": w(1, 2, 3, 4, 5, 6, 7, 8), "offset": -20}, {}),                                  # additive
    ("SynthBurglary_002@208", "SynthBurglary_002", "Burglary", "event", False,
     {"w": w(0, 0, 6, 6, 0, 0, 0, 0), "offset": -1}, {}),
    ("SynthBurglary_002@592", "SynthBurglary_002", "Burglary", "negative", False,
     {"w": w(5, 0, 0, 0, 0, 0, 0, 0), "offset": -1}, {"missing_labels": [["grey", [0, 1]]]}),
    ("SynthNormal_900@1504", "SynthNormal_900", "Normal", "normal", False,
     {"w": w(1, 1, 1, 1, 1, 1, 1, 1), "offset": -10}, {}),                                  # L0 = 0, all tied, all nonpositive
    ("SynthNormal_901@1504", "SynthNormal_901", "Normal", "normal", False,
     {"w": w(4, 0, 0, 0, 0, 0, 0, 4), "offset": -5}, {}),                                   # normal-video alert
    ("SynthRobbery_005@160", "SynthRobbery_005", "Robbery", "event", False,
     {"w": w(1, 0, 0, 0, 0, 0, 0, 0), "offset": 1}, {"baseline_label": None, "nosound_label": 1}),  # L0 missing
    ("RoadAccidents127_x264@2224", "RoadAccidents127_x264", "RoadAccidents", "event", True,
     {"w": w(0, 0, 0, 9, 9, 0, 0, 0), "offset": -10}, {}),                                  # development video
    ("SynthShooting_006@160", "SynthShooting_006", "Shooting", "event", False,
     {"w": w(8, 8, 0, 0, 0, 0, 0, 0), "offset": -10}, {}),
    ("SynthShooting_006@496", "SynthShooting_006", "Shooting", "event", False,
     {"w": w(2, 2, 2, 2, 2, 2, 2, 2), "offset": -13}, {}),                                  # all tied, every pair flips
    ("SynthStealing_007@160", "SynthStealing_007", "Stealing", "negative", False,
     {"w": w(-1, -1, -1, -1, -1, -1, -1, -1), "offset": 13}, {}),                           # all nonpositive, L0 = 1
]

EXPECTED = {
    "d_w": {"SynthArson_001@160": 0.75, "SynthArson_001@896": 0.0, "SynthBurglary_002@208": 1 - 1 / 28,
            "SynthBurglary_002@592": None, "SynthNormal_901@1504": 1 - 13 / 28, "RoadAccidents127_x264@2224": 1 - 13 / 28,
            "SynthShooting_006@160": 1 - 13 / 28, "SynthShooting_006@496": 0.0, "SynthStealing_007@160": 0.0},
    "bounds_SynthBurglary_002@600": (-7 / 28, 1 - 6 / 28),
    # video means over the confirmation set (no development, L0 = 1, d_w present), then their mean:
    "video_means": {"SynthArson_001": 0.375, "SynthBurglary_002": 1 - 1 / 28, "SynthNormal_901": 1 - 13 / 28,
                    "SynthShooting_006": (1 - 13 / 28) / 2, "SynthStealing_007": 0.0},
    "primary_pp": 100 * 3 / 7,
    "fixed_order_A": [0, 2, 1, 3, 4, 5, 6, 7],
    "adaptive_order_A": [0, 1, 2, 3],
    "effects_A": {0: 15.0, 1: 5.0, 2: 6.0},
    "conditional_after_0_A": {1: 5.0, 2: 1.0},
}


def build(out_dir, runs_root, run_id="synthetic_001", grey_overrides=None, limits=None, extra=None):
    """Write mock_spec.json and config.synthetic.json into out_dir; return the config path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec, cases = {}, []
    for uid, vid, cat, stratum, dev, blur, ex in INPUTS:
        s = {"blur": blur, "grey": copy.deepcopy(blur), "label_threshold": 0.0, **ex}
        if grey_overrides and uid in grey_overrides:
            s["grey"] = grey_overrides[uid]
        spec[uid] = s
        anchor = int(uid.split("@")[1])
        cases.append({"window_uid": uid, "video_id": vid, "category": cat, "anchor": anchor,
                      "frame_indices": [int(anchor - 150 + 37.5 * i) for i in range(8)],  # VERA arithmetic, unclipped
                      "frame_sha256": [f"{uid}-frame{i}" for i in range(8)], "video_sha256": f"{vid}-video",
                      "frames_dir": f"synthetic/{vid}", "strata": [stratum], "window_record_ids": [f"{stratum}:{uid}"],
                      "development": dev, "metadata_flags": [], "annotated_slots": [2, 3] if stratum == "event" else [],
                      "clipped": False, "status": "selected", "manifest_entry": f"frames/{cat}/{vid}.mp4",
                      "manifest_length": 3000})
    (out / "mock_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    cfg = json.loads((EXP / "config.integration.json").read_text())
    cfg.update({"run_id": run_id, "engine": "mock", "mock_spec": str((out / "mock_spec.json").resolve())})
    cfg["population"] = {"kind": "cases", "development_videos": ["RoadAccidents127_x264", "Normal_Videos_018_x264"],
                         "cases": cases}
    cfg["retrieval"] = {"mode": "none", "frames_root": "synthetic", "scratch_dir": str(Path(out) / "scratch")}
    cfg["prior_caches"] = []
    cfg["paths"] = {"runs_root": str(Path(runs_root).resolve())}
    cfg["limits"] = limits or {"max_new_transformed_inputs": None, "max_gpu_hours": None, "max_scratch_mib": None,
                               "max_model_calls": None}
    cfg["analysis"] = {"bootstrap_B": 2000, "alpha": 0.05, "stability_seed_offsets": [1, 2, 3, 4],
                       "secondary_scope": "confirmation"}
    cfg.pop("integration_checks", None)
    if extra:
        cfg.update(extra)
    p = out / "config.synthetic.json"
    p.write_text(json.dumps(cfg, indent=1) + "\n")
    return p
