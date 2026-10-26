"""Diagnostic: which clip subsampling reproduces VERA's released RoadAccidents127 features best?
num_samples 10 (LAVAD script) vs 32 vs all clip frames (N >= 300), on 21 rows; cosine to released."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/local/datasets/sp2117/firearm_visilab/code/VERA-firearm")
from vera_runner.imagebind_features import Extractor, clip_frame_indices, clip_tensor
frames = Path("/local/datasets/sp2117/firearm_visilab/features_check/RoadAccidents127/frames")
n = 2580
rows = list(range(0, 6)) + list(range(124, 136)) + [13, 59, 72]
released = np.load("/home/stu11/s5/sp2117/vera_research/shared/features/vera_official_ucf_imagebind/RoadAccidents127_x264.npy")
ext = Extractor("/local/datasets/sp2117/models/imagebind/imagebind_huge.pth")
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
res = {}
for N in (10, 32, 300):
    emb = np.concatenate([ext.embed([clip_tensor([frames / f"{i:06d}.jpg" for i in clip_frame_indices(r * 16, n, num_samples=N)])]) for r in rows])
    res[N] = {r: cos(emb[k], released[r]) for k, r in enumerate(rows)}
    print(N, "median", round(float(np.median(list(res[N].values()))), 5), "min", round(min(res[N].values()), 5), {r: round(v, 4) for r, v in res[N].items()}, flush=True)
Path(sys.argv[1]).write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
