"""Diagnostic: do the low-cosine rows of our RoadAccidents127 features come from frame decoding?
Decode with PyAV (instead of OpenCV), write default cv2 JPEGs, recompute chosen rows, compare to released."""
import json, sys
from pathlib import Path
import av, cv2, numpy as np
sys.path.insert(0, "/local/datasets/sp2117/firearm_visilab/code/VERA-firearm")
from vera_runner.imagebind_features import Extractor, clip_frame_indices, clip_tensor
out = Path(sys.argv[1]); frames = out / "frames_pyav"; frames.mkdir(parents=True, exist_ok=True)
video = "/local/datasets/sp2117/ucf_crime_test_videos/RoadAccidents127_x264.mp4"
n = 0
with av.open(video) as c:
    for f in c.decode(video=0):
        cv2.imwrite(str(frames / f"{n:06d}.jpg"), cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)); n += 1
cv_frames = Path("/local/datasets/sp2117/firearm_visilab/features_check/RoadAccidents127/frames")
diff = [float(np.abs(cv2.imread(str(frames / f"{i:06d}.jpg")).astype(int) - cv2.imread(str(cv_frames / f"{i:06d}.jpg")).astype(int)).mean()) for i in range(0, n, 97)]
rows = list(range(0, 6)) + list(range(124, 136)) + [13, 59, 72]
ext = Extractor("/local/datasets/sp2117/models/imagebind/imagebind_huge.pth")
emb = ext.embed([clip_tensor([frames / f"{i:06d}.jpg" for i in clip_frame_indices(r * 16, n)]) for r in rows])
released = np.load("/home/stu11/s5/sp2117/vera_research/shared/features/vera_official_ucf_imagebind/RoadAccidents127_x264.npy")
ours = np.load("/local/datasets/sp2117/firearm_visilab/features_check/RoadAccidents127/ours.npy")
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
res = {"pyav_frames": n, "mean_abs_pixel_diff_sampled_frames_pyav_vs_opencv": diff,
       "rows": [{"row": r, "pyav_vs_released": cos(emb[k], released[r]), "opencv_vs_released": cos(ours[r], released[r]),
                 "pyav_vs_opencv": cos(emb[k], ours[r])} for k, r in enumerate(rows)]}
(out / "decode_diag.json").write_text(json.dumps(res, indent=1))
for x in res["rows"]: print(x)
print("pixel diffs", [round(v, 3) for v in diff])
