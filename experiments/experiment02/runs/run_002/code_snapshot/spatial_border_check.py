import json, sys
import numpy as np
from PIL import Image
rd, root = sys.argv[1], sys.argv[2]
cases = {c["case_id"]: c for c in json.load(open(f"{rd}/cases.json"))["cases"]}
c = cases["R-accident"]
dark = []
for fi in c["frame_indices"]:
    im = np.asarray(Image.open(f"{root}/{c['frames_dir']}/{fi:06d}.jpg").convert("L").resize((448, 448), Image.BICUBIC), dtype=float)
    tok = im.reshape(32, 14, 32, 14).mean((1, 3))
    dark.append(tok < 12)
dark = np.stack(dark)
out = {"dark_token_area_share": float(dark.mean())}
for k in ("gradcam_vit_32x32", "gradcampp_vit_32x32"):
    a = np.load(f"{rd}/spatial/{k}.npy").astype(float)
    out[k + "_mass_on_dark_tokens"] = float(a[dark].sum() / a.sum())
g = np.load(f"{rd}/spatial/gradxinput_tokens_16x16.npy").astype(float)
d16 = dark.reshape(8, 16, 2, 16, 2).all((2, 4))
out["gradxinput_abs_mass_on_dark_tokens"] = float(np.abs(g)[d16].sum() / np.abs(g).sum())
out["dark_visual_token_area_share"] = float(d16.mean())
p = np.load(f"{rd}/spatial/pixel_grad_abs_448.npy").astype(float)
dpx = np.repeat(np.repeat(dark, 14, 1), 14, 2)
out["pixel_grad_mass_on_dark"] = float(p[dpx].sum() / p.sum())
print(json.dumps(out, indent=1))
json.dump(out, open(f"{rd}/spatial/border_mass_check.json", "w"), indent=1)
