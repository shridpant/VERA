"""Reference run of the experiment03A power-fixture gate (protocol v0.4 §13.3), stdlib only.

Imports round2_design_sim.py from this directory (same generator, same seed strings, so the first 200 draws of each
cell equal the 200-draw reference run) and runs the +0.5 nat power cells at 1,000 draws, PEAK rule only.

Cells:
  scorer tanh; A in {4, 6, 8}; w in {1.5, 2, 3}; syn = 0.5;
  "event":    event3, twopairs, S1like, plateau_bg, raccident, heterog
  "evidence": spread, event3, twopairs, S1like, plateau_bg, raccident, heterog
  margins:    U(0.2, 1.5) and U(0.05, 0.3) are gated; U(1.5, 3.0) is reported only.
Criteria (gated margins):
  C1  NOT SUPPORTED in <= 5 % of draws, in every cell;
  C2  in the gated layouts ("event": event3, twopairs, heterog; "evidence": spread, event3, twopairs, heterog), in every
      cell with >= 50 evaluable draws (verdict neither INSEPARABLE nor INCONCLUSIVE for |T| < 6): the one-sided 90 %
      Clopper-Pearson lower bound of SUPPORTED / evaluable is >= 0.40.
Usage: python3 -I -B round3_power_gate.py <draws> <margin_lo> <margin_hi> <mode>   (prints one line per cell)
       python3 -I -B round3_power_gate.py --check <output files...>                 (applies C1 and C2)
"""
import importlib.util
import math
import re
import sys
from pathlib import Path

GATED = {"event": ("event3", "twopairs", "heterog"), "evidence": ("spread", "event3", "twopairs", "heterog")}
MIXES = {"event": ("event3", "twopairs", "S1like", "plateau_bg", "raccident", "heterog"),
         "evidence": ("spread", "event3", "twopairs", "S1like", "plateau_bg", "raccident", "heterog")}
GATED_MARGINS = ("(0.2, 1.5)", "(0.05, 0.3)")


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Bin(n, p), via log-space terms."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def cp_lower(k, n, conf=0.90):
    """One-sided Clopper-Pearson lower bound: the p with P(X >= k | p) = 1 - conf (bisection)."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) < 1 - conf:
            lo = mid
        else:
            hi = mid
    return lo


def run(draws, margin, mode):
    spec = importlib.util.spec_from_file_location("sim", Path(__file__).with_name("round2_design_sim.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    for mix in MIXES[mode]:
        for A in (4, 6, 8):
            for w in (1.5, 2, 3):
                o, _ = sim.run_tanh(mix, A, w, 0.5, mode, draws, margin, rules=("PEAK",))
                print(f"gate syn=0.5 {mode} {mix} A={A} w={w} margin={margin} draws={draws}: {sim.fmt(o['PEAK'], draws)}", flush=True)


def check(files):
    rows = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            m = re.match(r"gate syn=0.5 (\w+) (\w+) A=(\d+) w=([\d.]+) margin=(\([^)]*\)) draws=(\d+): (.*)", line)
            if not m:
                continue
            mode, mix, A, w, margin, n, st = m.groups()
            c = {k: int(v) for k, v in re.findall(r"(SUPPORT|NOT_SUP|INCONCL_T|INCONCL|INSEPAR)=(\d+)", st)}
            rows.append((mode, mix, int(A), float(w), margin, int(n), c))
    fails, out = [], []
    for mode, mix, A, w, margin, n, c in rows:
        ev = n - c["INSEPAR"] - c["INCONCL_T"]
        gated_margin = margin in GATED_MARGINS
        c1 = c["NOT_SUP"] <= 0.05 * n
        lb = cp_lower(c["SUPPORT"], ev) if ev else None
        c2_applies = gated_margin and mix in GATED[mode] and ev >= 50
        c2 = (lb >= 0.40) if c2_applies else None
        if gated_margin and not c1:
            fails.append(("C1", mode, mix, A, w, margin, c["NOT_SUP"]))
        if c2 is False:
            fails.append(("C2", mode, mix, A, w, margin, round(lb, 3)))
        out.append(f"{mode:8s} {mix:10s} A={A} w={w:<3} margin={margin:12s} n={n} NOT_SUP={c['NOT_SUP']:4d} INSEPAR={c['INSEPAR']:4d} "
                   f"evaluable={ev:4d} SUP/ev={(c['SUPPORT'] / ev if ev else float('nan')):.3f} CP90_lower={lb if lb is None else round(lb, 3)} "
                   f"{'gated' if c2_applies else ('C1-only' if gated_margin else 'reported')}")
    print("\n".join(out))
    print(f"cells: {len(rows)}; failures at gated margins: {fails if fails else 'none'}")
    return not fails


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(0 if check(sys.argv[2:]) else 1)
    run(int(sys.argv[1]), (float(sys.argv[2]), float(sys.argv[3])), sys.argv[4])
