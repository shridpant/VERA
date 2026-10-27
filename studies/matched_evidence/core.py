"""CPU-only mathematical core. This module does not run a VLM or transform video."""
import math
from statistics import median


def support_score(logit_one, logit_zero):
    margin = float(logit_one) - float(logit_zero)
    if not math.isfinite(margin):
        raise ValueError("nonfinite candidate margin")
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-margin))
    value = math.exp(margin)
    return value / (1.0 + value)


def _valid_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def select_anchors(records, fraction=0.10, minimum=0.5):
    """One video's complete baseline records; output anchors without reading labels."""
    if not 0 < fraction <= 1 or not _valid_score(minimum):
        raise ValueError("invalid selection settings")
    if any(not isinstance(r["anchor"], int) or not _valid_score(r["score"]) for r in records):
        raise ValueError("invalid baseline record")
    if len({r["anchor"] for r in records}) != len(records):
        raise ValueError("duplicate anchor")
    eligible = [r for r in records if r["score"] >= minimum]
    eligible.sort(key=lambda r: (-r["score"], r["anchor"]))
    return [r["anchor"] for r in eligible[:math.ceil(fraction * len(records))]]


def corroborate(base, confirmation, views, input_checks=True, stability_guard=0.20):
    """Return audited correction from two operators and three controls per view.

    views[operator] contains delete_evidence, delete_controls,
    retain_evidence, retain_controls. Controls must be all other quadrants.
    """
    if not _valid_score(base):
        raise ValueError("invalid base score; no valid fallback exists")
    result = {"base": base, "score": base, "boost": 0.0, "status": "unverified", "contrasts": {}}
    if not input_checks:
        result["reason"] = "input_checks_failed"
        return result
    if not _valid_score(confirmation):
        result["reason"] = "missing_confirmation"
        return result
    if stability_guard is not None and abs(base - confirmation) > stability_guard:
        result["reason"] = "sampling_instability"
        return result
    if set(views) != {"blur", "mean"}:
        result["reason"] = "missing_operator"
        return result
    for name in ("blur", "mean"):
        view = views[name]
        if not isinstance(view, dict):
            result["reason"] = "invalid_view"
            return result
        dc, rc = view.get("delete_controls", []), view.get("retain_controls", [])
        if not isinstance(dc, (list, tuple)) or not isinstance(rc, (list, tuple)) or len(dc) != 3 or len(rc) != 3:
            result["reason"] = "incomplete_controls"
            return result
        de, re = view.get("delete_evidence"), view.get("retain_evidence")
        if not all(_valid_score(x) for x in [de, re, *dc, *rc]):
            result["reason"] = "invalid_view_score"
            return result
        result["contrasts"][name] = {"deletion": median(dc) - de, "retention": re - median(rc)}
    signed = [c for v in result["contrasts"].values() for c in v.values()]
    boost = max(0.0, min(signed))
    result.update(boost=boost, score=base + (1.0 - base) * boost,
                  status="corroborated" if boost > 0 else "unresolved",
                  reason="all_contrasts_positive" if boost > 0 else "inconclusive_evidence")
    return result
