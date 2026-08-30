"""
analysis/rq1_dcor_humanhuman.py -- Human-Human distance correlation (reviewer request).

The two RQ1 scripts deliberately skip Human-Human pairings. This adds them and
puts them on the same footing as the Human-LLM and Between-LLM cells so the
three are directly comparable:

  CIP        fully crossed (5 humans x 100 transcripts), so all 10 human pairs
             are estimable at the same paired n = 100 as the LLM cells.
  Menagerie  humans graded only their own group's submissions, so most human
             pairs share ZERO submissions. We keep only pairs overlapping on
             >= MIN_N submissions -- a non-random, within-group subset. Report
             with that caveat. Note the paired n for those pairs (~40) matches
             the Human-LLM cells, so HH vs HL is the fair comparison within
             Menagerie; Between-LLM runs at n = 279 and is less comparable.

Everything (bootstrap resample, shift-corrected percentile CI) matches the
parent scripts so the numbers can sit in one table.

Run: python analysis/rq1_dcor_humanhuman.py
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rq1_dcor as CIP
import rq1_dcor_menagerie as MEN

RESULTS = CIP.RESULTS
N_BOOT = 2000
SEED = 0

HH = "Human-Human"
HL = "Human-LLM"
LL = "Between-LLM"


# ---------- pair builders: return lists of (x, y) aligned vector pairs ----------
def cip_pairs(vec):
    """CIP, same rubric item. Human pairs are unordered (10), as are LLM pairs."""
    hh = [(vec[("human", a, t)], vec[("human", b, t)])
          for a, b in itertools.combinations(CIP.HUMANS, 2) for t in CIP.TASKS]
    hl = [(vec[("human", h, t)], vec[("llm", m, t)])
          for h in CIP.HUMANS for m in CIP.LLMS for t in CIP.TASKS]
    ll = [(vec[("llm", a, t)], vec[("llm", b, t)])
          for a, b in itertools.combinations(CIP.LLMS, 2) for t in CIP.TASKS]
    return {HH: hh, HL: hl, LL: ll}


def menagerie_pairs(vec, humans, llms, min_n=MEN.MIN_N):
    """Menagerie, same rubric item. Human-Human is restricted to pairs whose
    graded-submission sets overlap on >= min_n submissions (i.e. same group)."""
    hh = []
    kept = set()
    for a, b in itertools.combinations(humans, 2):
        # overlap is a property of the rater pair, not the skill; test on one skill
        x, y = vec[("human", a, "correctness")], vec[("human", b, "correctness")]
        if int((~(np.isnan(x) | np.isnan(y))).sum()) >= min_n:
            kept.add((a, b))
            for t in MEN.SKILLS:
                hh.append((vec[("human", a, t)], vec[("human", b, t)]))
    hl = [(vec[("human", h, t)], vec[("llm", m, t)])
          for h in humans for m in llms for t in MEN.SKILLS]
    ll = [(vec[("llm", a, t)], vec[("llm", b, t)])
          for a, b in itertools.combinations(llms, 2) for t in MEN.SKILLS]
    return {HH: hh, HL: hl, LL: ll}, kept


# ---------- estimation ----------
def pooled(pairs, dcor_fn, idx=None):
    """Mean dCor^2 over a list of vector pairs, optionally under a resample."""
    if idx is None:
        vals = [dcor_fn(x, y) for x, y in pairs]
    else:
        vals = [dcor_fn(x[idx], y[idx]) for x, y in pairs]
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def paired_n(pairs):
    ns = [int((~(np.isnan(x) | np.isnan(y))).sum()) for x, y in pairs]
    return int(np.median(ns))


def run(label, pair_sets, dcor_fn, n_obs, n_boot=N_BOOT, seed=SEED):
    """Point estimate + shift-corrected 95% CI for each cell, plus the
    HH-vs-HL and LL-vs-HH contrasts computed on the SAME bootstrap draws
    (so the paired dependence is handled correctly)."""
    point = {k: pooled(v, dcor_fn) for k, v in pair_sets.items()}
    draws = {k: [] for k in pair_sets}

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n_obs, n_obs)
        for k, v in pair_sets.items():
            draws[k].append(pooled(v, dcor_fn, idx))
        if (b + 1) % 250 == 0:
            print(f"  [{label}] bootstrap {b + 1}/{n_boot}", flush=True)
    draws = {k: np.asarray(v, float) for k, v in draws.items()}

    rows = []
    for k in (HH, HL, LL):
        lo, hi = shift_ci(draws[k], point[k])
        rows.append({"dataset": label, "cell": k, "n_pairs": len(pair_sets[k]),
                     "paired_n_median": paired_n(pair_sets[k]),
                     "dcor_sq": point[k], "ci_lo": lo, "ci_hi": hi})
    contrasts = []
    for a, b_ in ((LL, HH), (HH, HL), (LL, HL)):
        d = draws[a] - draws[b_]
        obs = point[a] - point[b_]
        lo, hi = shift_ci(d, obs)
        dd = d - (np.nanmean(d) - obs)
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        contrasts.append({"dataset": label, "contrast": f"{a} - {b_}",
                          "diff": obs, "ci_lo": lo, "ci_hi": hi,
                          "p_value": max(p, 1.0 / n_boot)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def shift_ci(draws, point, level=95):
    """Bias-corrected (shifted) percentile CI -- identical to the parent scripts."""
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    b = a.mean() - point
    lo = (100 - level) / 2
    return float(np.percentile(a, lo)) - b, float(np.percentile(a, 100 - lo)) - b


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cells, contrasts = [], []

    # ----- CIP -----
    print("CIP: fully crossed, all 10 human pairs estimable")
    df = CIP.load_long(CIP.COMPILED)
    vec_c = CIP.build_vectors(df)
    n_c = len(next(iter(vec_c.values())))
    ps = cip_pairs(vec_c)
    print(f"  transcripts={n_c}  HH pairs={len(ps[HH])}  "
          f"HL={len(ps[HL])}  LL={len(ps[LL])}")
    a, b = run("CIP", ps, CIP.dcor_sq, n_c)
    cells.append(a)
    contrasts.append(b)

    # ----- Menagerie -----
    print("\nMenagerie: Human-Human restricted to within-group pairs")
    vec_m, humans, llms, subs = MEN.load_vectors(MEN.COMPILED)
    n_m = len(subs)
    pm, kept = menagerie_pairs(vec_m, humans, llms)
    total_pairs = len(list(itertools.combinations(humans, 2)))
    print(f"  submissions={n_m}  humans={len(humans)}")
    print(f"  human pairs with >= {MEN.MIN_N} shared submissions: "
          f"{len(kept)}/{total_pairs} ({100 * len(kept) / total_pairs:.0f}%)")
    a, b = run("Menagerie", pm, MEN.dcor_sq, n_m)
    cells.append(a)
    contrasts.append(b)

    cells = pd.concat(cells, ignore_index=True)
    contrasts = pd.concat(contrasts, ignore_index=True)
    cells.to_csv(os.path.join(RESULTS, "dcor_humanhuman_cells.csv"), index=False)
    contrasts.to_csv(os.path.join(RESULTS, "dcor_humanhuman_contrasts.csv"), index=False)

    print("\n=== Same rubric item: pooled dCor^2 with 95% CI ===")
    for ds in ("CIP", "Menagerie"):
        print(f"\n  {ds}")
        for _, r in cells[cells.dataset == ds].iterrows():
            print(f"    {r.cell:14s} {r.dcor_sq:.3f}  [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                  f"   (k={r.n_pairs:4d} pairs, median paired n={r.paired_n_median})")

    print("\n=== Contrasts (same bootstrap draws, so dependence is handled) ===")
    for ds in ("CIP", "Menagerie"):
        print(f"\n  {ds}")
        for _, r in contrasts[contrasts.dataset == ds].iterrows():
            sig = "***" if r.p_value < 0.05 else "   "
            print(f"    {r.contrast:26s} {r['diff']:+.3f}  "
                  f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r.p_value:.4f} {sig}")

    print("\nsaved: results/dcor_humanhuman_cells.csv, dcor_humanhuman_contrasts.csv")


if __name__ == "__main__":
    main()
