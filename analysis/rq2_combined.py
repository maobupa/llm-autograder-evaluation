"""
analysis/rq2_combined.py — RQ2 combined G-study: rater TYPE as a fixed facet.

Design:  student x (rater:type)
  - student : random, object of measurement (100)
  - type    : FIXED facet, 2 levels (human, llm) — the whole universe
  - rater   : random, nested in type (5 per type)

Balanced -> standard ANOVA. Variance components (restricted mixed model):
  sigma^2(student)              real student differences shared by both systems
  sigma^2(student x type)       <-- THE NEW TERM: do the two systems rank-order
                                    students differently?
  sigma^2(rater:type)           within-system rater leniency
  sigma^2(student x rater:type) within-system disagreement + residual

Estimators (t = n_types = 2, r = raters/type = 5, p = n_students):
  s2_resid       = MS(sr:t)
  s2_student_type= (MS(st)   - MS(sr:t)) / r
  s2_rater_type  = (MS(r:t)  - MS(sr:t)) / p
  s2_student     = (MS(s)    - MS(st))   / (t*r)

Derived: the system-agreement correlation between the two systems' true
student scores,  r_sys = (s2_student - s2_student_type) /
                         (s2_student + s2_student_type)
(= 1 iff sigma^2(student x type) = 0; the G-theory twin of the disattenuated
correlation in rq2_gstudy.py).

Run:  python analysis/rq2_combined.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rq2_gstudy import (COMPILED, HUMANS, KEYS, LLMS, RESULTS, TASKS, ci,
                        variance_components)

N_BOOT = 2000
SEED = 0
N_TYPE = 2
N_RATER = 5


def build_combined(wide):
    """{key: (100 x 10) matrix}, columns = 5 humans then 5 LLMs."""
    M = {}
    for t in TASKS:
        cols = [f"{h}_{t}" for h in HUMANS] + [f"{m}_{t}" for m in LLMS]
        M[t] = wide[cols].to_numpy(float)
    M["composite"] = sum(M[t] for t in TASKS)
    return M


def combined_components(X, t=N_TYPE, r=N_RATER):
    """ANOVA variance components for the balanced student x (rater:type) design.
    X: (p, t*r), first r columns = type 0, next r = type 1."""
    p = X.shape[0]
    Xt = X.reshape(p, t, r)
    G = X.mean()
    Sbar = X.mean(axis=1)               # (p,)   student means over all raters
    Tbar = Xt.mean(axis=(0, 2))         # (t,)   type means
    Rbar = Xt.mean(axis=0)              # (t, r) rater means
    STbar = Xt.mean(axis=2)             # (p, t) student x type means

    SS_s = t * r * np.sum((Sbar - G) ** 2)
    SS_st = r * np.sum((STbar - Sbar[:, None] - Tbar[None, :] + G) ** 2)
    SS_rt = p * np.sum((Rbar - Tbar[:, None]) ** 2)
    resid = Xt - STbar[:, :, None] - Rbar[None, :, :] + Tbar[None, :, None]
    SS_srt = np.sum(resid ** 2)

    MS_s = SS_s / (p - 1)
    MS_st = SS_st / ((p - 1) * (t - 1))
    MS_rt = SS_rt / (t * (r - 1))
    MS_srt = SS_srt / (t * (p - 1) * (r - 1))

    s2_srt = MS_srt
    s2_st = max((MS_st - MS_srt) / r, 0.0)
    s2_rt = max((MS_rt - MS_srt) / p, 0.0)
    s2_s = max((MS_s - MS_st) / (t * r), 0.0)

    r_sys = (s2_s - s2_st) / (s2_s + s2_st) if (s2_s + s2_st) > 0 else np.nan
    return dict(
        s2_student=s2_s, s2_student_type=s2_st, s2_rater_type=s2_rt,
        s2_resid=s2_srt, type_effect=float(Tbar[0] - Tbar[1]), r_sys=r_sys,
    )


def main():
    os.makedirs(RESULTS, exist_ok=True)
    wide = pd.read_csv(COMPILED)
    M = build_combined(wide)

    # --- validation: combined residual == pooled separate-study sigma^2(s x r) ---
    print("validation — sigma^2(student x rater:type) vs pooled separate studies:")
    for key in KEYS:
        X = M[key]
        comb = combined_components(X)["s2_resid"]
        h = variance_components(X[:, :N_RATER])[2]
        l = variance_components(X[:, N_RATER:])[2]
        print(f"  {key:18s} combined={comb:.4f}   pooled(h,l)=({h:.4f},{l:.4f}) "
              f"mean={ (h + l) / 2:.4f}")

    # --- point estimates ---
    point = {key: combined_components(M[key]) for key in KEYS}

    # --- bootstrap over students ---
    rng = np.random.default_rng(SEED)
    n = M["composite"].shape[0]
    draws = {key: {k: [] for k in point[key]} for key in KEYS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for key in KEYS:
            c = combined_components(M[key][idx])
            for k, v in c.items():
                draws[key][k].append(v)
    print(f"\nbootstrap: {N_BOOT} replicates done\n")

    # --- assemble + save ---
    rows = []
    for key in KEYS:
        row = {"key": key}
        for k, v in point[key].items():
            lo, hi = ci(draws[key][k])
            row[k] = v
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "gstudy_cip_combined.csv"), index=False)

    # --- report ---
    print("COMBINED-DESIGN VARIANCE COMPONENTS  (point estimate)")
    print(f"  {'item':14s} {'σ²(stu)':>9} {'σ²(stu×type)':>13} {'σ²(rat:type)':>13} "
          f"{'σ²(resid)':>10}")
    for key in KEYS:
        d = point[key]
        print(f"  {key:14s} {d['s2_student']:9.3f} {d['s2_student_type']:13.3f} "
              f"{d['s2_rater_type']:13.3f} {d['s2_resid']:10.3f}")

    print("\nSYSTEM-AGREEMENT  (do human & LLM systems rank students alike?)")
    print("  r_sys = 1 -> identical ranking; < 1 -> systems differ")
    for key in KEYS:
        lo, hi = ci(draws[key]["r_sys"])
        st_lo, st_hi = ci(draws[key]["s2_student_type"])
        flag = "  systems DIFFER" if hi < 1.0 and st_lo > 0 else ""
        print(f"  {key:14s} r_sys={point[key]['r_sys']:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"σ²(stu×type)={point[key]['s2_student_type']:.3f} "
              f"[{st_lo:.3f}, {st_hi:.3f}]{flag}")

    print("\nTYPE FIXED EFFECT  (mean human − mean LLM grade; +ve = humans harsher,")
    print("  since higher score = worse on this rubric)")
    for key in KEYS:
        lo, hi = ci(draws[key]["type_effect"])
        sig = "***" if lo * hi > 0 else "   "
        print(f"  {key:14s} {point[key]['type_effect']:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")

    # --- figure ---
    sns.set_theme(style="whitegrid")
    short = {"input": "input", "conditional_logic": "cond_logic",
             "printing": "printing", "syntax_errors": "syntax", "composite": "COMPOSITE"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(KEYS))

    # A: variance composition (shares, stacked)
    comps = [("s2_student", "σ²(student)", "#4C72B0"),
             ("s2_student_type", "σ²(student×type)", "#C44E52"),
             ("s2_rater_type", "σ²(rater:type)", "#DD8452"),
             ("s2_resid", "σ²(student×rater:type)", "#8C8C8C")]
    bottom = np.zeros(len(KEYS))
    for stat, label, color in comps:
        vals = np.array([point[k][stat] for k in KEYS])
        tot = np.array([sum(point[k][s] for s, _, _ in comps) for k in KEYS])
        share = vals / tot
        axA.bar(x, share, 0.6, bottom=bottom, label=label, color=color)
        bottom += share
    axA.set_xticks(x)
    axA.set_xticklabels([short[k] for k in KEYS], rotation=20, ha="right")
    axA.set_ylabel("share of total variance")
    axA.set_title("Variance composition — combined design", fontweight="bold")
    axA.legend(fontsize=8, loc="upper right")

    # B: system-agreement correlation with CI
    rs = [point[k]["r_sys"] for k in KEYS]
    los = [max(rs[i] - ci(draws[KEYS[i]]["r_sys"])[0], 0) for i in range(len(KEYS))]
    his = [max(ci(draws[KEYS[i]]["r_sys"])[1] - rs[i], 0) for i in range(len(KEYS))]
    axB.bar(x, rs, 0.6, color="#55A868")
    axB.errorbar(x, rs, yerr=[los, his], fmt="none", ecolor="black", capsize=4, lw=1.2)
    axB.axhline(1.0, ls="--", color="gray", lw=1)
    axB.set_xticks(x)
    axB.set_xticklabels([short[k] for k in KEYS], rotation=20, ha="right")
    axB.set_ylim(0, 1.1)
    axB.set_ylabel("system-agreement correlation  r_sys")
    axB.set_title("Do human & LLM systems rank students alike?", fontweight="bold")

    fig.suptitle("CIP combined G-study — rater type as a fixed facet",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS, "gstudy_cip_combined_figure.png"),
                dpi=200, bbox_inches="tight")
    print("\nsaved: gstudy_cip_combined.csv, gstudy_cip_combined_figure.png")


if __name__ == "__main__":
    main()
