"""
analysis/rq2_combined_menagerie.py -- RQ2 combined G-study for Menagerie:
rater TYPE (human vs LLM) as a fixed facet, in one model.

Why a Menagerie-specific script: rq2_combined.py assumes a balanced
student x (rater:type) cube (CIP), which doesn't hold here -- humans are
nested in cohorts (each submission scored by only one human group), LLMs
are crossed (all 5 LLMs score every clean submission). So we fit a mixed
model with all variance components + a fixed type effect:

  score ~ 1 + C(rater_type)
          + (1|group)            nuisance: between-cohort mean drift
          + (1|submission)       shared submission signal across systems
          + (1|submission:type)  <<< THE NEW TERM: do systems rank
                                     submissions differently?
          + (1|rater)            within-type rater leniency
                                 (rater IDs are unique across types, so
                                  this is rater nested in type)
          + residual             within-cell noise

Single dummy grouping variable; everything goes through vc_formula.
Bootstrap = cluster-resample submissions WITH replacement (rename dupes
to keep them as distinct random-effect levels), 95% percentile CIs.

Derived: system-agreement correlation between the human and LLM true
submission scores,
  r_sys = s2_submission / (s2_submission + s2_sub_type)
= 1 iff sigma^2(submission x type) = 0; the G-theory twin of the
disattenuated correlation in rq2_gstudy_menagerie.py.

Run:  python analysis/rq2_combined_menagerie.py
      N_BOOT_COMBINED=N python analysis/rq2_combined_menagerie.py
"""

import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_gstudy_menagerie import KEYS, SKILLS, RESULTS, COMPILED, load, ci

N_BOOT = int(os.environ.get("N_BOOT_COMBINED", "200"))
SEED = 0


# ---------- 1. assemble combined long table for one key ----------
def combined_long(human_df, llm_X, clean_subs, llms, key):
    """Long table for one key with columns:
       submission(str), rater(str, unique across types via H_/L_ prefix),
       rater_type({'human','llm'}), group(str: human cohort or 'GLLM'),
       sub_type(str: submission + type, the interaction level), score(float).
    """
    h = human_df[human_df.skill == key][["submission", "rater", "group", "score"]].copy()
    h["rater"] = "H_" + h["rater"].astype(str)
    h["group"] = "G" + h["group"].astype(str)
    h["rater_type"] = "human"

    n_sub = len(clean_subs)
    n_llm = len(llms)
    X = llm_X[key]  # (n_sub, n_llm)
    ll = pd.DataFrame({
        "submission": np.tile(clean_subs, n_llm),
        "rater": np.repeat([f"L_{m}" for m in llms], n_sub),
        "score": X.T.ravel(),
    })
    ll = ll[np.isfinite(ll["score"])].copy()
    ll["group"] = "GLLM"
    ll["rater_type"] = "llm"

    df = pd.concat([h, ll], ignore_index=True)
    df["sub_type"] = df["submission"].astype(str) + "_" + df["rater_type"]
    df["_one"] = 1
    return df


# ---------- 2. mixed-model variance components ----------
def combined_components(df):
    """Fit the combined mixed model, return variance components +
    type fixed effect (human mean - LLM mean) + r_sys."""
    md = smf.mixedlm(
        "score ~ C(rater_type)",
        data=df,
        groups=df["_one"],
        vc_formula={
            "group":      "0 + C(group)",
            "submission": "0 + C(submission)",
            "sub_type":   "0 + C(sub_type)",
            "rater":      "0 + C(rater)",
        },
    )
    mdf = md.fit(reml=True, method="lbfgs")
    # vcomp order is alphabetical
    s2_group, s2_rater, s2_sub_type, s2_submission = (
        float(mdf.vcomp[0]), float(mdf.vcomp[1]),
        float(mdf.vcomp[2]), float(mdf.vcomp[3]))
    s2_resid = float(mdf.scale)
    # type fixed effect: model uses C(rater_type) with 'human' as baseline,
    # coefficient is (llm - human). Flip sign so type_effect = human - llm.
    coef_name = [n for n in mdf.fe_params.index if "rater_type" in n]
    type_effect = -float(mdf.fe_params[coef_name[0]]) if coef_name else float("nan")
    r_sys = s2_submission / (s2_submission + s2_sub_type) \
        if (s2_submission + s2_sub_type) > 0 else float("nan")
    return dict(
        s2_submission=s2_submission, s2_sub_type=s2_sub_type,
        s2_rater=s2_rater, s2_group=s2_group, s2_resid=s2_resid,
        type_effect=type_effect, r_sys=r_sys,
    )


# ---------- 3. bootstrap ----------
def bootstrap(human_df, llm_X, clean_subs, llms, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(clean_subs)
    subs = np.array(clean_subs)
    by_sub = {s: human_df[human_df.submission == s] for s in clean_subs}

    draws = {key: defaultdict(list) for key in KEYS}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # rebuild human side with renamed duplicate subs
        chunks = []
        new_subs = []
        for i, j in enumerate(idx):
            s = subs[j]
            new = f"{s}__{i}"
            new_subs.append(new)
            ch = by_sub[s].copy()
            ch["submission"] = new
            chunks.append(ch)
        human_b = pd.concat(chunks, ignore_index=True)
        llm_b = {key: X[idx] for key, X in llm_X.items()}
        for key in KEYS:
            df_b = combined_long(human_b, llm_b, new_subs, llms, key)
            try:
                c = combined_components(df_b)
            except Exception:
                c = {k: float("nan") for k in
                     ("s2_submission", "s2_sub_type", "s2_rater", "s2_group",
                      "s2_resid", "type_effect", "r_sys")}
            for k, v in c.items():
                draws[key][k].append(v)
        if (b + 1) % 20 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return draws


def boot_pvalue(d, n_boot):
    a = np.asarray(d, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    p = 2 * min((a <= 0).mean(), (a >= 0).mean())
    return max(p, 1.0 / n_boot)


# ---------- 4. figure ----------
def plot_combined(point, draws, out_path):
    sns.set_theme(style="whitegrid")
    short = {"correctness": "correct", "code_elegance": "elegance",
             "readability": "readable", "documentation": "docs",
             "composite": "COMPOSITE"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(KEYS))

    # A: variance-share stack
    comps = [("s2_submission", "σ²(submission)",      "#4C72B0"),
             ("s2_sub_type",   "σ²(submission×type)", "#C44E52"),
             ("s2_rater",      "σ²(rater:type)",      "#DD8452"),
             ("s2_group",      "σ²(group)",           "#937860"),
             ("s2_resid",      "σ²(residual)",        "#8C8C8C")]
    bottom = np.zeros(len(KEYS))
    for stat, label, color in comps:
        vals = np.array([point[k][stat] for k in KEYS])
        tot = np.array([sum(point[k][s] for s, _, _ in comps) for k in KEYS])
        share = np.where(tot > 0, vals / tot, 0)
        axA.bar(x, share, 0.6, bottom=bottom, label=label, color=color)
        bottom += share
    axA.set_xticks(x)
    axA.set_xticklabels([short[k] for k in KEYS], rotation=20, ha="right")
    axA.set_ylabel("share of total variance")
    axA.set_title("Variance composition — combined design", fontweight="bold")
    axA.legend(fontsize=8, loc="upper right")

    # B: system-agreement correlation with CI
    rs = [point[k]["r_sys"] for k in KEYS]
    los, his = [], []
    for k in KEYS:
        lo, hi = ci(draws[k]["r_sys"])
        los.append(max(point[k]["r_sys"] - lo, 0))
        his.append(max(hi - point[k]["r_sys"], 0))
    axB.bar(x, rs, 0.6, color="#55A868")
    axB.errorbar(x, rs, yerr=[los, his], fmt="none",
                 ecolor="black", capsize=4, lw=1.2)
    axB.axhline(1.0, ls="--", color="gray", lw=1)
    axB.set_xticks(x)
    axB.set_xticklabels([short[k] for k in KEYS], rotation=20, ha="right")
    axB.set_ylim(0, 1.1)
    axB.set_ylabel("system-agreement correlation  r_sys")
    axB.set_title("Do human & LLM systems rank submissions alike?",
                  fontweight="bold")

    fig.suptitle("Menagerie combined G-study — rater type as a fixed facet",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


# ---------- main ----------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    human_df, llm_X, clean_subs, llms = load(COMPILED)
    print(f"models: {llms}\n")

    print("point estimates (combined mixed model, one per key)...")
    point = {}
    for key in KEYS:
        df = combined_long(human_df, llm_X, clean_subs, llms, key)
        point[key] = combined_components(df)
        d = point[key]
        print(f"  {key:14s} σ²(sub)={d['s2_submission']:7.3f}  "
              f"σ²(sub×type)={d['s2_sub_type']:7.3f}  "
              f"σ²(rater)={d['s2_rater']:7.3f}  "
              f"σ²(group)={d['s2_group']:7.3f}  "
              f"σ²(resid)={d['s2_resid']:7.3f}  "
              f"r_sys={d['r_sys']:.3f}  "
              f"type(h-l)={d['type_effect']:+.3f}")

    print(f"\nrunning {N_BOOT} submission-level bootstrap replicates "
          f"(set N_BOOT_COMBINED=N to override)...")
    draws = bootstrap(human_df, llm_X, clean_subs, llms)

    # ---- table ----
    rows = []
    for key in KEYS:
        row = {"key": key}
        for k, v in point[key].items():
            lo, hi = ci(draws[key][k])
            row[k] = v
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "gstudy_menagerie_combined.csv"),
               index=False)

    # ---- figure ----
    fig_path = os.path.join(RESULTS, "gstudy_menagerie_combined_figure.png")
    plot_combined(point, draws, fig_path)

    # ---- report ----
    print("\nCOMBINED-DESIGN VARIANCE COMPONENTS  (point estimate + 95% CI)")
    print(f"  {'item':14s} {'σ²(sub)':>12} {'σ²(sub×type)':>14} "
          f"{'σ²(rater)':>12} {'σ²(group)':>12} {'σ²(resid)':>12}")
    for key in KEYS:
        d = point[key]
        print(f"  {key:14s} {d['s2_submission']:12.3f} {d['s2_sub_type']:14.3f} "
              f"{d['s2_rater']:12.3f} {d['s2_group']:12.3f} {d['s2_resid']:12.3f}")

    print("\nSYSTEM-AGREEMENT  (do human & LLM systems rank submissions alike?)")
    print("  r_sys = 1 -> identical ranking; < 1 -> systems differ")
    for key in KEYS:
        lo, hi = ci(draws[key]["r_sys"])
        st_lo, st_hi = ci(draws[key]["s2_sub_type"])
        flag = "  systems DIFFER" if hi < 1.0 and st_lo > 0 else ""
        print(f"  {key:14s} r_sys={point[key]['r_sys']:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"σ²(sub×type)={point[key]['s2_sub_type']:.3f} "
              f"[{st_lo:.3f}, {st_hi:.3f}]{flag}")

    print("\nTYPE FIXED EFFECT  (mean human - mean LLM score on the rubric scale)")
    for key in KEYS:
        lo, hi = ci(draws[key]["type_effect"])
        p = boot_pvalue(draws[key]["type_effect"], len(draws[key]["type_effect"]))
        sig = "***" if p < 0.05 else "   "
        print(f"  {key:14s} {point[key]['type_effect']:+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]  p={p:.4f} {sig}")

    print(f"\nsaved:")
    print(f"  results/gstudy_menagerie_combined.csv")
    print(f"  results/gstudy_menagerie_combined_figure.png")
    print(f"\ntotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
