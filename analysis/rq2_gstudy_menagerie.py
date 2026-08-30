"""
analysis/rq2_gstudy_menagerie.py -- RQ2 Generalizability study for Menagerie.

Two designs, one script:
  HUMAN-only   nested:   score ~ 1 + (1|group) + (1|group:submission)
                                       + (1|group:rater) + residual
               fit with statsmodels.MixedLM (REML).
  LLM-only     crossed:  submission x model  (balanced 5 x N_clean)
               fit with the same balanced-crossed ANOVA estimator CIP uses
               (analysis/rq2_gstudy.variance_components).

For each system and each of {4 skills, composite} we estimate:
  sigma^2(submission), sigma^2(rater), sigma^2(s x r residual),
  plus sigma^2(group) for the human side (nuisance), and the single-grader
  generalizability rho^2 and absolute coefficient Phi.

Bootstrap: resample the N_clean submissions WITH replacement (cluster bootstrap;
renaming sub IDs to keep duplicates as distinct random-effect levels for the
human MixedLM), recompute everything, take percentile CIs. Default N_BOOT=500;
override with `N_BOOT_GSTUDY=N python ...`.

RQ2 hypothesis tests (positive supports "LLMs are a more homogeneous grading
system than humans"):
  sigma^2(rater) human  -  sigma^2(rater) LLM
  sigma^2(s x r) human  -  sigma^2(s x r) LLM
  rho^2(1)       LLM    -  rho^2(1)       human

N_clean = submissions where every LLM has a score on every skill (no errors /
NaNs). The same N_clean set is used for the human-side fit so the two systems
are comparable on identical submissions.

Run: python analysis/rq2_gstudy_menagerie.py
"""

import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")  # suppress statsmodels REML boundary warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# Reuse the CIP balanced-crossed ANOVA estimator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_gstudy import g_coefficients, variance_components as llm_variance_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILED = os.path.join(ROOT, "runs", "menagerie_compiled.csv")
RESULTS = os.path.join(ROOT, "results")

SKILLS = ["correctness", "code_elegance", "readability", "documentation"]
KEYS = SKILLS + ["composite"]

N_BOOT = int(os.environ.get("N_BOOT_GSTUDY", "500"))
SEED = 0


# ---------- 1. load & clean ----------
def load(path):
    """Returns (human_df, llm_X, clean_subs, llms).

    human_df    long table: columns submission(str), rater(str), group(int),
                skill(str includes 'composite'), score(float).
    llm_X[key]  (n_clean, n_models) array, columns sorted by model name.
    clean_subs  list of submission IDs (str) used by BOTH systems.
    """
    df = pd.read_csv(path)
    df = df[df["score"].notna()].copy()
    df["score"] = df["score"].astype(float)
    df["submission"] = df["submission"].astype(int)

    llms_sorted = sorted(df.loc[df.rater_type == "llm", "rater"].unique())

    # Submissions for which every LLM has a score on every skill.
    clean = None
    for sk in SKILLS:
        p = (df[(df.rater_type == "llm") & (df.skill == sk)]
             .pivot(index="submission", columns="rater", values="score"))
        complete = set(p.dropna().index)
        clean = complete if clean is None else clean & complete
    clean_subs_int = sorted(clean)
    print(f"clean submission set (all LLMs have all 4 skills): "
          f"{len(clean_subs_int)} / {df['submission'].nunique()}")

    df = df[df["submission"].isin(clean_subs_int)].copy()

    # LLM matrices per skill (and composite).
    clean_subs_idx = pd.Index(clean_subs_int, name="submission")
    llm_X = {}
    for sk in SKILLS:
        p = (df[(df.rater_type == "llm") & (df.skill == sk)]
             .pivot(index="submission", columns="rater", values="score")
             .reindex(clean_subs_idx)[llms_sorted])
        llm_X[sk] = p.to_numpy(float)
    llm_X["composite"] = sum(llm_X[sk] for sk in SKILLS)

    # Human long table with composite rows appended. Composite per (sub, rater)
    # is the sum of the 4 skill scores -- only counted when the rater scored all
    # 4 skills (raters with missing skills contribute partial data only).
    human = df[df.rater_type == "human"].copy()
    wide = (human.pivot_table(index=["submission", "rater", "group"],
                              columns="skill", values="score")
                .dropna())
    composite = wide.sum(axis=1).reset_index(name="score")
    composite["skill"] = "composite"
    human_keep = human[["submission", "rater", "group", "skill", "score"]]
    human_df = pd.concat([human_keep, composite], ignore_index=True)
    human_df["submission"] = human_df["submission"].astype(str)
    human_df["rater"] = human_df["rater"].astype(str)
    human_df["group"] = human_df["group"].astype(int)

    return human_df, llm_X, [str(s) for s in clean_subs_int], llms_sorted


# ---------- 2. per-system component estimators ----------
def human_components(human_df, key):
    """REML variance components for the nested human design, one key."""
    sub = human_df[human_df.skill == key]
    md = smf.mixedlm("score ~ 1", sub, groups="group", re_formula="1",
                     vc_formula={"submission": "0 + C(submission)",
                                 "rater":      "0 + C(rater)"})
    mdf = md.fit(reml=True, method="lbfgs")
    s2_g = float(mdf.cov_re.iloc[0, 0])
    # vcomp order is alphabetical: ['rater', 'submission']
    s2_r = float(mdf.vcomp[0])
    s2_s = float(mdf.vcomp[1])
    s2_e = float(mdf.scale)
    # single-grader generalizability, same definition as CIP:
    # rho^2 ignores rater bias (cancels in ranking); Phi includes it.
    n_r = 1
    rho2 = s2_s / (s2_s + s2_e / n_r) if (s2_s + s2_e / n_r) > 0 else 0.0
    phi = s2_s / (s2_s + (s2_r + s2_e) / n_r) if (s2_s + (s2_r + s2_e) / n_r) > 0 else 0.0
    return {"s2_student": s2_s, "s2_rater": s2_r, "s2_sr": s2_e,
            "s2_group": s2_g, "total": s2_s + s2_r + s2_e,
            "rho2": rho2, "phi": phi}


def llm_components(X):
    """ANOVA variance components for the balanced-crossed LLM design."""
    s2p, s2r, s2sr = llm_variance_components(X)
    rho2, phi = g_coefficients(s2p, s2r, s2sr, n_r=1)
    return {"s2_student": s2p, "s2_rater": s2r, "s2_sr": s2sr,
            "s2_group": 0.0, "total": s2p + s2r + s2sr,
            "rho2": rho2, "phi": phi}


def estimate(human_df, llm_X):
    out = {}
    for key in KEYS:
        out[("human", key)] = human_components(human_df, key)
        out[("llm", key)] = llm_components(llm_X[key])
    return out


# ---------- 3. bootstrap ----------
def bootstrap(human_df, llm_X, clean_subs, n_boot=N_BOOT, seed=SEED):
    """Resample submissions WITH replacement; refit both systems each iter.

    Human side: duplicated subs are renamed to keep them as distinct random-
    effect levels (cluster / hierarchical bootstrap). LLM side: row-index the
    pivoted matrix with the resample directly."""
    rng = np.random.default_rng(seed)
    n = len(clean_subs)
    subs = np.array(clean_subs)
    by_sub = {s: human_df[human_df.submission == s] for s in clean_subs}

    stat_draws = defaultdict(lambda: defaultdict(list))
    diff_draws = defaultdict(lambda: defaultdict(list))

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # Human: stitch resampled chunks with unique sub IDs
        chunks = []
        for i, j in enumerate(idx):
            s = subs[j]
            ch = by_sub[s].copy()
            ch["submission"] = f"{s}__{i}"
            chunks.append(ch)
        human_b = pd.concat(chunks, ignore_index=True)
        llm_b = {key: X[idx] for key, X in llm_X.items()}

        est = {}
        for key in KEYS:
            est[("human", key)] = human_components(human_b, key)
            est[("llm", key)] = llm_components(llm_b[key])
        for k, d in est.items():
            for stat, val in d.items():
                stat_draws[k][stat].append(val)
        for key in KEYS:
            h, l = est[("human", key)], est[("llm", key)]
            diff_draws[key]["s2_rater"].append(h["s2_rater"] - l["s2_rater"])
            diff_draws[key]["s2_sr"].append(h["s2_sr"] - l["s2_sr"])
            diff_draws[key]["rho2"].append(l["rho2"] - h["rho2"])

        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return stat_draws, diff_draws


def ci(draws, level=95):
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    lo = (100 - level) / 2
    return float(np.percentile(a, lo)), float(np.percentile(a, 100 - lo))


def boot_pvalue(diff_draws, n_boot):
    d = np.asarray(diff_draws, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return max(p, 1.0 / n_boot)


# ---------- 4. disattenuated correlation (point estimate) ----------
def disattenuated(human_df, llm_X, clean_subs, point):
    """corr(human-mean per sub, LLM-mean per sub), disattenuated by each
    system's reliability for its full panel (n_r = 4 humans, n_r = 5 LLMs)."""
    rows = []
    subs_int = [int(s) for s in clean_subs]
    n_h = 4
    n_l = llm_X["composite"].shape[1]
    for key in KEYS:
        # per-sub human mean (within each sub's group of ~4 raters)
        h = human_df[human_df.skill == key]
        h_mean = (h.groupby(h["submission"].astype(int))["score"].mean()
                   .reindex(subs_int).to_numpy(float))
        l_mean = llm_X[key].mean(axis=1)
        mask = np.isfinite(h_mean) & np.isfinite(l_mean)
        if mask.sum() < 3:
            r_obs = float("nan")
        else:
            r_obs = float(np.corrcoef(h_mean[mask], l_mean[mask])[0, 1])
        # reliabilities at full panel size
        ph = point[("human", key)]
        pl = point[("llm", key)]
        rel_h = ph["s2_student"] / (ph["s2_student"] + ph["s2_sr"] / n_h) \
            if (ph["s2_student"] + ph["s2_sr"] / n_h) > 0 else 0.0
        rel_l, _ = g_coefficients(pl["s2_student"], pl["s2_rater"], pl["s2_sr"], n_r=n_l)
        denom = np.sqrt(rel_h * rel_l) if (rel_h > 0 and rel_l > 0) else float("nan")
        r_dis = r_obs / denom if denom and np.isfinite(denom) else float("nan")
        if np.isfinite(r_dis):
            r_dis = min(r_dis, 1.0)
        rows.append({"key": key, "r_obs": r_obs,
                     "rel_human": rel_h, "rel_llm": rel_l,
                     "r_disattenuated": r_dis})
    return pd.DataFrame(rows)


# ---------- 5. figure ----------
def plot_gstudy(point, stat_draws, out_path):
    sns.set_theme(style="whitegrid")
    colors = {"human": "#4C72B0", "llm": "#DD8452"}
    short = {"correctness": "correct", "code_elegance": "elegance",
             "readability": "readable", "documentation": "docs",
             "composite": "COMPOSITE"}
    panels = [
        ("s2_student", "σ²(submission)  share of total", True),
        ("s2_rater",   "σ²(rater)       share of total", True),
        ("s2_sr",      "σ²(s×r residual) share of total", True),
        ("rho2",       "Single-grader generalizability ρ²", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(KEYS))
    w = 0.38

    for ax, (stat, title, as_share) in zip(axes.flat, panels):
        for s, (system, off) in enumerate((("human", -w/2), ("llm", w/2))):
            heights, los, his = [], [], []
            for key in KEYS:
                draws = np.array(stat_draws[(system, key)][stat], float)
                pt = point[(system, key)][stat]
                if as_share:
                    tot = np.array(stat_draws[(system, key)]["total"], float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        draws = draws / tot
                    pt = pt / point[(system, key)]["total"] \
                        if point[(system, key)]["total"] > 0 else 0
                lo, hi = ci(draws)
                heights.append(pt)
                los.append(max(pt - lo, 0))
                his.append(max(hi - pt, 0))
            ax.bar(x + off, heights, w, label=system.upper(), color=colors[system])
            ax.errorbar(x + off, heights, yerr=[los, his], fmt="none",
                        ecolor="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short[k] for k in KEYS], rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend()

    fig.suptitle("Menagerie G-study — Human (nested) vs LLM (crossed) "
                 "(95% submission-bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


# ---------- main ----------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    human_df, llm_X, clean_subs, llms = load(COMPILED)
    print(f"models: {llms}\n")

    print("point estimates...")
    point = estimate(human_df, llm_X)

    print(f"running {N_BOOT} submission-level bootstrap replicates "
          f"(set N_BOOT_GSTUDY=N to override)...")
    stat_draws, diff_draws = bootstrap(human_df, llm_X, clean_subs)

    # ---- variance-component table ----
    vc_rows = []
    for (system, key), d in point.items():
        row = {"system": system, "key": key}
        for stat in ("s2_student", "s2_rater", "s2_sr", "s2_group", "total",
                     "rho2", "phi"):
            lo, hi = ci(stat_draws[(system, key)][stat])
            row[stat] = d[stat]
            row[f"{stat}_lo"], row[f"{stat}_hi"] = lo, hi
        vc_rows.append(row)
    vc = pd.DataFrame(vc_rows).sort_values(["key", "system"])
    vc.to_csv(os.path.join(RESULTS, "gstudy_menagerie_variance_components.csv"),
              index=False)

    # ---- difference test ----
    diff_rows = []
    labels = {"s2_rater": "σ²(rater)  human-LLM",
              "s2_sr":    "σ²(s×r)    human-LLM",
              "rho2":     "ρ²(1)      LLM-human"}
    for key in KEYS:
        for stat in ("s2_rater", "s2_sr", "rho2"):
            d = diff_draws[key][stat]
            if stat == "rho2":
                obs = point[("llm", key)][stat] - point[("human", key)][stat]
            else:
                obs = point[("human", key)][stat] - point[("llm", key)][stat]
            lo, hi = ci(d)
            diff_rows.append({"key": key, "contrast": labels[stat], "diff": obs,
                              "ci_lo": lo, "ci_hi": hi,
                              "p_value": boot_pvalue(d, len(d))})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(os.path.join(RESULTS, "gstudy_menagerie_difference_test.csv"),
                index=False)

    # ---- disattenuated correlation ----
    dis = disattenuated(human_df, llm_X, clean_subs, point)
    dis.to_csv(os.path.join(RESULTS, "gstudy_menagerie_disattenuated.csv"),
               index=False)

    # ---- figure ----
    fig_path = os.path.join(RESULTS, "gstudy_menagerie_figure.png")
    plot_gstudy(point, stat_draws, fig_path)

    # ---- report ----
    print("\nVARIANCE COMPONENTS (point estimate)")
    print(f"  {'system/key':28s} {'σ²(sub)':>9} {'σ²(rater)':>10} "
          f"{'σ²(s×r)':>9} {'ρ²(1)':>8}")
    for key in KEYS:
        for system in ("human", "llm"):
            d = point[(system, key)]
            print(f"  {system+'/'+key:28s} {d['s2_student']:9.3f} "
                  f"{d['s2_rater']:10.3f} {d['s2_sr']:9.3f} {d['rho2']:8.3f}")

    print("\nDIFFERENCE TEST  (positive supports the RQ2 homogeneity claim)")
    for _, r in diff.iterrows():
        sig = "***" if r.p_value < 0.05 else "   "
        print(f"  {r.key:14s} {r.contrast:22s} {r['diff']:+.3f}  "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r.p_value:.4f} {sig}")

    print("\nDISATTENUATED CORRELATION  (human-system mean vs LLM-system mean)")
    for _, r in dis.iterrows():
        print(f"  {r.key:14s} r_obs={r.r_obs:.3f}  rel_h={r.rel_human:.3f}  "
              f"rel_l={r.rel_llm:.3f}  ->  r_disatt={r.r_disattenuated:.3f}")

    print(f"\nsaved:")
    print(f"  results/gstudy_menagerie_variance_components.csv")
    print(f"  results/gstudy_menagerie_difference_test.csv")
    print(f"  results/gstudy_menagerie_disattenuated.csv")
    print(f"  results/gstudy_menagerie_figure.png")
    print(f"\ntotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
