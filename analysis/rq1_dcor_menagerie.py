"""
analysis/rq1_dcor_menagerie.py -- RQ1 distance-correlation analysis for Menagerie.

Mirrors the CIP rq1_dcor.py 2x3 figure:
  rows    = {Same skill, Different skills}
  columns = {Human-LLM, Between-LLM, Within-LLM}
The Within-LLM same-skill quadrant is skipped (degenerate without prompt/run
variation), exactly as in the CIP analysis. No Human-Human pairings -- CIP
doesn't compute them either.

Menagerie-specific adaptations:
  - Input is the long table runs/menagerie_compiled.csv (compile_menagerie.py).
  - Humans only graded their group's ~40 submissions, so human score vectors are
    length 279 with NaN where the rater didn't grade. dCor is then computed on
    the pairwise-complete subset (NaN-masked) -- so Human-LLM dCor uses ~40 paired
    points per pair, Between-LLM / Within-LLM uses the full 279.
  - Bootstrap resamples the 279 submissions globally (same resample applied to
    every NaN-padded vector). For LLM-LLM cells this preserves cross-rater
    alignment as in CIP; for Human-LLM cells it naturally re-intersects each
    human-LLM pair to whichever resampled subs that human originally graded.

Run: python analysis/rq1_dcor_menagerie.py
"""

import itertools
import os
from collections import defaultdict

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILED = os.path.join(ROOT, "runs", "menagerie_compiled.csv")
RESULTS = os.path.join(ROOT, "results")

SKILLS = ["correctness", "code_elegance", "readability", "documentation"]

YMAX = 0.90        # share with the CIP figure's y-axis cap
N_BOOT = 2000      # bootstrap replicates
SEED = 0
MIN_N = 10         # min paired (non-NaN) points required for dCor

SAME, DIFF = "On the Same Rubric Item", "On Different Rubric Items"
C_HL = "Human-LLM Correlations"
C_LL = "Between-LLMs Correlations"
C_WL = "Within-LLM Correlations"


# ---------- 1. load long table -> NaN-padded per-rater vectors ----------
def load_vectors(path):
    """Returns (vec, humans, llms, submissions).

    vec[(rater_type, rater_name, skill)] = float array of length n_submissions,
    NaN where that rater didn't grade that submission. All vectors share the
    same submission ordering so any positional index lines up.
    """
    df = pd.read_csv(path)
    df = df[df["score"].notna()].copy()
    df["score"] = df["score"].astype(float)
    df["submission"] = df["submission"].astype(int)
    submissions = sorted(df["submission"].unique())
    pos = {s: i for i, s in enumerate(submissions)}
    n = len(submissions)

    humans = sorted(r.split("human_", 1)[1]
                    for r in df.loc[df.rater_type == "human", "rater"].unique())
    llms = sorted(r.split("llm_", 1)[1]
                  for r in df.loc[df.rater_type == "llm", "rater"].unique())

    vec = {}
    for rtype, names in (("human", humans), ("llm", llms)):
        for name in names:
            rcol = f"{rtype}_{name}"
            for sk in SKILLS:
                arr = np.full(n, np.nan)
                sub = df[(df.rater == rcol) & (df.skill == sk)]
                for s, v in zip(sub["submission"], sub["score"]):
                    arr[pos[s]] = v
                vec[(rtype, name, sk)] = arr
    return vec, humans, llms, submissions


# ---------- 2. NaN-aware squared dCor ----------
def dcor_sq(x, y, min_n=MIN_N):
    """Unbiased squared distance correlation on the pairwise-complete subset.
    Returns NaN if fewer than `min_n` paired non-NaN points are available.
    (U-centering can produce small negative values that are noise -> floor 0.)"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if int(mask.sum()) < min_n:
        return np.nan
    return max(float(dcor.u_distance_correlation_sqr(x[mask], y[mask])), 0.0)


# ---------- 3. the 6 quadrants (CIP layout, Q3 skipped) ----------
def pairwise_records(vec, humans, llms):
    """All pairwise dcor^2 values as (model, row, col, value) tuples."""
    ordered = [(a, b) for a in SKILLS for b in SKILLS if a != b]   # 12 ordered
    unordered = list(itertools.combinations(SKILLS, 2))            # 6 unordered
    out = []
    for m in llms:
        others = [x for x in llms if x != m]

        # Q1 Human-LLM, same skill
        for h in humans:
            for t in SKILLS:
                out.append((m, SAME, C_HL,
                            dcor_sq(vec[("human", h, t)], vec[("llm", m, t)])))
        # Q2 Between-LLM, same skill
        for m2 in others:
            for t in SKILLS:
                out.append((m, SAME, C_LL,
                            dcor_sq(vec[("llm", m, t)], vec[("llm", m2, t)])))
        # Q3 Within-LLM, same skill -- SKIPPED (needs prompt or run variation)

        # Q4 Human-LLM, different skill
        for h in humans:
            for th, tm in ordered:
                out.append((m, DIFF, C_HL,
                            dcor_sq(vec[("human", h, th)], vec[("llm", m, tm)])))
        # Q5 Between-LLM, different skill
        for m2 in others:
            for tm, to in ordered:
                out.append((m, DIFF, C_LL,
                            dcor_sq(vec[("llm", m, tm)], vec[("llm", m2, to)])))
        # Q6 Within-LLM, different skill
        for ta, tb in unordered:
            out.append((m, DIFF, C_WL,
                        dcor_sq(vec[("llm", m, ta)], vec[("llm", m, tb)])))
    return out


def aggregate(records):
    """records -> (bar means by (model,row,col), quadrant means by (row,col))."""
    bar_sum, bar_n = defaultdict(float), defaultdict(int)
    quad_sum, quad_n = defaultdict(float), defaultdict(int)
    for m, r, c, v in records:
        if not np.isfinite(v):
            continue
        bar_sum[(m, r, c)] += v
        bar_n[(m, r, c)] += 1
        quad_sum[(r, c)] += v
        quad_n[(r, c)] += 1
    bar = {k: bar_sum[k] / bar_n[k] for k in bar_sum if bar_n[k] > 0}
    quad = {k: quad_sum[k] / quad_n[k] for k in quad_sum if quad_n[k] > 0}
    return bar, quad


# ---------- 4. submission-level bootstrap ----------
def bootstrap(vec, humans, llms, n_boot=N_BOOT, seed=SEED):
    """Resample n=279 submissions WITH replacement; apply the SAME resample to
    every rater-vector so cross-rater alignment is preserved (and human NaN
    masks travel along with the resample). Recompute the whole pipeline."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(vec.values())))
    bar_draws, quad_draws = defaultdict(list), defaultdict(list)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vb = {k: v[idx] for k, v in vec.items()}
        bar, quad = aggregate(pairwise_records(vb, humans, llms))
        for k, val in bar.items():
            bar_draws[k].append(val)
        for k, val in quad.items():
            quad_draws[k].append(val)
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return ({k: np.array(v) for k, v in bar_draws.items()},
            {k: np.array(v) for k, v in quad_draws.items()})


def percentile_ci(draws, point, level=95):
    """Bias-corrected (shifted) percentile CI.

    WHY NOT THE RAW PERCENTILES: resampling submissions WITH replacement
    duplicates submissions, and duplicates are exact ties in both distance
    matrices, which adds spurious dependence and inflates dCor^2. That
    inflation is O(1/n), so it is ~10x worse for the Human-LLM cells (paired
    n ~ 40, since each human graded only their group's submissions) than for
    the LLM-LLM cells (n = 279). Left uncorrected it pushed every Human-LLM
    raw percentile interval ENTIRELY ABOVE its own point estimate.

    So we subtract the bootstrap's own bias estimate b = mean(draws) - point
    from the percentiles: the interval keeps the bootstrap's width but is
    recentred on the point estimate. For the n = 279 cells b is ~0.008, i.e.
    this is very nearly a no-op there.
    """
    lo = (100 - level) / 2
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    b = a.mean() - point
    return float(np.percentile(a, lo)) - b, float(np.percentile(a, 100 - lo)) - b


def boot_pvalue(diff_draws, n_boot):
    """Two-sided bootstrap p-value for H0: difference = 0."""
    d = np.asarray(diff_draws, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return max(p, 1.0 / n_boot)


# ---------- 5. difference test: Same-skill Between-LLM minus Human-LLM ----------
def difference_test(point_bar, point_quad, bar_draws, quad_draws, llms):
    rows = []
    d = quad_draws[(SAME, C_LL)] - quad_draws[(SAME, C_HL)]
    obs = point_quad[(SAME, C_LL)] - point_quad[(SAME, C_HL)]
    lo, hi = percentile_ci(d, obs)
    rows.append(["POOLED", obs, lo, hi, boot_pvalue(d - (np.nanmean(d) - obs), len(d))])
    for m in llms:
        dm = bar_draws[(m, SAME, C_LL)] - bar_draws[(m, SAME, C_HL)]
        obs_m = point_bar[(m, SAME, C_LL)] - point_bar[(m, SAME, C_HL)]
        lo_m, hi_m = percentile_ci(dm, obs_m)
        dm = dm - (np.nanmean(dm) - obs_m)   # same bias shift for the p-value
        rows.append([m, obs_m, lo_m, hi_m, boot_pvalue(dm, len(dm))])
    return pd.DataFrame(rows, columns=["scope", "diff", "ci_lo", "ci_hi", "p_value"])


# ---------- 6. figure ----------
def plot_figure3(point_bar, bar_ci, llms, out_path):
    sns.set_theme(style="whitegrid")
    palette = dict(zip(llms, sns.color_palette("tab10", len(llms))))
    layout = [[(SAME, C_HL), (SAME, C_LL), (SAME, C_WL)],
              [(DIFF, C_HL), (DIFF, C_LL), (DIFF, C_WL)]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    x = np.arange(len(llms))

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            row, col = layout[i][j]
            keys = [(m, row, col) for m in llms]
            if all(k in point_bar for k in keys):
                heights = np.array([point_bar[k] for k in keys])
                ci_lo = np.array([bar_ci[k][0] for k in keys])
                ci_hi = np.array([bar_ci[k][1] for k in keys])
                # No clipping: a point estimate outside its own CI is a bug in
                # the CI, and clipping would render it as a plausible-looking
                # one-sided spike instead of failing.
                assert np.all(ci_lo <= heights + 1e-9) and np.all(heights <= ci_hi + 1e-9), (
                    f"point estimate outside its CI in panel {row} / {col}: "
                    f"{list(zip(llms, heights, ci_lo, ci_hi))}"
                )
                yerr = np.vstack([heights - ci_lo, ci_hi - heights])
                ax.bar(x, heights, width=0.7, color=[palette[m] for m in llms])
                ax.errorbar(x, heights, yerr=yerr, fmt="none",
                            ecolor="black", capsize=4, lw=1.2)
                ax.set_xticks(x)
                ax.set_xticklabels(llms, rotation=30, ha="right")
            else:
                # blank panel: hide it WITHOUT set_yticks([]) -- the axes are
                # sharey=True, so clearing ticks here would clear them on all six.
                ax.set_xticks([])
                ax.tick_params(left=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(False)
                ax.set_facecolor("white")
            ax.set_ylim(0, YMAX)
            ax.set_yticks(np.arange(0, YMAX + 1e-9, 0.1))
            if i == 0:
                ax.set_title(col, fontsize=12, fontweight="bold")
        # leftmost panel of each row carries the y tick labels
        axes[i, 0].tick_params(left=True, labelleft=True)
        axes[i, 0].annotate(layout[i][0][0], xy=(-0.30, 0.5),
                            xycoords="axes fraction", rotation=90,
                            fontsize=12, fontweight="bold",
                            ha="center", va="center")

    fig.supylabel("Mean Squared Bias-Corrected Distance Correlation")
    fig.suptitle("Menagerie -- Distance Correlation of Gradings "
                 "(95% submission-bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


# ---------- main ----------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    vec, humans, llms, submissions = load_vectors(COMPILED)
    print(f"loaded {COMPILED}")
    print(f"  submissions={len(submissions)}  humans={len(humans)}  llms={len(llms)}")

    # diagnostic: median paired-n per quadrant
    def median_pair_n(pair_iter):
        ns = []
        for x, y in pair_iter:
            ns.append(int((~(np.isnan(x) | np.isnan(y))).sum()))
        return int(np.median(ns)) if ns else 0
    h_l_pairs = [(vec[("human", h, t)], vec[("llm", m, t)])
                 for h in humans for m in llms for t in SKILLS]
    l_l_pairs = [(vec[("llm", m, t)], vec[("llm", m2, t)])
                 for m in llms for m2 in llms if m != m2 for t in SKILLS]
    print(f"  paired-n  Human-LLM same-skill (median): {median_pair_n(h_l_pairs)}")
    print(f"  paired-n  Between-LLM same-skill (median): {median_pair_n(l_l_pairs)}")

    # point estimates
    records = pairwise_records(vec, humans, llms)
    tidy = pd.DataFrame(records,
                        columns=["Model", "Condition_Row", "Condition_Col", "dCor_Sq"])
    tidy.to_csv(os.path.join(RESULTS, "dcor_menagerie_pairwise.csv"), index=False)
    point_bar, point_quad = aggregate(records)
    valid = sum(1 for *_, v in records if np.isfinite(v))
    print(f"point estimates: {valid}/{len(records)} pairwise values finite")

    # bootstrap
    print(f"running {N_BOOT} submission-level bootstrap replicates...")
    bar_draws, quad_draws = bootstrap(vec, humans, llms)
    bar_ci = {k: percentile_ci(v, point_bar[k]) for k, v in bar_draws.items()}

    ci_df = pd.DataFrame(
        [{"Model": m, "Condition_Row": r, "Condition_Col": c,
          "dCor_Sq": point_bar[(m, r, c)], "ci_lo": lo, "ci_hi": hi}
         for (m, r, c), (lo, hi) in bar_ci.items()]
    ).sort_values(["Condition_Row", "Condition_Col", "Model"])
    ci_df.to_csv(os.path.join(RESULTS, "dcor_menagerie_ci.csv"), index=False)

    diff_df = difference_test(point_bar, point_quad, bar_draws, quad_draws, llms)
    diff_df.to_csv(os.path.join(RESULTS, "dcor_menagerie_difference_test.csv"),
                   index=False)

    fig_path = os.path.join(RESULTS, "dcor_menagerie_figure3.png")
    plot_figure3(point_bar, bar_ci, llms, fig_path)

    # ---- report ----
    print("\nmean dCor^2 with 95% CI, by quadrant and model:")
    quadrants = [(SAME, C_HL), (SAME, C_LL),
                 (DIFF, C_HL), (DIFF, C_LL), (DIFF, C_WL)]
    for (r, c) in quadrants:
        print(f"\n  {r}  |  {c}")
        for m in llms:
            if (m, r, c) not in point_bar:
                continue
            lo, hi = bar_ci[(m, r, c)]
            print(f"    {m:24s} {point_bar[(m, r, c)]:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n=== DIFFERENCE TEST: Same-Skill  Between-LLM  -  Human-LLM ===")
    print("  (positive = LLMs agree with each other more than with humans)")
    for _, row in diff_df.iterrows():
        sig = "***" if row.p_value < 0.05 else "   "
        print(f"  {row.scope:24s} diff={row['diff']:+.3f}  "
              f"95% CI [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]  "
              f"p={row.p_value:.4f} {sig}")

    print(f"\nsaved: results/dcor_menagerie_pairwise.csv, dcor_menagerie_ci.csv, "
          f"dcor_menagerie_difference_test.csv, dcor_menagerie_figure3.png")


if __name__ == "__main__":
    main()
