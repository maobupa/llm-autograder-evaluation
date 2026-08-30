"""
analysis/rq1_dcor.py — RQ1 distance-correlation analysis for the CIP dataset.

(Not named dcor.py on purpose: `import dcor` from a file named dcor.py would
import the file itself instead of the installed dcor library.)

Reproduces the 2x3 "Figure 3" panel of squared, bias-corrected (U-statistic)
distance correlations between CIP graders, with **transcript-level bootstrap
95% CIs** and a **bootstrap difference test**.

Pipeline:
  1. load runs/cip_compiled.csv (wide) -> long format
  2. dcor^2 for the 6 quadrants (rows {Same, Different task} x
     columns {Human-LLM, Between-LLM, Within-LLM})
  3. transcript-level bootstrap (resample the 100 transcripts WITH replacement,
     apply the SAME resample to every rater/task vector, recompute everything)
     -> proper 95% CIs
  4. difference test: same-task Between-LLM minus Human-LLM
  5. outputs -> results/dcor_cip_{pairwise,ci,difference_test}.csv
             -> results/dcor_cip_figure3.png

Run:  python analysis/rq1_dcor.py

WHY THE BOOTSTRAP: the uncertainty that matters is "which 100 transcripts did
we sample." Resampling transcripts and recomputing the whole pipeline captures
exactly that — and handles the dependence between sub-comparisons automatically,
which seaborn's naive errorbar over sub-comparisons does not.
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
COMPILED = os.path.join(ROOT, "runs", "cip_compiled.csv")
RESULTS = os.path.join(ROOT, "results")

TASKS = ["input", "conditional_logic", "printing", "syntax_errors"]
HUMANS = ["human_1", "human_2", "human_3", "human_4", "human_5"]
LLMS = ["gpt-4.1", "o3", "gemini-2.5-flash", "sonnet-4.6", "deepseek"]

YMAX = 0.90      # dCor^2 axis cap (same-task agreement reaches ~0.76)
N_BOOT = 2000    # bootstrap replicates (~90s)
SEED = 0

SAME, DIFF = "On the Same Rubric Item", "On Different Rubric Items"
C_HL = "Human-LLM Correlations"
C_LL = "Between-LLMs Correlations"
C_WL = "Within-LLM Correlations"


# ---------- 1. load wide -> long ----------
def load_long(path):
    """runs/cip_compiled.csv is wide (<rater>_<item> columns); melt to long
    columns [transcript_id, rater_type, rater_id, task_id, score]."""
    wide = pd.read_csv(path)
    recs = []
    for col in wide.columns:
        if col == "student_id":
            continue
        task = next(t for t in TASKS if col.endswith("_" + t))
        rater = col[: -(len(task) + 1)]
        rtype = "human" if rater in HUMANS else "llm"
        for sid, score in zip(wide["student_id"], wide[col]):
            recs.append((sid, rtype, rater, task, score))
    return pd.DataFrame(
        recs, columns=["transcript_id", "rater_type", "rater_id", "task_id", "score"]
    )


# ---------- 2. helpers ----------
def get_score_vector(df, rater_type, rater_id, task_id):
    """1-D score array for one (rater, task), sorted by transcript_id so any
    two returned vectors are aligned transcript-for-transcript."""
    sub = df[
        (df.rater_type == rater_type)
        & (df.rater_id == rater_id)
        & (df.task_id == task_id)
    ].sort_values("transcript_id")
    return sub["score"].to_numpy(dtype=float)


def dcor_sq(x, y):
    """Unbiased squared distance correlation, floored at 0 (U-centering can
    produce small negative values that are noise)."""
    return max(float(dcor.u_distance_correlation_sqr(x, y)), 0.0)


def build_vectors(df):
    """All 40 (rater, task) score vectors, each aligned by transcript."""
    vec = {}
    for rtype, ids in (("human", HUMANS), ("llm", LLMS)):
        for rid in ids:
            for t in TASKS:
                vec[(rtype, rid, t)] = get_score_vector(df, rtype, rid, t)
    return vec


# ---------- 3. the 6 quadrants ----------
def pairwise_records(vec):
    """All 750 pairwise dcor^2 values as (model, row, col, value) tuples."""
    ordered = [(a, b) for a in TASKS for b in TASKS if a != b]   # 12 ordered task pairs
    unordered = list(itertools.combinations(TASKS, 2))           # 6 unordered task pairs
    out = []
    for m in LLMS:
        others = [x for x in LLMS if x != m]

        # Q1 — Human-LLM, same task (20)
        for h in HUMANS:
            for t in TASKS:
                out.append((m, SAME, C_HL, dcor_sq(vec[("human", h, t)], vec[("llm", m, t)])))
        # Q2 — Between-LLM, same task (16)
        for m2 in others:
            for t in TASKS:
                out.append((m, SAME, C_LL, dcor_sq(vec[("llm", m, t)], vec[("llm", m2, t)])))
        # Q3 — Within-LLM, same task: SKIPPED (needs prompt/run variation)
        # Q4 — Human-LLM, different task (60)
        for h in HUMANS:
            for th, tm in ordered:
                out.append((m, DIFF, C_HL, dcor_sq(vec[("human", h, th)], vec[("llm", m, tm)])))
        # Q5 — Between-LLM, different task (48)
        for m2 in others:
            for tm, to in ordered:
                out.append((m, DIFF, C_LL, dcor_sq(vec[("llm", m, tm)], vec[("llm", m2, to)])))
        # Q6 — Within-LLM, different task (6)
        for ta, tb in unordered:
            out.append((m, DIFF, C_WL, dcor_sq(vec[("llm", m, ta)], vec[("llm", m, tb)])))
    return out


def aggregate(records):
    """records -> (bar_means {(model,row,col): mean}, quad_means {(row,col): mean})."""
    bar_sum, bar_n = defaultdict(float), defaultdict(int)
    quad_sum, quad_n = defaultdict(float), defaultdict(int)
    for m, r, c, v in records:
        bar_sum[(m, r, c)] += v
        bar_n[(m, r, c)] += 1
        quad_sum[(r, c)] += v
        quad_n[(r, c)] += 1
    bar = {k: bar_sum[k] / bar_n[k] for k in bar_sum}
    quad = {k: quad_sum[k] / quad_n[k] for k in quad_sum}
    return bar, quad


# ---------- 4. transcript-level bootstrap ----------
def bootstrap(vec, n_boot=N_BOOT, seed=SEED):
    """Resample the n transcripts WITH replacement; apply the SAME resample to
    every (rater, task) vector so the cross-rater alignment is preserved; then
    recompute the whole pipeline. Returns per-bar and per-quadrant draws."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(vec.values())))
    bar_draws, quad_draws = defaultdict(list), defaultdict(list)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                       # one resample, shared by all vectors
        vb = {k: v[idx] for k, v in vec.items()}
        bar, quad = aggregate(pairwise_records(vb))
        for k, val in bar.items():
            bar_draws[k].append(val)
        for k, val in quad.items():
            quad_draws[k].append(val)
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    bar_draws = {k: np.array(v) for k, v in bar_draws.items()}
    quad_draws = {k: np.array(v) for k, v in quad_draws.items()}
    return bar_draws, quad_draws


def percentile_ci(draws, point, level=95):
    """Bias-corrected (shifted) percentile CI -- kept identical to the Menagerie
    analysis so both figures report the same kind of interval.

    Resampling transcripts WITH replacement duplicates transcripts, and
    duplicates are exact ties in both distance matrices, which adds spurious
    dependence and inflates dCor^2. Here every rater graded all 100
    transcripts, so the bias is small and uniform across cells; in Menagerie,
    where the Human-LLM cells have paired n ~ 40, it was large enough to push
    those intervals entirely above their point estimates. Subtracting
    b = mean(draws) - point recentres the interval on the point estimate while
    keeping the bootstrap's width.
    """
    lo = (100 - level) / 2
    a = np.asarray(draws, float)
    b = a.mean() - point
    return float(np.percentile(a, lo)) - b, float(np.percentile(a, 100 - lo)) - b


def boot_pvalue(diff_draws, n_boot):
    """Two-sided bootstrap p-value for H0: difference = 0."""
    p = 2 * min((diff_draws <= 0).mean(), (diff_draws >= 0).mean())
    return max(p, 1.0 / n_boot)  # cannot resolve below 1/B


# ---------- 5. difference test ----------
def difference_test(point_bar, point_quad, bar_draws, quad_draws):
    """Same-task Between-LLM minus Human-LLM: pooled and per model."""
    rows = []
    # pooled across all models
    d = quad_draws[(SAME, C_LL)] - quad_draws[(SAME, C_HL)]
    obs = point_quad[(SAME, C_LL)] - point_quad[(SAME, C_HL)]
    lo, hi = percentile_ci(d, obs)
    rows.append(["POOLED", obs, lo, hi, boot_pvalue(d - (d.mean() - obs), len(d))])
    # per model
    for m in LLMS:
        dm = bar_draws[(m, SAME, C_LL)] - bar_draws[(m, SAME, C_HL)]
        obs_m = point_bar[(m, SAME, C_LL)] - point_bar[(m, SAME, C_HL)]
        lo_m, hi_m = percentile_ci(dm, obs_m)
        dm = dm - (dm.mean() - obs_m)   # same bias shift for the p-value
        rows.append([m, obs_m, lo_m, hi_m, boot_pvalue(dm, len(dm))])
    return pd.DataFrame(rows, columns=["scope", "diff", "ci_lo", "ci_hi", "p_value"])


# ---------- 6. figure ----------
def plot_figure3(point_bar, bar_ci, out_path):
    sns.set_theme(style="whitegrid")
    palette = dict(zip(LLMS, sns.color_palette("tab10", len(LLMS))))
    layout = [[(SAME, C_HL), (SAME, C_LL), (SAME, C_WL)],
              [(DIFF, C_HL), (DIFF, C_LL), (DIFF, C_WL)]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    x = np.arange(len(LLMS))

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            row, col = layout[i][j]
            keys = [(m, row, col) for m in LLMS]
            if all(k in point_bar for k in keys):
                heights = np.array([point_bar[k] for k in keys])
                ci_lo = np.array([bar_ci[k][0] for k in keys])
                ci_hi = np.array([bar_ci[k][1] for k in keys])
                # No clipping: a point estimate outside its own CI is a bug in
                # the CI, and clipping would render it as a plausible-looking
                # one-sided spike instead of failing.
                assert np.all(ci_lo <= heights + 1e-9) and np.all(heights <= ci_hi + 1e-9), (
                    f"point estimate outside its CI in panel {row} / {col}: "
                    f"{list(zip(LLMS, heights, ci_lo, ci_hi))}"
                )
                yerr = np.vstack([heights - ci_lo, ci_hi - heights])
                ax.bar(x, heights, width=0.7, color=[palette[m] for m in LLMS])
                ax.errorbar(x, heights, yerr=yerr, fmt="none",
                            ecolor="black", capsize=4, lw=1.2)
                ax.set_xticks(x)
                ax.set_xticklabels(LLMS, rotation=30, ha="right")
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
        axes[i, 0].annotate(layout[i][0][0], xy=(-0.30, 0.5), xycoords="axes fraction",
                            rotation=90, fontsize=12, fontweight="bold",
                            ha="center", va="center")

    fig.supylabel("Mean Squared Bias-Corrected Distance Correlation")
    fig.suptitle("CIP — Distance Correlation of Gradings (95% transcript-bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = load_long(COMPILED)
    vec = build_vectors(df)

    # point estimates
    records = pairwise_records(vec)
    tidy = pd.DataFrame(records, columns=["Model", "Condition_Row", "Condition_Col", "dCor_Sq"])
    tidy.to_csv(os.path.join(RESULTS, "dcor_cip_pairwise.csv"), index=False)
    point_bar, point_quad = aggregate(records)
    print(f"point estimates: {len(records)} pairwise values")

    # bootstrap
    print(f"running {N_BOOT} transcript-level bootstrap replicates (~90s)...")
    bar_draws, quad_draws = bootstrap(vec)
    bar_ci = {k: percentile_ci(v, point_bar[k]) for k, v in bar_draws.items()}

    ci_df = pd.DataFrame(
        [{"Model": m, "Condition_Row": r, "Condition_Col": c,
          "dCor_Sq": point_bar[(m, r, c)], "ci_lo": lo, "ci_hi": hi}
         for (m, r, c), (lo, hi) in bar_ci.items()]
    ).sort_values(["Condition_Row", "Condition_Col", "Model"])
    ci_df.to_csv(os.path.join(RESULTS, "dcor_cip_ci.csv"), index=False)

    # difference test
    diff_df = difference_test(point_bar, point_quad, bar_draws, quad_draws)
    diff_df.to_csv(os.path.join(RESULTS, "dcor_cip_difference_test.csv"), index=False)

    # figure
    fig_path = os.path.join(RESULTS, "dcor_cip_figure3.png")
    plot_figure3(point_bar, bar_ci, fig_path)

    # ---- report ----
    print("\nmean dCor^2 with 95% CI, by quadrant and model:")
    for (r, c) in [(SAME, C_HL), (SAME, C_LL), (DIFF, C_HL), (DIFF, C_LL), (DIFF, C_WL)]:
        print(f"\n  {r}  |  {c}")
        for m in LLMS:
            lo, hi = bar_ci[(m, r, c)]
            print(f"    {m:18s} {point_bar[(m, r, c)]:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n=== DIFFERENCE TEST: Same-Task  Between-LLM  -  Human-LLM ===")
    print("  (positive = LLMs agree with each other more than with humans)")
    for _, row in diff_df.iterrows():
        sig = "***" if row.p_value < 0.05 else "   "
        print(f"  {row.scope:18s} diff={row['diff']:+.3f}  "
              f"95% CI [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]  p={row.p_value:.4f} {sig}")

    print(f"\nsaved: results/dcor_cip_pairwise.csv, dcor_cip_ci.csv, "
          f"dcor_cip_difference_test.csv, dcor_cip_figure3.png")


if __name__ == "__main__":
    main()
