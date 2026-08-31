"""
analysis/rq1_dcor_perhuman.py -- CIP Figure 3, but per individual HUMAN grader.

rq1_dcor.py builds the 2x3 panel with one bar per LLM. This is the mirror
image: one bar per human grader, so individual human idiosyncrasy is visible
the same way individual model idiosyncrasy is in Figure 3.

  rows    {On the Same Rubric Item, On Different Rubric Items}
  columns {Human-LLM, Between-Humans, Within-Human}
  bars    human_1 ... human_5

Only CIP. Menagerie humans graded just their own group's submissions, so most
human pairs share zero submissions and per-human bars are not estimable at a
common n (see rq1_dcor_humanhuman.py).

Pair counts per human h, mirroring the per-LLM script exactly:
  same item   Human-LLM      5 LLMs   x 4 items          = 20
              Between-Human  4 others x 4 items          = 16
              Within-Human   -- empty (one grading pass per human,
                                 exactly as Within-LLM is empty in Fig 3)
  diff item   Human-LLM      5 LLMs   x 12 ordered pairs  = 60
              Between-Human  4 others x 12 ordered pairs  = 48
              Within-Human   6 unordered item pairs       = 6

Everything else -- dcor_sq, transcript-level bootstrap, shift-corrected
percentile CI -- is imported from rq1_dcor so the numbers sit on the same
footing as Figure 3.

Run:  python analysis/rq1_dcor_perhuman.py
"""

import itertools
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_dcor import (COMPILED, HUMANS, LLMS, RESULTS, SEED, TASKS, YMAX,
                      build_vectors, dcor_sq, load_long, percentile_ci)

N_BOOT = int(os.environ.get("N_BOOT_PERHUMAN", "2000"))

SAME, DIFF = "On the Same Rubric Item", "On Different Rubric Items"
C_HL = "Human-LLM Correlations"
C_HH = "Between-Humans Correlations"
C_WH = "Within-Human Correlations"


# ---------- pairwise records, indexed by human ----------
def pairwise_records(vec):
    """All pairwise dcor^2 values as (human, row, col, value) tuples."""
    ordered = [(a, b) for a in TASKS for b in TASKS if a != b]   # 12 ordered item pairs
    unordered = list(itertools.combinations(TASKS, 2))           # 6 unordered item pairs
    out = []
    for h in HUMANS:
        others = [x for x in HUMANS if x != h]

        # same item -- Human-LLM (20)
        for m in LLMS:
            for t in TASKS:
                out.append((h, SAME, C_HL, dcor_sq(vec[("human", h, t)], vec[("llm", m, t)])))
        # same item -- Between-Humans (16)
        for h2 in others:
            for t in TASKS:
                out.append((h, SAME, C_HH, dcor_sq(vec[("human", h, t)], vec[("human", h2, t)])))
        # same item -- Within-Human: SKIPPED (needs repeat grading by the same human)
        # different item -- Human-LLM (60)
        for m in LLMS:
            for th, tm in ordered:
                out.append((h, DIFF, C_HL, dcor_sq(vec[("human", h, th)], vec[("llm", m, tm)])))
        # different item -- Between-Humans (48)
        for h2 in others:
            for ta, tb in ordered:
                out.append((h, DIFF, C_HH, dcor_sq(vec[("human", h, ta)], vec[("human", h2, tb)])))
        # different item -- Within-Human (6)
        for ta, tb in unordered:
            out.append((h, DIFF, C_WH, dcor_sq(vec[("human", h, ta)], vec[("human", h, tb)])))
    return out


def aggregate(records):
    """records -> (bar means {(human,row,col)}, quadrant means {(row,col)})."""
    bar_sum, bar_n = defaultdict(float), defaultdict(int)
    quad_sum, quad_n = defaultdict(float), defaultdict(int)
    for h, r, c, v in records:
        bar_sum[(h, r, c)] += v
        bar_n[(h, r, c)] += 1
        quad_sum[(r, c)] += v
        quad_n[(r, c)] += 1
    return ({k: bar_sum[k] / bar_n[k] for k in bar_sum},
            {k: quad_sum[k] / quad_n[k] for k in quad_sum})


def bootstrap(vec, n_boot=N_BOOT, seed=SEED):
    """Transcript-level resample, shared across every (rater, item) vector."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(vec.values())))
    bar_draws, quad_draws = defaultdict(list), defaultdict(list)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bar, quad = aggregate(pairwise_records({k: v[idx] for k, v in vec.items()}))
        for k, val in bar.items():
            bar_draws[k].append(val)
        for k, val in quad.items():
            quad_draws[k].append(val)
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return ({k: np.array(v) for k, v in bar_draws.items()},
            {k: np.array(v) for k, v in quad_draws.items()})


# ---------- figure ----------
def plot(point_bar, bar_ci, out_path):
    sns.set_theme(style="whitegrid")
    palette = dict(zip(HUMANS, sns.color_palette("tab10", len(HUMANS))))
    layout = [[(SAME, C_HL), (SAME, C_HH), (SAME, C_WH)],
              [(DIFF, C_HL), (DIFF, C_HH), (DIFF, C_WH)]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    x = np.arange(len(HUMANS))

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            row, col = layout[i][j]
            keys = [(h, row, col) for h in HUMANS]
            if all(k in point_bar for k in keys):
                heights = np.array([point_bar[k] for k in keys])
                ci_lo = np.array([bar_ci[k][0] for k in keys])
                ci_hi = np.array([bar_ci[k][1] for k in keys])
                assert np.all(ci_lo <= heights + 1e-9) and np.all(heights <= ci_hi + 1e-9), (
                    f"point estimate outside its CI in panel {row} / {col}: "
                    f"{list(zip(HUMANS, heights, ci_lo, ci_hi))}"
                )
                ax.bar(x, heights, width=0.7, color=[palette[h] for h in HUMANS])
                ax.errorbar(x, heights, yerr=np.vstack([heights - ci_lo, ci_hi - heights]),
                            fmt="none", ecolor="black", capsize=4, lw=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels([h.replace("human_", "human ") for h in HUMANS],
                               rotation=30, ha="right")
            ax.set_ylim(0, YMAX)
            if i == 0:
                ax.set_title(col, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(row, fontsize=11, fontweight="bold")

    fig.suptitle("CIP — Distance Correlation of Gradings, per human grader "
                 "(95% transcript-bootstrap CIs)", fontsize=14, fontweight="bold")
    fig.supylabel("Mean Squared Bias-Corrected Distance Correlation", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    vec = build_vectors(load_long(COMPILED))

    point_bar, point_quad = aggregate(pairwise_records(vec))
    print(f"bootstrap: {N_BOOT} replicates")
    bar_draws, quad_draws = bootstrap(vec)

    bar_ci = {k: percentile_ci(bar_draws[k], point_bar[k]) for k in point_bar}
    quad_ci = {k: percentile_ci(quad_draws[k], point_quad[k]) for k in point_quad}

    rows = [{"human": h, "row": r, "col": c, "dcor2": v,
             "ci_lo": bar_ci[(h, r, c)][0], "ci_hi": bar_ci[(h, r, c)][1]}
            for (h, r, c), v in sorted(point_bar.items())]
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(RESULTS, "dcor_cip_perhuman.csv"), index=False)

    print("\nPOOLED QUADRANTS")
    for (r, c), v in sorted(point_quad.items()):
        lo, hi = quad_ci[(r, c)]
        print(f"  {r:28s} {c:30s} {v:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\nPER-HUMAN, same rubric item")
    for c in (C_HL, C_HH):
        print(f"  {c}")
        for h in HUMANS:
            v = point_bar[(h, SAME, c)]
            lo, hi = bar_ci[(h, SAME, c)]
            print(f"    {h:9s} {v:.3f} [{lo:.3f}, {hi:.3f}]")

    plot(point_bar, bar_ci, os.path.join(RESULTS, "dcor_cip_perhuman.png"))
    print("saved: results/dcor_cip_perhuman.csv")


if __name__ == "__main__":
    main()
