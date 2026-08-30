"""
analysis/appendix_c_outputs.py -- figures and LaTeX tables for Appendix C.

Produces:
  results/appendix_c_dcor_humanhuman.png   Human-Human vs Human-LLM vs Between-LLM,
                                           both datasets, echoing main-text Figure 3
  paper/tables/dcor_humanhuman.tex         same numbers as a table
  paper/tables/gstudy_cip.tex              per-system variance components, CIP
  paper/tables/gstudy_menagerie.tex        per-system variance components, Menagerie

Reads only existing results/*.csv -- no re-estimation, so these stay in sync with
whatever the analysis scripts last wrote.

Run: python analysis/appendix_c_outputs.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "paper", "tables")

CELLS = ["Human-Human", "Human-LLM", "Between-LLM"]
YMAX = 0.90   # shared with the main-text Figure 3 panels

# --- table width control -------------------------------------------------
# Which generated tables span BOTH columns (LaTeX `table*`) vs sit in one
# (`table`). Flip a value and re-run this script to try the other layout.
#
# NOTE: a `table*` float can only be placed at the TOP of a page -- [h]/[b]
# are silently ignored by LaTeX. That is why WIDE tables use [t] below.
WIDE = {
    "dcor_humanhuman":     True,   # 5 cols incl. a bracketed CI string
    "dcor_contrasts":      True,   # 4 cols, CI + p-value
    "gstudy_cip":          False,   # 7 cols -- needs the full width
    "gstudy_menagerie":    False,   # 7 cols -- needs the full width
}
# Set TIGHT to shave column padding on the widest tables (default 6pt).
TIGHT_COLSEP_PT = 4


def table_env(name):
    """Return (begin, end) for a table float, honouring the WIDE switch."""
    star = "*" if WIDE.get(name, False) else ""
    return (f"\\begin{{table{star}}}[t]", f"\\end{{table{star}}}")

CIP_ITEMS = ["input", "conditional_logic", "printing", "syntax_errors", "composite"]
MEN_ITEMS = ["correctness", "code_elegance", "readability", "documentation", "composite"]

PRETTY = {
    "input": "Input", "conditional_logic": "Conditional logic",
    "printing": "Printing", "syntax_errors": "Syntax errors",
    "correctness": "Correctness", "code_elegance": "Code elegance",
    "readability": "Readability", "documentation": "Documentation",
    "composite": "Composite",
}


def esc(s):
    """Escape LaTeX specials that appear in our labels."""
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------- figure ----------
def figure_humanhuman(cells, out_path):
    """Grouped bars: the three same-rubric-item cells, per dataset.

    Every bar is direct-labelled with its value, so the comparison is readable
    from the text alone and identity never rests on colour. The y axis is capped
    at the main figure's 0.90 so this panel can be read against Figure 3
    directly -- Menagerie's bars are genuinely that small.
    """
    sns.set_theme(style="whitegrid")
    palette = dict(zip(CELLS, sns.color_palette("tab10", 3)))
    datasets = ["CIP", "Menagerie"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    x = np.arange(len(CELLS))

    for ax, ds in zip(axes, datasets):
        sub = cells[cells.dataset == ds].set_index("cell").loc[CELLS]
        h = sub["dcor_sq"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        assert np.all(lo <= h + 1e-9) and np.all(h <= hi + 1e-9), \
            f"point estimate outside its CI for {ds}"

        ax.bar(x, h, width=0.62, color=[palette[c] for c in CELLS])
        ax.errorbar(x, h, yerr=np.vstack([h - lo, hi - h]), fmt="none",
                    ecolor="black", capsize=5, lw=1.3)
        for xi, hv, hiv in zip(x, h, hi):
            ax.annotate(f"{hv:.3f}", xy=(xi, hiv), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CELLS, rotation=15, ha="right")
        ax.set_ylim(0, YMAX)
        ax.set_yticks(np.arange(0, YMAX + 1e-9, 0.1))
        ax.set_title(ds, fontsize=12, fontweight="bold")
        n = sub["paired_n_median"].to_numpy()
        k = sub["n_pairs"].to_numpy()
        ax.set_xlabel("\n".join([
            "  ".join(f"{c.split('-')[0][0]}{c.split('-')[1][0]}: k={ki}, n={ni}"
                      for c, ki, ni in zip(CELLS, k, n))]),
            fontsize=8, color="0.35")

    axes[0].set_ylabel("Mean squared bias-corrected distance correlation")
    # plain % -- matplotlib is not in LaTeX mode here, a backslash would render literally
    fig.suptitle("Same rubric item: grader-pair agreement by pair type "
                 "(95% bootstrap CIs)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- tables ----------
def table_humanhuman(cells, contrasts, path):
    beg, end = table_env("dcor_humanhuman")
    lines = [
        beg, r"\centering", r"\small",
        f"\\setlength{{\\tabcolsep}}{{{TIGHT_COLSEP_PT}pt}}",
        r"\caption{Squared bias-corrected distance correlation between grader "
        r"pairs on the same rubric item, with 95\% bootstrap confidence "
        r"intervals. $k$ is the number of grader-pair $\times$ rubric-item "
        r"comparisons pooled; $n$ is the median number of paired submissions "
        r"per comparison.}",
        r"\label{tab:dcor-humanhuman}",
        r"\begin{tabular}{llrrl}", r"\toprule",
        r"Dataset & Pair type & $k$ & $n$ & $d\mathrm{Cor}^2$ [95\% CI] \\",
        r"\midrule",
    ]
    for ds in ("CIP", "Menagerie"):
        sub = cells[cells.dataset == ds].set_index("cell").loc[CELLS]
        for i, (cell, r) in enumerate(sub.iterrows()):
            lines.append(
                f"{ds if i == 0 else ''} & {cell} & {int(r.n_pairs)} & "
                f"{int(r.paired_n_median)} & "
                f"{r.dcor_sq:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] \\\\")
        lines.append(r"\midrule" if ds == "CIP" else r"\bottomrule")
    lines += [r"\end{tabular}", end, ""]

    beg2, end2 = table_env("dcor_contrasts")
    lines += [
        beg2, r"\centering", r"\small",
        r"\caption{Contrasts between pair types, computed on the same bootstrap "
        r"replicates so the dependence between cells is accounted for.}",
        r"\label{tab:dcor-humanhuman-contrasts}",
        r"\begin{tabular}{llrl}", r"\toprule",
        r"Dataset & Contrast & $\Delta$ & 95\% CI \quad ($p$) \\", r"\midrule",
    ]
    for ds in ("CIP", "Menagerie"):
        sub = contrasts[contrasts.dataset == ds]
        for i, (_, r) in enumerate(sub.iterrows()):
            star = r"$^{*}$" if r.p_value < 0.05 else ""
            lines.append(
                f"{ds if i == 0 else ''} & {r.contrast} & {r['diff']:+.3f}{star} & "
                f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] \\quad ({r.p_value:.4f}) \\\\")
        lines.append(r"\midrule" if ds == "CIP" else r"\bottomrule")
    lines += [r"\end{tabular}", end2, ""]
    write(path, lines)


def table_gstudy(vc, items, dataset, path, has_group):
    """Per-system variance components, human vs LLM, side by side per item."""
    cap = (f"Per-system generalizability study for {dataset}. "
           r"$\sigma^2_{\text{student}}$ is variance from genuine differences "
           r"between students; $\sigma^2_{\text{disagree}}$ is variance from "
           r"graders ranking students differently from one another. $\rho^2$ "
           r"is the share of a \emph{single} grader's score variance that "
           r"reflects real student differences, so $1$ is a perfectly reliable "
           r"grader and $0$ is noise. Brackets give 95\% bootstrap CIs.")
    beg, end = table_env(f"gstudy_{dataset.lower()}")
    lines = [
        beg, r"\centering", r"\small",
        f"\\setlength{{\\tabcolsep}}{{{TIGHT_COLSEP_PT}pt}}",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:gstudy-{dataset.lower()}}}",
        r"\begin{tabular}{llrrl}", r"\toprule",
        r"Rubric item & Graders & $\sigma^2_{\text{student}}$ & "
        r"$\sigma^2_{\text{disagree}}$ & $\rho^2$ [95\% CI] \\", r"\midrule",
    ]
    for item in items:
        for j, sysname in enumerate(("human", "llm")):
            r = vc[(vc.key == item) & (vc.system == sysname)]
            if r.empty:
                continue
            r = r.iloc[0]
            label = PRETTY.get(item, item) if j == 0 else ""
            lines.append(
                f"{esc(label)} & {'Human' if sysname == 'human' else 'LLM'} & "
                f"{r.s2_student:.2f} & {r.s2_sr:.2f} & "
                f"{r.rho2:.3f} [{r.rho2_lo:.3f}, {r.rho2_hi:.3f}] \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", end, ""]
    write(path, lines)


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.relpath(path, ROOT))


def main():
    os.makedirs(TABLES, exist_ok=True)
    r = lambda n: pd.read_csv(os.path.join(RESULTS, n))

    cells = r("dcor_humanhuman_cells.csv")
    contrasts = r("dcor_humanhuman_contrasts.csv")
    figure_humanhuman(cells, os.path.join(RESULTS, "appendix_c_dcor_humanhuman.png"))
    print("wrote results/appendix_c_dcor_humanhuman.png")
    table_humanhuman(cells, contrasts, os.path.join(TABLES, "dcor_humanhuman.tex"))

    table_gstudy(r("gstudy_cip_variance_components.csv"), CIP_ITEMS, "CIP",
                 os.path.join(TABLES, "gstudy_cip.tex"), has_group=False)
    table_gstudy(r("gstudy_menagerie_variance_components.csv"), MEN_ITEMS,
                 "Menagerie", os.path.join(TABLES, "gstudy_menagerie.tex"),
                 has_group=True)


if __name__ == "__main__":
    main()
