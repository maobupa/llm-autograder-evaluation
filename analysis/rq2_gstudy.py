"""
analysis/rq2_gstudy.py — RQ2 Generalizability (G) study for the CIP dataset.

Two-facet  student x rater  crossed G-study (CIP is balanced + fully crossed),
run per rubric item and on the 4-item composite, for the human-only and the
LLM-only grading systems.

Per (system, item) it estimates:
  - variance components sigma^2(student), sigma^2(rater), sigma^2(student x rater)
  - generalizability coefficients rho^2 (relative) and Phi (absolute),
    reported for a SINGLE grader (n_r = 1)
  - 95% CIs by resampling the 100 students (transcript-level bootstrap)
Plus:
  - difference test, human - LLM, for sigma^2(rater) and sigma^2(s x r)
    (RQ2 hypothesis: both smaller for LLMs)
  - disattenuated correlation between the human-system and LLM-system means

Variance components use the standard balanced-crossed ANOVA estimators
(Brennan, Generalizability Theory). Cross-checked against GeneralizIT:
human/input -> sigma^2(person)=0.4237, sigma^2(rater)=-0.000, s x r=0.0100.

Run:  python analysis/rq2_gstudy.py
"""

import os
from collections import defaultdict

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
KEYS = TASKS + ["composite"]
N_BOOT = 2000
SEED = 0


# ---------- G-theory estimators ----------
def variance_components(X):
    """Balanced student x rater crossed design, one observation per cell.
    X: (n_students, n_raters). Returns (s2_student, s2_rater, s2_sr).
    Negative point estimates of s2_student / s2_rater are floored at 0."""
    p, r = X.shape
    grand = X.mean()
    row = X.mean(axis=1)
    col = X.mean(axis=0)
    ms_p = (r * np.sum((row - grand) ** 2)) / (p - 1)
    ms_r = (p * np.sum((col - grand) ** 2)) / (r - 1)
    resid = X - row[:, None] - col[None, :] + grand
    ms_pr = np.sum(resid ** 2) / ((p - 1) * (r - 1))
    s2_sr = ms_pr
    s2_p = max((ms_p - ms_pr) / r, 0.0)
    s2_r = max((ms_r - ms_pr) / p, 0.0)
    return s2_p, s2_r, s2_sr


def g_coefficients(s2_p, s2_r, s2_sr, n_r=1):
    """Relative (rho^2) and absolute (Phi) generalizability coefficients for a
    decision based on the average of n_r raters (n_r=1 -> single grader)."""
    rho2 = s2_p / (s2_p + s2_sr / n_r) if (s2_p + s2_sr / n_r) > 0 else 0.0
    abs_err = (s2_r + s2_sr) / n_r
    phi = s2_p / (s2_p + abs_err) if (s2_p + abs_err) > 0 else 0.0
    return rho2, phi


# ---------- data ----------
def build_matrices(wide):
    """{(system, key): (100 x 5) score matrix} for the 4 items + composite."""
    M = {}
    for system, raters in (("human", HUMANS), ("llm", LLMS)):
        per_item = {t: wide[[f"{r}_{t}" for r in raters]].to_numpy(float) for t in TASKS}
        M[(system, "composite")] = sum(per_item[t] for t in TASKS)
        for t in TASKS:
            M[(system, t)] = per_item[t]
    return M


def estimate(M, idx=None):
    """Variance components + coefficients for every (system, key)."""
    out = {}
    for (system, key), X in M.items():
        if idx is not None:
            X = X[idx]
        s2p, s2r, s2sr = variance_components(X)
        rho2, phi = g_coefficients(s2p, s2r, s2sr, n_r=1)
        out[(system, key)] = dict(
            s2_student=s2p, s2_rater=s2r, s2_sr=s2sr,
            total=s2p + s2r + s2sr, rho2=rho2, phi=phi,
        )
    return out


def disattenuated(M, idx=None):
    """corr(human-system mean, LLM-system mean) corrected for each system's
    unreliability: r_obs / sqrt(rel_human * rel_llm), reliabilities = rho^2 of
    the full 5-rater average."""
    out = {}
    for key in KEYS:
        Xh, Xl = M[("human", key)], M[("llm", key)]
        if idx is not None:
            Xh, Xl = Xh[idx], Xl[idx]
        r_obs = np.corrcoef(Xh.mean(1), Xl.mean(1))[0, 1]
        rel_h = g_coefficients(*variance_components(Xh), n_r=Xh.shape[1])[0]
        rel_l = g_coefficients(*variance_components(Xl), n_r=Xl.shape[1])[0]
        denom = np.sqrt(rel_h * rel_l)
        r_dis = r_obs / denom if denom > 0 else np.nan
        out[key] = dict(r_obs=r_obs, rel_human=rel_h, rel_llm=rel_l,
                        r_disattenuated=min(r_dis, 1.0) if np.isfinite(r_dis) else np.nan)
    return out


# ---------- bootstrap ----------
def bootstrap(M, n_boot=N_BOOT, seed=SEED):
    """Resample the 100 students WITH replacement; apply the SAME resample to
    every matrix so human/LLM stay paired; recompute everything."""
    rng = np.random.default_rng(seed)
    n = next(iter(M.values())).shape[0]
    stat_draws = defaultdict(lambda: defaultdict(list))   # (system,key) -> stat -> [B]
    diff_draws = defaultdict(lambda: defaultdict(list))   # key -> stat -> [B]
    disatt_draws = defaultdict(list)                      # key -> [B]
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        est = estimate(M, idx)
        for k, d in est.items():
            for stat, val in d.items():
                stat_draws[k][stat].append(val)
        for key in KEYS:
            h, l = est[("human", key)], est[("llm", key)]
            diff_draws[key]["s2_rater"].append(h["s2_rater"] - l["s2_rater"])
            diff_draws[key]["s2_sr"].append(h["s2_sr"] - l["s2_sr"])
            diff_draws[key]["rho2"].append(l["rho2"] - h["rho2"])  # LLM - human
        for key, d in disattenuated(M, idx).items():
            disatt_draws[key].append(d["r_disattenuated"])
    return stat_draws, diff_draws, disatt_draws


def ci(draws, level=95):
    lo = (100 - level) / 2
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, 100 - lo))


def boot_p(diff_draws, n_boot):
    d = np.asarray(diff_draws, float)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return max(p, 1.0 / n_boot)


# ---------- figure ----------
def plot_gstudy(point, stat_draws, out_path):
    sns.set_theme(style="whitegrid")
    colors = {"human": "#4C72B0", "llm": "#DD8452"}
    short = {"input": "input", "conditional_logic": "cond_logic",
             "printing": "printing", "syntax_errors": "syntax", "composite": "COMPOSITE"}
    panels = [
        ("s2_student", "σ²(student) — share of total variance", True),
        ("s2_rater", "σ²(rater) — share of total variance", True),
        ("s2_sr", "σ²(student×rater) — share of total variance", True),
        ("rho2", "Single-grader generalizability ρ²", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(KEYS))
    w = 0.38

    for ax, (stat, title, as_share) in zip(axes.flat, panels):
        for s, (system, off) in enumerate((("human", -w / 2), ("llm", w / 2))):
            heights, los, his = [], [], []
            for key in KEYS:
                draws = np.array(stat_draws[(system, key)][stat], float)
                pt = point[(system, key)][stat]
                if as_share:
                    tot = np.array(stat_draws[(system, key)]["total"], float)
                    draws = draws / tot
                    pt = pt / point[(system, key)]["total"]
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

    fig.suptitle("CIP G-study — Human vs LLM grading systems (95% bootstrap CIs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    wide = pd.read_csv(COMPILED)
    M = build_matrices(wide)

    # validation against GeneralizIT reference (human/input)
    ref = variance_components(M[("human", "input")])
    print("validation human/input: numpy =",
          tuple(round(v, 4) for v in ref),
          " | GeneralizIT = (0.4237, 0.0, 0.0100)")

    point = estimate(M)
    stat_draws, diff_draws, disatt_draws = bootstrap(M)
    print(f"bootstrap: {N_BOOT} replicates done\n")

    # ---- variance-component table ----
    vc_rows = []
    for (system, key), d in point.items():
        row = {"system": system, "key": key}
        for stat in ("s2_student", "s2_rater", "s2_sr", "rho2", "phi"):
            lo, hi = ci(stat_draws[(system, key)][stat])
            row[stat] = d[stat]
            row[f"{stat}_lo"], row[f"{stat}_hi"] = lo, hi
        vc_rows.append(row)
    vc = pd.DataFrame(vc_rows).sort_values(["key", "system"])
    vc.to_csv(os.path.join(RESULTS, "gstudy_cip_variance_components.csv"), index=False)

    # ---- difference test ----
    diff_rows = []
    labels = {"s2_rater": "σ²(rater)  human-LLM", "s2_sr": "σ²(s×r)  human-LLM",
              "rho2": "ρ²  LLM-human"}
    for key in KEYS:
        for stat in ("s2_rater", "s2_sr", "rho2"):
            d = diff_draws[key][stat]
            obs = (point[("human", key)][stat] - point[("llm", key)][stat]
                   if stat != "rho2"
                   else point[("llm", key)][stat] - point[("human", key)][stat])
            lo, hi = ci(d)
            diff_rows.append({"key": key, "contrast": labels[stat], "diff": obs,
                              "ci_lo": lo, "ci_hi": hi, "p_value": boot_p(d, N_BOOT)})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(os.path.join(RESULTS, "gstudy_cip_difference_test.csv"), index=False)

    # ---- disattenuated correlation ----
    dis_point = disattenuated(M)
    dis_rows = []
    for key in KEYS:
        lo, hi = ci(disatt_draws[key])
        dis_rows.append({"key": key, **dis_point[key], "r_disatt_lo": lo, "r_disatt_hi": hi})
    dis = pd.DataFrame(dis_rows)
    dis.to_csv(os.path.join(RESULTS, "gstudy_cip_disattenuated.csv"), index=False)

    plot_gstudy(point, stat_draws, os.path.join(RESULTS, "gstudy_cip_figure.png"))

    # ---- report ----
    print("VARIANCE COMPONENTS (point estimate)")
    print(f"  {'system/item':28s} {'σ²(stu)':>9} {'σ²(rater)':>10} {'σ²(s×r)':>9} {'ρ²(1)':>8}")
    for key in KEYS:
        for system in ("human", "llm"):
            d = point[(system, key)]
            print(f"  {system+'/'+key:28s} {d['s2_student']:9.3f} {d['s2_rater']:10.3f} "
                  f"{d['s2_sr']:9.3f} {d['rho2']:8.3f}")

    print("\nDIFFERENCE TEST  (positive supports the RQ2 hypothesis)")
    for _, r in diff.iterrows():
        sig = "***" if r.p_value < 0.05 else "   "
        print(f"  {r.key:14s} {r.contrast:22s} {r['diff']:+.3f}  "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r.p_value:.4f} {sig}")

    print("\nDISATTENUATED CORRELATION  (human-system vs LLM-system)")
    for _, r in dis.iterrows():
        print(f"  {r.key:14s} r_obs={r.r_obs:.3f}  rel_h={r.rel_human:.3f}  "
              f"rel_l={r.rel_llm:.3f}  ->  r_disatt={r.r_disattenuated:.3f} "
              f"[{r.r_disatt_lo:.3f}, {r.r_disatt_hi:.3f}]")

    print("\nsaved: gstudy_cip_variance_components.csv, gstudy_cip_difference_test.csv,")
    print("       gstudy_cip_disattenuated.csv, gstudy_cip_figure.png")


if __name__ == "__main__":
    main()
