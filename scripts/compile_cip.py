"""
Compile all CIP grading runs into one wide CSV for dCor / G-study analysis.

Merges, on student_id:
  - 5 human graders          (from gpt41_human_compiled.csv)
  - GPT-4.1                  (see GPT-4.1 SOURCE note below)
  - O3, Gemini 2.5 Flash, Sonnet 4.6   (from multi_model_o3_gemini_sonnet.csv)
  - DeepSeek-v4-flash        (from multi_model_deepseek.csv)

Output: runs/cip_compiled.csv — one row per student, columns student_id +
        <rater>_<item> for every rater x rubric item.
        Rubric items use the canonical rubric.json IDs:
        input, conditional_logic, printing, syntax_errors.
        To get the long/tidy form for a G-study:  df.melt(id_vars='student_id')

GPT-4.1 SOURCE: the GPT-4.1 grades (`initial_*` in gpt41_human_compiled.csv) are
the single-pass output of the 1.0 Grader. Verified 2026-05-21: that prompt is
substantively identical to grade_multi_model.py's (same system prompt, rubric
formatter, code block, instructions, JSON schema, temp 0.3, json mode) — so
`initial_*` is directly comparable to the other models. (If a clean re-run
runs/multi_model_gpt41.csv exists it is used instead, but it is not needed.)

Usage:  python scripts/compile_cip.py
"""

import os
import sys

import pandas as pd

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

CANON_ITEMS = ["input", "conditional_logic", "printing", "syntax_errors"]

# old short item names (gpt41_human_compiled.csv) -> canonical rubric.json ids
OLD_TO_CANON = {
    "input": "input",
    "logic": "conditional_logic",
    "syntax": "syntax_errors",
    "print": "printing",
}

# Human graders are PSEUDONYMOUS everywhere downstream. Their real identifiers
# exist only in the private source file (runs/gpt41_human_compiled.csv, which is
# gitignored); we discover them from its column names at load time and map them
# to stable human_1..human_N on the way out. Nothing in this repository -- code,
# compiled data, or results -- should ever carry a real grader name.
N_HUMANS = 5
HUMANS = [f"human_{i}" for i in range(1, N_HUMANS + 1)]


def discover_human_prefixes(df):
    """Rater prefixes in the 1.0 compiled file, excluding the LLM stages.

    Returns them sorted, so the real-name -> pseudonym assignment is stable
    across runs. Raises if the count is unexpected rather than silently
    mis-assigning graders.
    """
    reserved = {"initial", "final"}
    prefixes = sorted(
        {c.rsplit("_", 1)[0] for c in df.columns if "_" in c and c != "student_id"}
        - reserved
    )
    if len(prefixes) != N_HUMANS:
        raise ValueError(
            f"expected {N_HUMANS} human grader prefixes in the source file, "
            f"found {len(prefixes)}: {prefixes}"
        )
    return prefixes


def extract_rater(df, src_prefix, out_rater, item_name_map):
    """Pull one rater's 4 item columns out of df, renamed to <out_rater>_<canon>.

    item_name_map: canonical_item -> column suffix used in df.
    """
    cols = {}
    for canon in CANON_ITEMS:
        src_col = f"{src_prefix}{item_name_map[canon]}"
        if src_col not in df.columns:
            raise KeyError(f"expected column {src_col!r} not found")
        cols[f"{out_rater}_{canon}"] = df[src_col]
    out = pd.DataFrame(cols)
    out["student_id"] = df["student_id"].values
    return out


def main():
    canon_map = {c: c for c in CANON_ITEMS}                 # canonical -> itself
    old_map = {c: old for old, c in OLD_TO_CANON.items()}   # canonical -> old name

    # --- humans + (fallback) GPT-4.1, from the 1.0 compiled file ---
    human_file = os.path.join(RUNS, "gpt41_human_compiled.csv")
    hc = pd.read_csv(human_file)
    print(f"loaded {human_file}: {len(hc)} rows")

    merged = hc[["student_id"]].copy()
    source_ids = discover_human_prefixes(hc)
    print(f"  mapping {len(source_ids)} human graders to {HUMANS[0]}..{HUMANS[-1]}")
    for src, h in zip(source_ids, HUMANS):
        merged = merged.merge(extract_rater(hc, f"{src}_", h, old_map), on="student_id")

    # --- GPT-4.1: prefer a clean re-run, else fall back to initial_* ---
    gpt41_clean = os.path.join(RUNS, "multi_model_gpt41.csv")
    if os.path.exists(gpt41_clean):
        gc = pd.read_csv(gpt41_clean)
        merged = merged.merge(
            extract_rater(gc, "gpt-4.1_", "gpt-4.1", canon_map), on="student_id"
        )
        print(f"GPT-4.1: using clean re-run {gpt41_clean}")
    else:
        merged = merged.merge(
            extract_rater(hc, "initial_", "gpt-4.1", old_map), on="student_id"
        )
        print(
            "GPT-4.1: using `initial_*` (single-pass grade) from gpt41_human_compiled.csv.\n"
            "         Verified 2026-05-21 — the 1.0 Grader prompt is substantively identical"
            " to grade_multi_model.py's, so this is directly comparable to the other models."
        )

    # --- O3 / Gemini / Sonnet ---
    msf = os.path.join(RUNS, "multi_model_o3_gemini_sonnet.csv")
    ms = pd.read_csv(msf)
    print(f"loaded {msf}: {len(ms)} rows")
    for label in ["o3", "gemini-2.5-flash", "sonnet-4.6"]:
        merged = merged.merge(
            extract_rater(ms, f"{label}_", label, canon_map), on="student_id"
        )

    # --- DeepSeek ---
    dsf = os.path.join(RUNS, "multi_model_deepseek.csv")
    ds = pd.read_csv(dsf)
    print(f"loaded {dsf}: {len(ds)} rows")
    merged = merged.merge(
        extract_rater(ds, "deepseek_", "deepseek", canon_map), on="student_id"
    )

    # --- order columns: student_id, then rater-blocks ---
    raters = HUMANS + ["gpt-4.1", "o3", "gemini-2.5-flash", "sonnet-4.6", "deepseek"]
    ordered = ["student_id"] + [f"{r}_{it}" for r in raters for it in CANON_ITEMS]
    merged = merged[ordered]

    out = os.path.join(RUNS, "cip_compiled.csv")
    merged.to_csv(out, index=False)

    n_missing = merged.drop(columns="student_id").isna().sum().sum()
    print(
        f"\nwrote {out}\n"
        f"  {len(merged)} students x {len(raters)} raters x {len(CANON_ITEMS)} items\n"
        f"  {len(merged.columns) - 1} score columns, {n_missing} missing cells"
    )
    if n_missing:
        bad = merged.drop(columns="student_id").isna().sum()
        print("  missing by column:", bad[bad > 0].to_dict())


if __name__ == "__main__":
    sys.exit(main())
