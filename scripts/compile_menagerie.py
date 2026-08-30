"""
Compile Menagerie grades into one tidy LONG table for dCor / G-study analysis.

Merges:
  - human grades : <github-archive>/data/grades.csv   (28 graders, 4 skills)
  - LLM grades   : runs/menagerie_multi_model.csv      (grade_multi_model.py output)

Output: runs/menagerie_compiled.csv -- one row per (submission, rater, skill):

    submission, skill, rater, rater_type, group, round, score

  * score is a 1-14 ordinal shared by humans and LLMs (F=1 ... A++=14):
        human : letter grade           -> ordinal
        LLM   : optionId (0=A++..13=F) -> 14 - optionId
  * group is the grading group (1-7) that graded the submission. Each submission
    belongs to exactly one group; the group is attached to the LLM rows too.
  * round is the human grading round (1/2); empty for LLM rows.

Long format is used (not CIP's wide format) because the human design is NESTED
-- each submission is graded by only one group of 4 -- so a wide rater x item
table would be ~90% empty. For the LLM-only crossed G-study, pivot the LLM rows
back to wide:  llm.pivot(index='submission', columns='rater', values='score').

Usage:
    python scripts/compile_menagerie.py
    # smoke test on a partial grading run:
    python scripts/compile_menagerie.py --llm runs/menagerie_smoke.csv \
        --output runs/menagerie_smoke_compiled.csv
"""

import argparse
import collections
import csv
import os
import sys

csv.field_size_limit(10 ** 8)   # the LLM csv carries a large `code` column

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUBRIC_IDS = ["correctness", "code_elegance", "readability", "documentation"]

# letter grade -> 1-14 ordinal (F worst = 1, A++ best = 14)
GRADE_ORDINAL = {
    "A++": 14, "A+": 13, "A": 12, "A-": 11,
    "B+": 10, "B": 9, "B-": 8,
    "C+": 7, "C": 6, "C-": 5,
    "D+": 4, "D": 3, "D-": 2, "F": 1,
}

# grades.csv skill label (lower-cased) -> canonical rubric id
SKILL_CANON = {
    "correctness": "correctness",
    "code elegance": "code_elegance",
    "readability": "readability",
    "documentation": "documentation",
}

OUT_FIELDS = ["submission", "skill", "rater", "rater_type", "group", "round", "score"]


def submission_number(student_id):
    """`18~19_Submission_36` -> 36."""
    return int(str(student_id).rsplit("_", 1)[-1])


def grader_group(pid):
    """participant_id 1-28 -> group 1-7 ({1-4}->1, {5-8}->2, ...)."""
    return (pid - 1) // 4 + 1


def load_human(grades_path):
    """Returns (rows, submission->group map, n_dropped_missing)."""
    rows, sub_group, dropped = [], {}, 0
    with open(grades_path) as f:
        for r in csv.DictReader(f):
            grade = (r.get("grade") or "").strip()
            if grade not in GRADE_ORDINAL:
                dropped += 1
                continue
            sub = int(float(r["assignment_number"]))
            pid = int(float(r["participant_id"]))
            skill = SKILL_CANON.get((r.get("skill") or "").strip().lower())
            if skill is None:
                raise ValueError(f"unknown skill label: {r.get('skill')!r}")
            grp = grader_group(pid)
            rnd = (r.get("batch") or "").strip()
            rnd = str(int(float(rnd))) if rnd else ""
            if sub in sub_group and sub_group[sub] != grp:
                raise ValueError(
                    f"submission {sub} graded by >1 group "
                    f"({sub_group[sub]} and {grp}) -- design assumption violated")
            sub_group[sub] = grp
            rows.append({
                "submission": sub, "skill": skill,
                "rater": f"human_{pid}", "rater_type": "human",
                "group": grp, "round": rnd, "score": GRADE_ORDINAL[grade],
            })
    return rows, sub_group, dropped


def detect_models(fieldnames):
    """Model labels that have all four `<model>_<rubric_id>` score columns."""
    candidates = set()
    for rid in RUBRIC_IDS:
        suffix = "_" + rid
        for c in fieldnames:
            if c.endswith(suffix):
                candidates.add(c[: -len(suffix)])
    return sorted(
        m for m in candidates
        if all(f"{m}_{rid}" in fieldnames for rid in RUBRIC_IDS)
    )


def load_llm(llm_path, sub_group):
    """Returns (rows, models, error_counts, bad_score_counts, n_submissions)."""
    rows = []
    errors = collections.Counter()      # model -> # submissions that errored
    bad = collections.Counter()         # model -> # unparseable scores (non-error)
    n_subs = 0
    with open(llm_path) as f:
        reader = csv.DictReader(f)
        models = detect_models(reader.fieldnames or [])
        if not models:
            raise ValueError(f"no `<model>_<rubric_id>` columns in {llm_path}")
        for r in reader:
            n_subs += 1
            sub = submission_number(r["student_id"])
            grp = sub_group.get(sub, "")
            for m in models:
                errored = bool((r.get(f"{m}_error") or "").strip())
                if errored:
                    errors[m] += 1
                for rid in RUBRIC_IDS:
                    raw = (r.get(f"{m}_{rid}") or "").strip()
                    try:
                        opt = int(float(raw))
                    except (TypeError, ValueError):
                        opt = None
                    if opt is None or not (0 <= opt <= 13):
                        if not errored:
                            bad[m] += 1
                        score = ""
                    else:
                        score = 14 - opt
                    rows.append({
                        "submission": sub, "skill": rid,
                        "rater": f"llm_{m}", "rater_type": "llm",
                        "group": grp, "round": "", "score": score,
                    })
    return rows, models, errors, bad, n_subs


def mean_by(rows, rater_type):
    """Mean ordinal score per skill for one rater type (ignores missing)."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["rater_type"] == rater_type and r["score"] != "":
            acc[r["skill"]].append(r["score"])
    return {s: (sum(v) / len(v) if v else float("nan")) for s, v in acc.items()}


def smoke_comparison(human_rows, llm_rows, models):
    """Per-submission human-vs-LLM table -- only for small (smoke) runs."""
    subs = sorted({r["submission"] for r in llm_rows})
    print("\nSMOKE COMPARISON  (ordinal 1-14: F=1 ... A++=14)")
    for sub in subs:
        grp = next((r["group"] for r in llm_rows if r["submission"] == sub), "?")
        print(f"\n  submission {sub}  (group {grp})")
        print(f"    {'skill':14s} {'human grades':22s} "
              + " ".join(f"{m:>16s}" for m in models))
        for rid in RUBRIC_IDS:
            hs = sorted(r["score"] for r in human_rows
                        if r["submission"] == sub and r["skill"] == rid)
            hcell = f"{hs} m={sum(hs)/len(hs):.1f}" if hs else "(no human grades)"
            lcells = []
            for m in models:
                v = [r["score"] for r in llm_rows
                     if r["submission"] == sub and r["skill"] == rid
                     and r["rater"] == f"llm_{m}"]
                lcells.append(f"{v[0]!s:>16s}" if v else f"{'-':>16s}")
            print(f"    {rid:14s} {hcell:22s} " + " ".join(lcells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", nargs="+",
                    default=[os.path.join(ROOT, "runs", "menagerie_multi_model.csv")],
                    help="One or more LLM grading output CSVs (grade_multi_model.py "
                         "format). Pass a shell glob, e.g. "
                         "`runs/menagerie_full_*.csv`, to merge per-model files.")
    ap.add_argument("--grades", default=os.path.join(ROOT, "..", "github-archive",
                                                     "data", "grades.csv"),
                    help="Menagerie human grades.csv.")
    ap.add_argument("--output", default=os.path.join(ROOT, "runs", "menagerie_compiled.csv"))
    args = ap.parse_args()

    llm_files = [p for p in args.llm if os.path.exists(p)]
    missing = [p for p in args.llm if not os.path.exists(p)]
    for p in missing:
        print(f"  (skipped, not found) {p}", file=sys.stderr)
    if not llm_files:
        sys.exit("no LLM grading files found. Run grade_multi_model.py first.")

    human_rows, sub_group, dropped = load_human(args.grades)
    print(f"human grades : {len(human_rows)} rows from {len(sub_group)} submissions, "
          f"7 groups, {dropped} missing/NAN dropped")

    # Merge LLM rows across one-or-more per-model files; reject duplicate models.
    llm_rows, all_models, subs_seen = [], [], set()
    errors, bad = collections.Counter(), collections.Counter()
    for path in llm_files:
        rows_p, models_p, err_p, bad_p, n_p = load_llm(path, sub_group)
        dup = set(all_models) & set(models_p)
        if dup:
            sys.exit(f"model {sorted(dup)} appears in more than one file "
                     f"(last seen in {path})")
        all_models += models_p
        llm_rows += rows_p
        errors.update(err_p)
        bad.update(bad_p)
        subs_seen.update(int(r["submission"]) for r in rows_p)
        print(f"  loaded {path}: {n_p} subs, models {models_p}")
    models = sorted(all_models)
    n_subs = len(subs_seen)
    print(f"LLM grades   : {len(llm_rows)} rows | {len(models)} models {models} "
          f"| {n_subs} submissions across {len(llm_files)} file(s)")
    if errors:
        print(f"  model errors (failed submissions): {dict(errors)}")
    if bad:
        print(f"  unparseable scores (non-error):    {dict(bad)}")
    no_group = sorted({r["submission"] for r in llm_rows if r["group"] == ""})
    if no_group:
        print(f"  WARNING: {len(no_group)} LLM submissions absent from grades.csv "
              f"(no group): {no_group}")

    all_rows = human_rows + llm_rows
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {args.output}  ({len(all_rows)} rows)")

    # ---- sanity summary ----
    hmean, lmean = mean_by(all_rows, "human"), mean_by(all_rows, "llm")
    print("\nmean ordinal score per skill  (higher = better grade)")
    print(f"  {'skill':14s} {'human':>8s} {'llm':>8s}")
    for rid in RUBRIC_IDS:
        h = hmean.get(rid, float("nan"))
        l = lmean.get(rid, float("nan"))
        print(f"  {rid:14s} {h:8.2f} {l:8.2f}")

    scores = [r["score"] for r in all_rows if r["score"] != ""]
    if scores and (min(scores) < 1 or max(scores) > 14):
        print(f"  WARNING: scores out of 1-14 range: "
              f"min={min(scores)} max={max(scores)}")

    if n_subs <= 12:
        smoke_comparison(human_rows, llm_rows, models)


if __name__ == "__main__":
    main()
