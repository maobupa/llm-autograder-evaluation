"""
Build the Menagerie LLM-grading input CSV.

Unzips each graded submission project from <batches>/1..14, concatenates its
.java files into one code blob, and writes a CSV with columns `student_id, code`
-- the format scripts/grade_multi_model.py expects.

Batches 1..14 are the 7 grading groups x 2 rounds; batch 0 is the pilot and is
skipped. Submission 105 appears in two batches (graded in both rounds); it is
de-duplicated to a single row.

Usage (run from the autograder_analysis/ project root):
    python scripts/build_menagerie_input.py
"""

import argparse
import csv
import glob
import os
import zipfile


def submission_number(student_id):
    """`18~19_Submission_36` -> 36 (the globally-unique assignment number)."""
    return int(student_id.rsplit("_", 1)[-1])


def load_java_blob(zip_path):
    """Concatenate every .java file in a submission zip into one string.

    Returns (blob, n_java_files).
    """
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith(".java") and "__MACOSX" not in n
        )
        for n in names:
            text = zf.read(n).decode("utf-8", errors="replace")
            parts.append(f"// ===== File: {os.path.basename(n)} =====\n{text}")
    return "\n\n".join(parts), len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="../github-archive/data/batches",
                    help="Path to the Menagerie data/batches directory.")
    ap.add_argument("--grades", default="../github-archive/data/grades.csv",
                    help="grades.csv -- used only to verify the submission set.")
    ap.add_argument("--output", default="data/menagerie/submissions.csv")
    args = ap.parse_args()

    # Collect unique submission zips across the 14 graded batches.
    zips = {}
    for b in range(1, 15):
        for zp in glob.glob(os.path.join(args.batches, str(b), "*.zip")):
            sid = os.path.basename(zp)[:-4]
            zips.setdefault(sid, zp)   # first occurrence wins (handles dups)
    print(f"Found {len(zips)} unique submission zips in batches 1-14.")

    rows, empty = [], []
    for sid in sorted(zips, key=submission_number):
        code, n_java = load_java_blob(zips[sid])
        if n_java == 0:
            empty.append(sid)
        rows.append({"student_id": sid, "code": code})

    # Restrict to the submissions that have human grades, so the LLM-grading
    # set matches the analysis set exactly. A handful of zips in the batch
    # folders are never graded in grades.csv and would be dropped at join time.
    if os.path.exists(args.grades):
        with open(args.grades) as f:
            graded = {int(float(r["assignment_number"]))
                      for r in csv.DictReader(f)
                      if r.get("assignment_number")}
        ours = {submission_number(r["student_id"]) for r in rows}
        missing, extra = graded - ours, ours - graded
        print(f"grades.csv graded submissions: {len(graded)} | batch zips: "
              f"{len(ours)} | missing from batches: {len(missing)} | "
              f"ungraded zips dropped: {len(extra)}")
        if missing:
            print(f"  missing assignment numbers: {sorted(missing)}")
        if extra:
            print(f"  dropped (ungraded) assignment numbers: {sorted(extra)}")
        rows = [r for r in rows if submission_number(r["student_id"]) in graded]

    if empty:
        print(f"WARNING: {len(empty)} zips contained no .java files: {empty}")

    sizes = sorted(len(r["code"]) for r in rows)
    print(f"code blob chars: min={sizes[0]} median={sizes[len(sizes) // 2]} "
          f"max={sizes[-1]} (~{sizes[-1] // 4} tokens for the largest)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["student_id", "code"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
