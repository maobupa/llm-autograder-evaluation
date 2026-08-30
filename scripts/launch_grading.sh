#!/usr/bin/env bash
# Launch the 5 LLM graders in parallel on the Menagerie submissions.
#
# Usage:
#   bash scripts/launch_grading.sh smoke --limit 3   # smoke test on 3 subs
#   bash scripts/launch_grading.sh full              # full run on all 279
#
# Outputs:
#   runs/menagerie_<tag>_<model>.csv   (grades)
#   runs/menagerie_<tag>_<model>.log   (stdout+stderr)
#
# Re-running is safe: each grade_multi_model.py process resumes from its own
# output file, so re-launching after a crash just picks up where it left off.
# Make sure no stale processes are still running first:
#     ps -ef | grep grade_multi_model | grep -v grep

set -euo pipefail

tag="${1:-full}"; shift || true
cd "$(dirname "$0")/.."

models=(gpt-4.1 o3 sonnet-4.6 gemini-2.5-flash deepseek-v4-flash)

for m in "${models[@]}"; do
  nohup caffeinate -i python scripts/grade_multi_model.py \
    --input data/menagerie/submissions.csv \
    --rubric data/menagerie/rubric.json \
    --models "$m" \
    --lang java \
    --output "runs/menagerie_${tag}_${m}.csv" \
    --save-every 1 \
    "$@" \
    > "runs/menagerie_${tag}_${m}.log" 2>&1 &
  echo "$m -> PID $!"
done

echo
echo "Logs:   tail -f runs/menagerie_${tag}_<model>.log"
echo "Status: wc -l runs/menagerie_${tag}_*.csv"
echo "Procs:  ps -ef | grep grade_multi_model | grep -v grep"
