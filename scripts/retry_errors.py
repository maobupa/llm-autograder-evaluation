"""
scripts/retry_errors.py -- re-grade rows with errors in per-model output CSVs.

For each input CSV (default: runs/menagerie_full_*.csv):
  1. Detect rows where <model>_error is non-empty.
  2. Re-query the model -- same prompt / system / temperature / json mode as
     grade_multi_model.py, so the retry is the same grading process.
  3. Up to 3 attempts per row: attempt 0 is identical; attempts 1+ append a
     one-line "please escape quotes/newlines properly" reminder, which
     overwhelmingly fixes the DeepSeek-class "unescaped quote in string value"
     failures we've seen.
  4. EVERY raw response is appended to runs/retry_raw/retry_raw_<model>.jsonl
     -- so if a parse still fails the text is preserved (lesson learned from
     the first run, where the raw response was lost when parse raised).
  5. CSV write is atomic (tmp -> rename) with a one-time .bak backup.

Usage:
  python scripts/retry_errors.py                # default glob, real retry
  python scripts/retry_errors.py --dry-run      # just count errors, no APIs
  python scripts/retry_errors.py --files runs/menagerie_full_o3.csv
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import pandas as pd
from dotenv import load_dotenv

# Reuse the main grader's prompt builder, system prompt, model alias map,
# provider client getters, and JSON extractor -- so this retry is the SAME
# grading process, not a fork.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade_multi_model as G

RUBRIC_IDS = ["correctness", "code_elegance", "readability", "documentation"]
MAX_ATTEMPTS = 3            # parse-retries with stricter prompt
MAX_TRANSIENT_RETRIES = 8   # rate-limit + 5xx backoff retries (don't count as attempt)
STRICTER_NOTE = (
    "\n\nIMPORTANT: Return ONLY a valid JSON object. Properly escape all "
    "double-quotes (\\\") and newlines (\\n) inside any string value, "
    "including the explanation and feedback fields."
)


def _is_rate_limit(e):
    s = f"{type(e).__name__} {e}"
    return ("RateLimit" in type(e).__name__) or "429" in s or "rate_limit" in s.lower()


def _is_transient(e):
    name = type(e).__name__
    s = str(e)
    return (name in {"ServerError", "InternalServerError", "APIConnectionError",
                     "APITimeoutError", "ConnectionError", "Timeout",
                     "ReadTimeout", "ConnectTimeout"}
            or "502" in s or "503" in s or "504" in s
            or "Bad Gateway" in s or "Service Unavailable" in s)


# ---------- model dispatch (mirrors grade_multi_model.grade_with_model but
#            returns the raw response BEFORE the JSON parse) ----------
def call_model_raw(provider, model, code, rubric, stricter=False):
    user = G.build_user_prompt(code, rubric)
    if stricter:
        user += STRICTER_NOTE

    if provider == "openai":
        client = G._get_openai()
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": G.SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        if not (model.startswith("o3") or model.startswith("o1")
                or model.startswith("o4")):
            kwargs["temperature"] = 0.3
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    if provider == "deepseek":
        client = G._get_deepseek()
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": G.SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        if model.startswith("deepseek-v4"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    if provider == "anthropic":
        client = G._get_anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            system=G.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text")

    if provider == "gemini":
        client = G._get_gemini()
        from google.genai import types as gtypes
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=gtypes.GenerateContentConfig(
                system_instruction=G.SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        return resp.text

    raise ValueError(f"unknown provider: {provider}")


# ---------- helpers ----------
def detect_model_label(path):
    """Read the header to find the `<label>_<rubric_id>` column family."""
    cols = pd.read_csv(path, nrows=0).columns
    for rid in RUBRIC_IDS:
        suf = "_" + rid
        for c in cols:
            if c.endswith(suf):
                return c[: -len(suf)]
    return None


def resolve_provider(label):
    if label in G.MODEL_ALIASES:
        return G.MODEL_ALIASES[label]["provider"], G.MODEL_ALIASES[label]["model"]
    raise ValueError(
        f"unknown model alias {label!r}; "
        f"known: {sorted(G.MODEL_ALIASES)}"
    )


def _call_with_backoff(provider, model, code, rubric, stricter, raw_log):
    """Call the model. Transparently sleep + retry on rate-limit / 5xx /
    connection errors; raise everything else. Returns raw text on success."""
    for k in range(MAX_TRANSIENT_RETRIES):
        try:
            return call_model_raw(provider, model, code, rubric, stricter=stricter)
        except Exception as e:
            if _is_rate_limit(e):
                wait = min(60 * (2 ** k), 300)        # 60, 120, 240, 300, ...
                msg = f"rate limit (try {k+1}/{MAX_TRANSIENT_RETRIES}); sleeping {wait}s"
            elif _is_transient(e):
                wait = min(10 * (2 ** k), 120)        # 10, 20, 40, 80, 120
                msg = f"transient {type(e).__name__} (try {k+1}/{MAX_TRANSIENT_RETRIES}); sleeping {wait}s"
            else:
                raise
            print(f"    [backoff] {msg}: {str(e)[:120]}")
            raw_log({"backoff": True, "wait_s": wait,
                     "exception_class": type(e).__name__,
                     "exception_msg": str(e)[:500]})
            time.sleep(wait)
    raise RuntimeError(f"exceeded {MAX_TRANSIENT_RETRIES} transient retries")


def retry_one(provider, model, code, rubric, raw_log, max_attempts=MAX_ATTEMPTS):
    """Returns (parsed_dict_or_None, error_string).

    Outer loop: up to `max_attempts` parse attempts (each attempt > 0 appends a
    stricter-JSON instruction to the prompt).  Inside each attempt, rate-limit
    and 5xx errors trigger transparent sleep + re-call without consuming a
    parse-attempt slot."""
    last_err = ""
    for attempt in range(max_attempts):
        stricter = attempt > 0
        try:
            raw = _call_with_backoff(provider, model, code, rubric, stricter, raw_log)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            raw_log({"attempt": attempt, "stricter": stricter,
                     "exception": last_err})
            continue
        raw_log({"attempt": attempt, "stricter": stricter, "raw": raw})
        try:
            return G._extract_json(raw), ""
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
            continue
    return None, last_err


def process_file(path, rubric, raw_log_dir, dry_run=False):
    label = detect_model_label(path)
    if not label:
        print(f"  [skip] no `<model>_<rubric_id>` columns: {path}")
        return 0, 0, 0
    err_col = f"{label}_error"
    df = pd.read_csv(path)
    errored = df[df[err_col].fillna("").str.len() > 0]
    print(f"\n{path}")
    print(f"  model={label}  errored={len(errored)} / {len(df)}")
    if dry_run or len(errored) == 0:
        return len(errored), 0, 0

    provider, model = resolve_provider(label)
    os.makedirs(raw_log_dir, exist_ok=True)
    log_path = os.path.join(raw_log_dir, f"retry_raw_{label}.jsonl")

    fixed, still_failed = 0, 0
    t0 = time.time()
    for idx, row in errored.iterrows():
        sid = row["student_id"]
        def raw_log(d, _sid=sid):
            with open(log_path, "a") as f:
                f.write(json.dumps({**d, "student_id": _sid,
                                    "ts": time.time()}) + "\n")
        result, err = retry_one(provider, model, row["code"], rubric, raw_log)
        if result is not None:
            scores = result.get("scores", {}) or {}
            for rid in RUBRIC_IDS:
                entry = scores.get(rid, {}) or {}
                opt = entry.get("option")
                try:
                    opt = int(opt) if opt is not None else None
                except (TypeError, ValueError):
                    pass
                df.at[idx, f"{label}_{rid}"] = opt
                df.at[idx, f"{label}_{rid}_explanation"] = entry.get("explanation", "")
            df.at[idx, f"{label}_feedback"] = result.get("feedback", "")
            df.at[idx, err_col] = ""
            fixed += 1
            print(f"  FIXED  {sid}")
        else:
            df.at[idx, err_col] = err
            still_failed += 1
            print(f"  FAILED {sid}  {err}")

    # Atomic write with one-time backup.
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"  done in {(time.time()-t0):.1f}s. "
          f"fixed={fixed} still_failed={still_failed}  (backup: {bak})")
    return len(errored), fixed, still_failed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+",
                    default=sorted(glob.glob("runs/menagerie_full_*.csv")),
                    help="Per-model grade_multi_model output CSVs.")
    ap.add_argument("--rubric", default="data/menagerie/rubric.json")
    ap.add_argument("--raw-log-dir", default="runs/retry_raw")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report error counts without calling any APIs.")
    args = ap.parse_args()

    load_dotenv()
    with open(args.rubric) as f:
        rubric = json.load(f)

    files = [p for p in args.files if os.path.exists(p)]
    missing = [p for p in args.files if not os.path.exists(p)]
    for p in missing:
        print(f"  [skip] not found: {p}", file=sys.stderr)
    if not files:
        sys.exit("no input files found (default glob: runs/menagerie_full_*.csv)")

    total_err, total_fixed, total_failed = 0, 0, 0
    for p in files:
        e, f, x = process_file(p, rubric, args.raw_log_dir, dry_run=args.dry_run)
        total_err += e
        total_fixed += f
        total_failed += x

    print(f"\n=== SUMMARY ===")
    print(f"  files processed: {len(files)}")
    print(f"  errored rows:    {total_err}")
    if not args.dry_run:
        print(f"  fixed:           {total_fixed}")
        print(f"  still failing:   {total_failed}")
        if total_failed:
            print(f"  raw responses preserved in {args.raw_log_dir}/")


if __name__ == "__main__":
    main()
