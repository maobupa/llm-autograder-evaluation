"""
Grade student submissions with multiple foundation models against a rubric.

Usage:
    python scripts/grade_multi_model.py \
        --input data/graded/diag1_100_random_batch1.csv \
        --rubric data/rubrics/raw/diagnostic1/rubric.json \
        --models o3,gemini-2.5-flash,sonnet-4.6 \
        --output results/diag1_multi_model.csv

Output CSV columns:
    student_id, code,
    <model>_<rubric_id>            (numeric score, easy to process)
    <model>_<rubric_id>_explanation
    <model>_feedback
    <model>_error                  (set when grading failed)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Allow `from src...` imports when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import format_rubric


# Friendly alias -> (provider, model_id)
MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    "o3": {"provider": "openai", "model": "o3"},
    "gpt-4.1": {"provider": "openai", "model": "gpt-4.1"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gemini-2.5-flash": {"provider": "gemini", "model": "gemini-2.5-flash"},
    "gemini-2.5-pro": {"provider": "gemini", "model": "gemini-2.5-pro"},
    "sonnet-4.6": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "opus-4.7": {"provider": "anthropic", "model": "claude-opus-4-7"},
    "haiku-4.5": {"provider": "anthropic", "model": "claude-haiku-4-5"},
    "deepseek": {"provider": "deepseek", "model": "deepseek-v4-flash"},
    "deepseek-v4-flash": {"provider": "deepseek", "model": "deepseek-v4-flash"},
    "deepseek-v4-pro": {"provider": "deepseek", "model": "deepseek-v4-pro"},
}


SYSTEM_PROMPT = (
    "You are a precise and fair grading assistant. "
    "Always respond with valid JSON only."
)

# Language label for the student-code fence in the prompt. Set from --lang.
CODE_LANG = "python"


def build_user_prompt(code: str, rubric: Dict[str, Any]) -> str:
    rubric_text = format_rubric(rubric)
    return f"""You are a grading assistant. Grade the following student code against the provided rubric.

RUBRIC:
{rubric_text}

STUDENT CODE:
```{CODE_LANG}
{code}
```

INSTRUCTIONS:
1. Grade each rubric item by selecting the appropriate option (0 for best, higher numbers for worse).
2. Provide a brief explanation for each grade.
3. Give overall feedback summarizing strengths and weaknesses.
4. Cite specific lines of code in your explanations using line numbers (e.g., "Line 5: ...").

Respond ONLY with a valid JSON object in this exact format:
{{
    "scores": {{
        "rubric_item_id": {{
            "option": <integer>,
            "explanation": "<your explanation>"
        }}
    }},
    "feedback": "<overall feedback>"
}}"""


# ---------- Provider clients (lazy-initialized) ----------

_openai_client = None
_anthropic_client = None
_gemini_client = None
_deepseek_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_deepseek():
    """DeepSeek exposes an OpenAI-compatible API, so we reuse the openai SDK
    pointed at the DeepSeek base URL."""
    global _deepseek_client
    if _deepseek_client is None:
        import openai
        _deepseek_client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
    return _deepseek_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _gemini_client


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def grade_with_model(
    provider: str,
    model: str,
    code: str,
    rubric: Dict[str, Any],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(code, rubric)

    if provider == "openai":
        client = _get_openai()
        # o-series reasoning models don't accept temperature/response_format the same way
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not (model.startswith("o3") or model.startswith("o1") or model.startswith("o4")):
            kwargs["temperature"] = 0.3
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return _extract_json(resp.choices[0].message.content)

    if provider == "deepseek":
        client = _get_deepseek()
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        # DeepSeek V4 models default to thinking mode ON. Disable it explicitly
        # so this is a plain (non-reasoning) grading pass.
        if model.startswith("deepseek-v4"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = client.chat.completions.create(**kwargs)
        return _extract_json(resp.choices[0].message.content)

    if provider == "anthropic":
        client = _get_anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate any text blocks
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _extract_json(text)

    if provider == "gemini":
        client = _get_gemini()
        from google.genai import types as gtypes
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        return _extract_json(resp.text)

    raise ValueError(f"Unknown provider: {provider}")


def resolve_models(spec: str):
    """Parse --models argument into [(label, provider, model_id), ...]."""
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            provider, model = token.split(":", 1)
            label = token.replace(":", "_")
            out.append((label, provider, model))
        elif token in MODEL_ALIASES:
            a = MODEL_ALIASES[token]
            out.append((token, a["provider"], a["model"]))
        else:
            raise ValueError(
                f"Unknown model alias: {token!r}. "
                f"Known aliases: {sorted(MODEL_ALIASES)}. "
                f"Or pass provider:model_id (e.g. openai:gpt-4o)."
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Student submissions CSV (needs student_id, code columns).")
    parser.add_argument("--rubric", required=True, help="Rubric JSON path.")
    parser.add_argument("--models", required=True, help="Comma-separated model aliases or provider:model_id (e.g. o3,gemini-2.5-flash,sonnet-4.6).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows to grade.")
    parser.add_argument("--save-every", type=int, default=5, help="Flush to disk every N submissions.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between model calls.")
    parser.add_argument("--lang", default="python", help="Language label for the student-code fence in the prompt (e.g. java).")
    args = parser.parse_args()

    global CODE_LANG
    CODE_LANG = args.lang

    load_dotenv()

    models = resolve_models(args.models)
    print(f"Models: {[f'{l} ({p}:{m})' for l, p, m in models]}")

    with open(args.rubric) as f:
        rubric = json.load(f)
    rubric_ids = [item["id"] for item in rubric["items"]]
    print(f"Rubric items: {rubric_ids}")

    df = pd.read_csv(args.input)
    if "student_id" not in df.columns or "code" not in df.columns:
        raise ValueError("Input CSV must have 'student_id' and 'code' columns.")
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} submissions from {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resume from existing output if present.
    if os.path.exists(args.output):
        prior = pd.read_csv(args.output)
        results = prior.to_dict(orient="records")
        done_ids = {r["student_id"] for r in results}
        print(f"Resuming: {len(done_ids)} already graded.")
    else:
        results = []
        done_ids = set()

    pbar = tqdm(df.iterrows(), total=len(df))
    for _, row in pbar:
        sid = row["student_id"]
        if sid in done_ids:
            continue
        code = row["code"]
        record: Dict[str, Any] = {"student_id": sid, "code": code}

        for label, provider, model_id in models:
            pbar.set_postfix_str(f"{sid} :: {label}")
            try:
                result = grade_with_model(provider, model_id, code, rubric)
                scores = result.get("scores", {}) or {}
                for rid in rubric_ids:
                    entry = scores.get(rid, {}) or {}
                    opt = entry.get("option")
                    # Coerce to int when possible so the column stays numeric.
                    try:
                        opt = int(opt) if opt is not None else None
                    except (TypeError, ValueError):
                        pass
                    record[f"{label}_{rid}"] = opt
                    record[f"{label}_{rid}_explanation"] = entry.get("explanation", "")
                record[f"{label}_feedback"] = result.get("feedback", "")
                record[f"{label}_error"] = ""
            except Exception as e:
                for rid in rubric_ids:
                    record[f"{label}_{rid}"] = None
                    record[f"{label}_{rid}_explanation"] = ""
                record[f"{label}_feedback"] = ""
                record[f"{label}_error"] = f"{type(e).__name__}: {e}"
                print(f"\n[error] {label} on {sid}: {e}")
            if args.sleep:
                time.sleep(args.sleep)

        results.append(record)
        if len(results) % args.save_every == 0:
            pd.DataFrame(results).to_csv(args.output, index=False)

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\nDone. Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
