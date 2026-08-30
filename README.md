# Do LLM Graders Agree With Humans, or With Each Other?

Analysis code for a measurement study comparing LLM code graders to human
graders on two introductory-programming datasets.

This repository is a **methods reference**, not a reproduction package. The
student submissions and the human grades cannot be released (see
[Data availability](#data-availability)), so the scripts here will not run
end-to-end. What they do provide is the complete, readable implementation of
every statistic reported in the paper: how the distance correlations were
computed, how the bootstrap confidence intervals were constructed and
bias-corrected, and how the generalizability-theory variance components were
estimated.

If you are doing similar work, the two files most likely to be useful to you
are `analysis/rq1_dcor.py` (the bootstrap and its bias correction) and
`analysis/rq2_gstudy.py` (the G-study estimators).

---

## What the study does

Five LLM graders (GPT-4.1, o3, Gemini 2.5 Flash, Claude Sonnet 4.6, and
DeepSeek V4 Flash) graded the same student work as the human graders who
originally marked it, using the same rubric. We then asked how much the graders
agreed, and where the agreement broke down.

Two datasets were chosen to sit at opposite ends of the assessment space:

| | **CIP** | **Menagerie** |
|---|---|---|
| Task | one short Python console program | multi-file Java project |
| Specification | fully determined input/output | open-ended; student invents the domain |
| Submissions | 100 | 279 |
| Human graders | 5, each grading all 100 | 28, in 7 groups of 4, each grading ~40 |
| Design | fully crossed | nested within groups |
| Rubric | 4 items, 3-point (0 = correct) | 4 skills, 14-point (`A++`…`F`) |
| Rubric items | input, conditional logic, printing, syntax errors | correctness, code elegance, readability, documentation |

Two research questions are addressed here:

- **RQ1 — Alignment.** Squared bias-corrected distance correlation between
  graders: human–LLM, between-LLM, within-LLM, and human–human, on the same
  rubric item and across different rubric items.
- **RQ2 — Grading systems.** Generalizability-theory decomposition of score
  variance, run separately for the human and LLM systems, and combined with
  rater type as a fixed facet.

## Repository layout

```
analysis/     all statistics reported in the paper
  rq1_dcor.py                  RQ1, CIP: dCor + transcript bootstrap
  rq1_dcor_menagerie.py        RQ1, Menagerie: NaN-aware dCor for the nested design
  rq1_dcor_humanhuman.py       human-human agreement + pair-type contrasts
  rq2_gstudy.py                RQ2, CIP: per-system G-study
  rq2_gstudy_menagerie.py      RQ2, Menagerie: per-system G-study
  rq2_combined.py              RQ2, CIP: rater type as a fixed facet
  rq2_combined_menagerie.py    RQ2, Menagerie: same
  appendix_c_outputs.py        regenerates appendix figures and LaTeX tables

scripts/      data collection and preparation
  grade_multi_model.py         multi-provider grading runner (prompts + decoding config)
  retry_errors.py              re-queries failed gradings with an identical prompt
  compile_cip.py               builds runs/cip_compiled.csv from raw grader outputs
  compile_menagerie.py         builds runs/menagerie_compiled.csv
  build_menagerie_input.py     assembles multi-file Java projects into single blobs

src/utils.py  rubric formatting shared by the grading scripts
results/      aggregate outputs: correlations, variance components, CIs, figures
```

`data/` and `runs/` are intentionally absent; see below.

## Data availability

**Student submissions and individual human grades are not published.** They are
coursework produced by identifiable students and marked by identifiable
instructors, and cannot be redistributed.

`data/` (submissions and rubrics) and `runs/` (raw grading outputs and compiled
score matrices) are therefore excluded from this repository. `results/`
**is** included: it contains only aggregate statistics — correlations, variance
components, and confidence intervals — with no student work, no student
identifiers, and no per-grader scores.

Human graders are pseudonymous throughout. `compile_cip.py` discovers grader
identifiers from the private source file at load time and maps them to
`human_1`…`human_N` on output, so no real name appears in this repository, in
the compiled data, or in any result.

## Input data format

The analysis scripts read two compiled files. If you want to run this code on
your own data, these are the formats to produce.

**`runs/cip_compiled.csv`** — wide, one row per submission:

```
student_id, human_1_input, human_1_conditional_logic, ..., deepseek_syntax_errors
```

That is, `student_id` followed by one column per (rater × rubric item), named
`<rater>_<item>`. Raters are `human_1`…`human_5` and the five model aliases.
Scores are integers 0–2. The CIP design is fully crossed, so there are no
missing cells.

**`runs/menagerie_compiled.csv`** — long, one row per grading:

```
submission, skill, rater, rater_type, group, round, score
```

`rater` is prefixed by type (`human_3`, `llm_gpt-4.1`); `rater_type` is
`human` or `llm`; `score` is an integer 1–14 (`F` = 1, `A++` = 14) with human
letter grades and LLM option indices both mapped onto that shared ordinal.
Human graders only marked their own group's submissions, so the human side is
sparse by design.

## Using the code

```bash
pip install -r requirements.txt
python analysis/rq1_dcor.py            # writes results/dcor_cip_*
python analysis/rq1_dcor_menagerie.py
python analysis/rq1_dcor_humanhuman.py
python analysis/rq2_gstudy.py          # writes results/gstudy_cip_*
python analysis/rq2_combined.py
python analysis/appendix_c_outputs.py  # figures + LaTeX tables from the CSVs above
```

Each script is self-contained, reads a compiled CSV, and writes CSVs and a
figure into `results/`. They take a few minutes each: the bootstrap default is
2000 replicates, and every replicate recomputes the entire pipeline. Lower
`N_BOOT` at the top of a script for a faster pass.

Running the grading itself requires API keys for all five providers in a `.env`
file, plus the private data:

```bash
python scripts/grade_multi_model.py \
    --input data/cip/submissions.csv \
    --rubric data/cip/rubric/rubric.json \
    --models gpt-4.1,o3,gemini-2.5-flash,sonnet-4.6,deepseek \
    --output runs/multi_model.csv --lang python
```

## Two methodological notes

**The bootstrap confidence intervals are bias-corrected, and they need to be.**
Resampling submissions with replacement creates duplicate observations, which
are exact ties in both distance matrices. Ties add spurious dependence and
inflate `dCor²`. The inflation scales as `O(1/n)`, so it is roughly ten times
worse for cells with ~40 paired observations (human–LLM, human–human on
Menagerie, where each human graded only their own group) than for cells with
279. Left uncorrected it pushed every affected percentile interval *entirely
above its own point estimate*. All intervals subtract the bootstrap bias
estimate `b = mean(draws) − point` from both percentiles, preserving the
bootstrap's dispersion while recentring on the point estimate. See
`percentile_ci` in `analysis/rq1_dcor_menagerie.py`. If you bootstrap a
distance correlation on a small sample, check for this.

**dCor is computed pairwise-complete.** Menagerie's human graders are nested
within groups, so most human pairs share no submissions at all. Distance
correlations are computed on the non-missing intersection of each pair, and
pairs overlapping on fewer than `MIN_N = 10` submissions are dropped rather
than estimated. Human–human agreement on Menagerie is therefore reported only
for the 42 of 378 pairs that clear that threshold — a within-group,
non-random subset.

## Requirements

Python 3.9+. Key dependencies: `dcor`, `pandas`, `numpy`, `scipy`,
`matplotlib`, `seaborn`. Provider SDKs (`openai`, `anthropic`, `google-genai`)
are needed only to run the grading scripts, not the analysis. Full list in
`requirements.txt`.

## Citation

Citation details will be added on publication.
