# Evaluating LLM Autograders Without Ground Truth

Analysis code for the paper **"Evaluating LLM Autograders Without Ground Truth"** —
Huijun Mao and Chris Piech, Stanford University.

LLM autograders are usually evaluated against human "ground truth": a single
reference label from one annotator, or an aggregate of several. This study
instead treats humans and LLMs as **two grading systems** and characterizes
their agreement and disagreement directly, using distance correlation and
generalizability theory across two introductory-programming datasets.

This repository is primarily a **methods reference**. The CIP submissions and
human grades cannot be released, so that half will not run end-to-end; the
Menagerie dataset is public, so that half can be reproduced if you obtain it
(see [Data availability](#data-availability)). What the code provides in either
case is the complete, readable implementation of every statistic reported in the
paper: how the distance correlations were computed, how the bootstrap confidence
intervals were constructed and bias-corrected, and how the
generalizability-theory variance components were estimated.

If you are doing similar work, the two files most likely to be useful to you
are `analysis/rq1_dcor.py` (the bootstrap and its bias correction) and
`analysis/rq2_gstudy.py` (the G-study estimators).

---

## What the study does

Five LLM graders (GPT-4.1, o3, Gemini 2.5 Flash, Claude Sonnet 4.6, and
DeepSeek-V4) graded the same student work as the human graders who originally
marked it, using the same prompt and rubric the humans used, at temperature
0.3. We then asked how much the graders agreed, and where the agreement broke
down.

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

The paper abbreviates the CIP rubric items; the code uses the fuller column
names. They map as `input` → `input`, `logic` → `conditional_logic`,
`print` → `printing`, `syntax` → `syntax_errors`.

The repository applies two established methods
to this setting:

- **The dCor analysis — alignment.** Squared bias-corrected distance
  correlation between graders: human–LLM, between-LLM, within-LLM, and
  human–human, on the same rubric item and across different rubric items.
  The measure and its use for LLM–human misalignment follow
  [Hardy and Kim (2026)](https://arxiv.org/abs/2603.00883).
- **The G-study — grading systems.** Generalizability-theory decomposition of
  score variance, run separately for the human and LLM systems, and combined
  with rater type as a fixed facet. The design and estimators follow Brennan,
  *Generalizability Theory* (Springer, 2001); the variance components here are
  estimated with standard Python libraries rather than Brennan's own software.

## Repository layout

```
analysis/     all statistics reported in the paper
  rq1_dcor.py                  dCor, CIP: dCor + transcript bootstrap
  rq1_dcor_menagerie.py        dCor, Menagerie: NaN-aware dCor for the nested design
  rq1_dcor_humanhuman.py       human-human agreement + pair-type contrasts
  rq1_dcor_perhuman.py         dCor, CIP: the same 2x3 panel, one bar per human
  rq2_gstudy.py                G-study, CIP: per-system variance components
  rq2_gstudy_menagerie.py      G-study, Menagerie: per-system variance components
  rq2_combined.py              G-study, CIP: rater type as a fixed facet
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

## Finding the code behind each result

| In the paper | Script | Output |
|---|---|---|
| Figure 1 — Menagerie dCor panel | `analysis/rq1_dcor_menagerie.py` | `results/dcor_menagerie_figure3.png` |
| Figure 2 — CIP dCor panel | `analysis/rq1_dcor.py` | `results/dcor_cip_figure3.png` |
| §3.1 bootstrap gap test (Δ dCor²*n*, *p*) | same two scripts | `results/dcor_{cip,menagerie}_difference_test.csv` |
| Table 1 — σ²*ₛ*ₓ*ₜ* and *r*ₛᵧₛ | `analysis/rq2_combined.py`, `analysis/rq2_combined_menagerie.py` | `results/gstudy_{cip,menagerie}_combined.csv` |
| §3.2 per-system variance components | `analysis/rq2_gstudy.py`, `analysis/rq2_gstudy_menagerie.py` | `results/gstudy_{cip,menagerie}_variance_components.csv` |
| Appendix A — grading prompts | `scripts/grade_multi_model.py`, `src/utils.py` | — |
| Appendix C.1 — Human–Human dCor | `analysis/rq1_dcor_humanhuman.py` | `results/appendix_c_dcor_humanhuman.png`, `results/dcor_humanhuman_{cells,contrasts}.csv` |
| Appendix figures and LaTeX tables | `analysis/appendix_c_outputs.py` | `results/*.png`, `paper/tables/*.tex` |

Not in the paper: `analysis/rq1_dcor_perhuman.py` reruns the CIP panel with one
bar per human grader instead of per LLM (`results/dcor_cip_perhuman.png`). It
shows that human–LLM agreement is flat across the five humans (0.50–0.52),
while *within*-human cross-rubric correlation varies widely (0.23–0.77) —
individual halo effect, largely absent in the LLMs.

## Data availability

**CIP student submissions and individual human grades are not published.** They
are coursework produced by identifiable students and marked by identifiable
instructors, and cannot be redistributed.

**Menagerie is a public dataset** [Messer et al. (2025)](https://osf.io/q8jbt/overview) and can be obtained from
its authors. If you get it and build `runs/menagerie_compiled.csv` in the format
below, the Menagerie half of this analysis runs end to end. The CIP half cannot
be reproduced without the private data.

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

A BibTeX entry and venue details will be added on publication.

Citations for the two methods applied here:

```bibtex
@misc{hardy2026knowledgewisdommeasuringmisalignment,
  title         = {Knowledge without Wisdom: Measuring Misalignment between LLMs and Intended Impact},
  author        = {Michael Hardy and Yunsung Kim},
  year          = {2026},
  eprint        = {2603.00883},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2603.00883}
}

@book{Brennan2001,
  author    = {Brennan, Robert L.},
  title     = {Generalizability Theory},
  year      = {2001},
  publisher = {Springer},
  address   = {New York, NY},
  series    = {Statistics for Social and Behavioral Sciences},
  doi       = {10.1007/978-1-4757-3456-0},
  isbn      = {978-1-4757-3456-0}
}
```

The Menagerie dataset is due to [Messer et al. (2025)](https://osf.io/q8jbt/overview); CIP submissions come from
[Code in Place (Stanford University, 2024)](https://codeinplace.stanford.edu/). Please cite those sources
independently if you use them.
