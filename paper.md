---
title: 'PyReport: A package for Publication-Ready Statistical Summaries in Python'
tags:
  - Python
  - statistics
  - APA formatting
  - reproducible research
  - open science
authors:
  - name: Arjun B
    orcid: 0009-0003-0241-9796
    affiliation: 1
affiliations:
  - name: All India institute of Medical Sciences, Bhopal
    index: 1
date: 26-04-2026
bibliography: paper.bib
---

# Summary

`pyreport` is a Python package that converts statistical result objects into
publication-ready, human-readable plain-English summaries. It accepts the output
of widely used Python scientific libraries — `scipy`, `statsmodels`, `pandas`, and
`pingouin` — and returns an APA 7th edition formatted narrative string along with a
structured dictionary of extracted statistics. A single function call, `pyreport.report()`,
handles the full dispatch pipeline: type detection, effect-size computation, narrative
generation, and assumption-violation warnings.

For example, a two-sample Welch t-test produces output such as:

> *A Welch two-sample t-test indicated a significant difference between Group A
> (M = 4.23, SD = 1.10) and Group B (M = 3.45, SD = 0.98), t(47.32) = 2.84,
> p = .007, d = 0.75, 95% CI [0.21, 1.28]. The effect size is considered large.*

The package supports thirteen statistical procedures: independent and paired t-tests,
Pearson, Spearman, and Kendall correlations, chi-square and Fisher's exact tests,
Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, OLS regression, logistic /
GLM regression, one-way ANOVA, and descriptive summaries of `pd.DataFrame` objects.

# Statement of Need

Translating raw Python statistical output into manuscript-ready text is a common
bottleneck in empirical research workflows. A researcher running a t-test with
`scipy.stats.ttest_ind` receives a `TtestResult` namedtuple containing a statistic
and a p-value; they must then manually look up degrees of freedom, compute an
effect size (Cohen's *d*), format the p-value according to APA rules (no leading
zero; exact value unless `p < .001`), and compose a grammatically correct sentence —
repeating this process for every test in the manuscript.

This manual step is error-prone and time-consuming. Surveys of published psychology
papers have documented high rates of statistical reporting errors [@Nuijten2016], many
of which stem from transcription mistakes during exactly this translation step.

R's `{report}` package [@Makowski2023] solved this problem elegantly for R users by
providing a single `report()` generic that dispatches on the class of its argument.
No equivalent tool exists for Python: packages such as `pingouin` [@Vallat2018] produce
tidy DataFrames of statistics but leave narrative composition to the user, while APA
formatting helpers (e.g., `pyapa`) are limited to number formatting and do not generate
prose. `pyreport` fills this gap by bringing the same dispatch-and-narrate philosophy
to the Python ecosystem.

The target audience is empirical researchers in psychology, cognitive science,
neuroscience, education, and related fields who conduct their analyses in Python but
need to write up results in APA style. The package is particularly useful in
Jupyter-notebook-based workflows where a rich `Report` object can be printed inline
and its statistics dictionary piped directly into downstream reporting code or
serialised to JSON without custom encoders.

# State of the Field

Several Python tools address adjacent problems:

- **`pingouin`** [@Vallat2018] provides an excellent collection of statistical tests
  with tidy DataFrame output and statistical tables; it does not, however, produce
  narrative prose or APA-formatted sentences.
- **`researchpy`** generates summary tables suitable for APA style but is limited to
  a subset of tests and does not produce complete narrative sentences.
- **`scipy.stats`** and **`statsmodels`** are the standard workhorses for statistical
  computation in Python; their output objects contain raw numbers with no formatting
  or narrative.
- **`pandas`** `DataFrame.describe()` provides basic descriptive statistics but no
  APA-formatted narrative.
- **R `{report}`** [@Makowski2023] is the direct inspiration for `pyreport` and
  offers mature, tested implementations for a wide range of R statistical objects.

`pyreport` occupies a unique niche: it sits on top of existing Python statistical
libraries rather than reimplementing them, acting as a *reporting layer* that transforms
already-computed results into manuscript-ready text with correct APA 7 formatting,
effect sizes, and interpretive labels.

# Design

## Architecture

`pyreport` follows a dispatcher pattern. The public `report()` function (in `core.py`)
uses duck-typed attribute inspection to detect the type of its argument and route it
to the appropriate `BaseReporter` subclass:

```
report(obj, **kwargs)
    │
    ├── pd.DataFrame ──────────────────────────── DataFrameReporter
    ├── scipy TtestResult ──────────────────────── TTestReporter
    ├── scipy PearsonRResult / SpearmanrResult ─── CorrelationReporter
    ├── scipy Chi2ContingencyResult ─────────────── ChiSquareReporter
    ├── scipy OddsRatioResult (Fisher) ──────────── FisherReporter
    ├── scipy MannwhitneyuResult ─────────────────── MannWhitneyReporter
    ├── scipy WilcoxonResult ─────────────────────── WilcoxonReporter
    ├── scipy KruskalResult ──────────────────────── KruskalWallisReporter
    ├── statsmodels RegressionResultsWrapper ────── OLSReporter
    ├── statsmodels LogitResults / GLMResults ────── LogisticReporter
    ├── pingouin t-test DataFrame ─────────────────── TTestReporter (pingouin path)
    └── pingouin ANOVA DataFrame ──────────────────── PingouinAnovaReporter
```

## `Report` Object

Every reporter returns a `Report` instance with four output methods:

| Method | Description |
|---|---|
| `print(r)` / `repr(r)` | Plain-text APA narrative (terminal / Jupyter) |
| `r.to_text()` | Full narrative string including any warnings |
| `r.to_dict()` | `dict` of all extracted statistics (JSON-serialisable) |
| `r.to_dataframe()` | Single-row `pd.DataFrame` for tabular reporting |

All numeric values in `to_dict()` are converted from NumPy scalar types to plain
Python `float`/`int` at construction time, making the dictionary directly
JSON-serialisable without a custom encoder.

## APA 7 Formatting

All formatting is centralised in `formatters.py`:

- **No leading zero** for statistics bounded between −1 and 1 (r, β, p-value).
- **Exact p-values** unless `p < .001`.
- **Negative-zero guard**: `−0.00` → `0.00`.
- **Welch df** rounded to two decimal places.
- **Confidence intervals** rendered as `95% CI [x.xx, x.xx]`.

## Effect Sizes

`effect_sizes.py` implements Cohen's *d* (raw and from *t*), Pearson *r* from *t*,
η², ω², odds ratios with log-method confidence intervals, and rank-biserial *r*,
each paired with Cohen (1988) interpretation labels (small / medium / large).

## Raw-Data Shortcut

For common two-sample comparisons, `pyreport` can run the underlying SciPy test
internally:

```python
pyreport.report((x, y), test="ttest", group_names=("Control", "Treatment"))
pyreport.report((x, y), test="mannwhitney", n1=30, n2=30)
pyreport.report((x, y, z), test="kruskal", k=3, n=90)
```

# Acknowledgements

`pyreport` is inspired by R's `{report}` package developed by Dominique Makowski and
collaborators [@Makowski2023]. The author thanks the developers of `scipy`,
`statsmodels`, `pandas`, and `pingouin` whose work provides the statistical foundation
on which `pyreport` builds.

# References
