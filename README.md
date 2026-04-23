# PyReport

A Python package for publication-ready statistical reporting — a Python port of R's `{report}` package.

`pyreport` takes statistical objects (models, tests, dataframes) as input and returns
human-readable plain-English summaries with statistics formatted to **APA 7** standards.

---

## Installation

```bash
pip install pyreport
```

Or install from source:

```bash
git clone https://github.com/Arjun-Babu-Raj/PyReport.git
cd PyReport
pip install -e ".[dev]"
```

---

## Quick Start

### T-Test

```python
import numpy as np
from scipy import stats
import pyreport

rng = np.random.default_rng(42)
x = rng.normal(4.23, 1.10, 30)   # Group A
y = rng.normal(3.45, 0.98, 30)   # Group B

result = stats.ttest_ind(x, y)
r = pyreport.report(result, group_data=(x, y), group_names=("Group A", "Group B"))
print(r)
```

**Example output:**
```
A Welch two-sample t-test indicated a significant difference between
Group A (M = 4.23, SD = 1.10) and Group B (M = 3.45, SD = 0.98),
t(47.32) = 2.84, p = .007, d = 0.75, 95% CI [0.21, 1.28].
The effect size is considered large.
```

---

### OLS Regression

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pyreport

rng = np.random.default_rng(42)
n = 100
age    = rng.normal(35, 10, n)
income = rng.normal(52000, 14000, n)
educ   = rng.normal(14, 3, n)
score  = 0.32 * age + 0.45 * income/10000 + 0.08 * educ + rng.normal(0, 5, n)

X = sm.add_constant(pd.DataFrame({"Age": age, "income": income, "Education": educ}))
model = sm.OLS(score, X).fit()

r = pyreport.report(model, outcome_name="score")
print(r)
```

**Example output:**
```
The linear regression model explained a significant proportion of variance in score
(R² = .43, F(3, 96) = 24.21, p < .001, adj. R² = .41).
Age (β = 0.32, t = 3.11, p = .002) and income (β = 0.45, t = 4.67, p < .001)
were significant predictors. Education was not significant
(β = 0.08, t = 0.91, p = .365).
```

---

### DataFrame Descriptives

```python
import pandas as pd
import pyreport

df = pd.DataFrame({
    "age":    [25, 34, 45, 23, 55, 41, 38, 29, 62, 18],
    "income": [45000, 52000, 71000, 38000, 83000, 60000, 55000, 42000, 90000, 31000],
    "gender": ["F", "M", "F", "M", "M", "F", "F", "M", "M", "F"],
})

r = pyreport.report(df)
print(r)
```

**Example output:**
```
The dataframe contains 10 observations and 3 variables: 2 numeric
(age: M = 37.00, SD = 13.98, range: [18.00, 62.00]; income: M = 56700.00,
SD = 18793.15, range: [31000.00, 90000.00]) and 1 categorical
(gender: F 50%, M 50%). No missing values were detected.
```

---

### Pearson Correlation

```python
from scipy import stats
import pyreport

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]

result = stats.pearsonr(x, y)
r = pyreport.report(result, n=10, var_names=("Study Hours", "Exam Score"))
print(r)
```

---

### Chi-Square Test

```python
from scipy import stats
import pyreport

table = [[30, 10], [15, 45]]
result = stats.chi2_contingency(table)
r = pyreport.report(result, n=100, var_names=("Treatment", "Outcome"))
print(r)
```

---

### Non-Parametric Tests

```python
import numpy as np
from scipy import stats
import pyreport

rng = np.random.default_rng(42)
x = rng.normal(5, 1, 40)
y = rng.normal(3, 1, 40)

# Mann-Whitney U
result = stats.mannwhitneyu(x, y)
r = pyreport.report(result, n1=40, n2=40, group_names=("Treatment", "Control"))
print(r)

# Kruskal-Wallis H
z = rng.normal(4, 1, 40)
result = stats.kruskal(x, y, z)
r = pyreport.report(result, k=3, n=120)
print(r)
```

---

## The `Report` Object

Every call to `report()` returns a `Report` object with four methods:

```python
r = pyreport.report(result)

r.to_text()        # → plain string for writing/printing
r.to_dict()        # → dict of all extracted statistics
r.to_dataframe()   # → pd.DataFrame (tabular)
repr(r)            # → clean display in Jupyter / terminal
```

### Combining Reports

```python
r1 = pyreport.report(ttest_result, group_data=(x, y))
r2 = pyreport.report(df)
combined = r1 + r2
print(combined)
```

---

## `report_table()` Convenience Function

Returns a `pd.DataFrame` directly (like `report_table()` in R):

```python
tbl = pyreport.report_table(ols_model)
print(tbl)
```

---

## APA 7 Formatting

All statistics are formatted according to APA 7th edition:

| Value type | Example output |
|---|---|
| p-value (any) | `p = .043` or `p < .001` |
| Correlation | `r = .75` |
| Standardised β | `β = -.32` |
| F-ratio | `F(3, 96) = 24.21` |
| t-statistic | `t(47.32) = 2.84` |
| Confidence interval | `95% CI [0.21, 1.28]` |

Key rules:
- **No leading zero** for statistics bounded between −1 and 1 (r, β, p-value)
- **Exact p-values** unless `p < .001`
- **Negative zero guard**: `-0.00` → `0.00`

---

## Supported Input Types

| Object | Source |
|---|---|
| `pd.DataFrame` | pandas |
| `TtestResult` | `scipy.stats.ttest_ind/rel/1samp` |
| `PearsonRResult` / `SignificanceResult` (r, ρ, τ) | `scipy.stats.pearsonr/spearmanr/kendalltau` |
| `Chi2ContingencyResult` | `scipy.stats.chi2_contingency` |
| `SignificanceResult` (Fisher) | `scipy.stats.fisher_exact` |
| `MannwhitneyuResult` | `scipy.stats.mannwhitneyu` |
| `WilcoxonResult` | `scipy.stats.wilcoxon` |
| `KruskalResult` | `scipy.stats.kruskal` |
| OLS results | `statsmodels.OLS.fit()` |
| Logit/GLM results | `statsmodels.Logit/GLM.fit()` |
| Pingouin DataFrames | `pingouin.ttest/anova/corr` |

---

## Package Structure

```
pyreport/
├── __init__.py              # exposes report(), report_table(), Report
├── core.py                  # Report class + report() dispatcher
├── reporters/
│   ├── base.py              # BaseReporter ABC
│   ├── dataframe.py         # pd.DataFrame → descriptives
│   ├── ttest.py             # scipy/pingouin t-test
│   ├── anova.py             # pingouin ANOVA
│   ├── correlation.py       # pearsonr, spearmanr, kendalltau
│   ├── regression.py        # statsmodels OLS, Logit/GLM
│   ├── chi_square.py        # chi2_contingency, fisher_exact
│   └── nonparametric.py     # Mann-Whitney, Kruskal-Wallis, Wilcoxon
├── formatters.py            # APA 7 number/CI/p-value formatters
├── effect_sizes.py          # Cohen's d, r, η², ω², OR, rank-biserial
└── utils.py                 # type detection helpers
```

---

## Dependencies

- `numpy >= 1.22`
- `scipy >= 1.9`
- `pandas >= 1.4`
- `statsmodels >= 0.13`
- `pingouin >= 0.5` (optional, for pingouin output support)

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ --cov=pyreport
```

---

## License

MIT
