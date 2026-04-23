"""
pyreport — Python equivalent of R's {report} package.

Publication-ready, human-readable statistical summaries with APA 7 formatting.

Quick start
-----------
>>> import pyreport
>>> from scipy import stats
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> x = rng.normal(4.2, 1.1, 30)
>>> y = rng.normal(3.5, 1.0, 30)
>>> result = stats.ttest_ind(x, y)
>>> r = pyreport.report(result, group_data=(x, y))
>>> print(r)
"""

from .core import Report, ReportError, ReportWarning, report, report_table
from . import formatters, effect_sizes, utils

__all__ = [
    "report",
    "report_table",
    "Report",
    "ReportError",
    "ReportWarning",
    "formatters",
    "effect_sizes",
    "utils",
]

__version__ = "0.1.0"
