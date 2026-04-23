"""
DataFrame descriptive statistics reporter.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..core import Report
from ..formatters import fmt_stat
from .base import BaseReporter


class DataFrameReporter(BaseReporter):
    """Generate descriptive statistics for a ``pd.DataFrame``."""

    def report(self) -> Report:
        df: pd.DataFrame = self.obj
        n_obs, n_vars = df.shape

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Missing values
        n_missing = int(df.isnull().sum().sum())

        # Build description strings
        parts: List[str] = []
        statistics: dict = {
            "n_observations": n_obs,
            "n_variables": n_vars,
            "n_numeric": len(numeric_cols),
            "n_categorical": len(cat_cols),
            "n_missing": n_missing,
        }

        # Header
        parts.append(
            f"The dataframe contains {n_obs} observation{'s' if n_obs != 1 else ''} "
            f"and {n_vars} variable{'s' if n_vars != 1 else ''}: "
            f"{len(numeric_cols)} numeric"
        )

        # Numeric summaries
        num_details: List[str] = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            m = series.mean()
            sd = series.std(ddof=1)
            mn = series.min()
            mx = series.max()
            col_str = f"{col}: M = {fmt_stat(m, 2)}, SD = {fmt_stat(sd, 2)}, range: [{fmt_stat(mn, 2)}, {fmt_stat(mx, 2)}]"
            num_details.append(col_str)
            statistics[f"{col}_mean"] = round(m, 4)
            statistics[f"{col}_sd"] = round(sd, 4)
            statistics[f"{col}_min"] = round(float(mn), 4)
            statistics[f"{col}_max"] = round(float(mx), 4)

        if num_details:
            parts[-1] += f" ({'; '.join(num_details)})"
        else:
            parts[-1] += " (none)"

        # Categorical summaries
        cat_details: List[str] = []
        for col in cat_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            vc = series.value_counts(normalize=True)
            cat_str = col + ": " + ", ".join(
                f"{cat} {pct:.0%}" for cat, pct in vc.items()
            )
            cat_details.append(cat_str)
            statistics[f"{col}_counts"] = series.value_counts().to_dict()

        if cat_details:
            parts[-1] += f" and {len(cat_cols)} categorical ({'; '.join(cat_details)})"
        elif len(cat_cols) > 0:
            parts[-1] += f" and {len(cat_cols)} categorical"

        # Missing
        if n_missing == 0:
            parts.append("No missing values were detected.")
        else:
            parts.append(f"{n_missing} missing value(s) detected across the dataset.")
            self._warn(f"{n_missing} missing values detected; results based on available cases.")

        text = " ".join(parts)
        return Report(text, statistics, self._warnings)
