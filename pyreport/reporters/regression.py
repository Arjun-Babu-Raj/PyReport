"""
Regression reporters — statsmodels OLS and Logistic/GLM.
"""

from __future__ import annotations

from typing import List

from ..core import Report
from ..formatters import fmt_stat, fmt_bound, fmt_p_full, fmt_df
from .base import BaseReporter


class OLSReporter(BaseReporter):
    """
    Report a fitted statsmodels OLS result.

    kwargs
    ------
    outcome_name : str
        Name of the dependent variable (default: "the outcome").
    effectsize : bool
        Whether to include effect sizes (default: True).
    ci_level : float
        Confidence level (default: 0.95).
    """

    def report(self) -> Report:
        obj = self.obj
        outcome = self.kwargs.get("outcome_name", "the outcome")
        ci_level: float = self.kwargs.get("ci_level", 0.95)

        r2 = float(obj.rsquared)
        adj_r2 = float(obj.rsquared_adj)
        F_val = float(obj.fvalue)
        p_model = float(obj.f_pvalue)
        df_model = float(obj.df_model)
        df_resid = float(obj.df_resid)

        params = obj.params
        pvalues = obj.pvalues
        tvalues = obj.tvalues

        # Confidence intervals
        try:
            ci = obj.conf_int(alpha=1 - ci_level)
        except Exception:
            ci = None

        F_str = fmt_stat(F_val)
        df1_str = fmt_df(df_model)
        df2_str = fmt_df(df_resid)
        p_str = fmt_p_full(p_model)

        sig_model = "significant" if p_model < 0.05 else "non-significant"

        text = (
            f"The linear regression model explained a {sig_model} proportion "
            f"of variance in {outcome} (R² = {fmt_bound(r2)}, "
            f"F({df1_str}, {df2_str}) = {F_str}, {p_str}, "
            f"adj. R² = {fmt_bound(adj_r2)}). "
        )

        # Predictor details — skip 'Intercept' / 'const'
        sig_preds: List[str] = []
        ns_preds: List[str] = []
        skip = {"intercept", "const"}

        for name in params.index:
            if str(name).lower() in skip:
                continue
            b = float(params[name])
            t = float(tvalues[name])
            p = float(pvalues[name])
            b_str = fmt_bound(b)
            t_str = fmt_stat(t)
            p_full = fmt_p_full(p)
            pred_clause = f"{name} (β = {b_str}, t = {t_str}, {p_full})"
            if p < 0.05:
                sig_preds.append(pred_clause)
            else:
                ns_preds.append(pred_clause)

        if sig_preds:
            text += ", ".join(sig_preds) + " "
            text += f"{'was' if len(sig_preds) == 1 else 'were'} significant predictor{'s' if len(sig_preds) > 1 else ''}. "

        if ns_preds:
            if len(ns_preds) > 1:
                text += ", ".join(ns_preds[:-1]) + ", and " + ns_preds[-1]
            else:
                text += ns_preds[0]
            text += f" {'was' if len(ns_preds) == 1 else 'were'} not significant."

        # Statistics dict
        statistics: dict = {
            "test": "OLS regression",
            "R2": round(r2, 4),
            "adj_R2": round(adj_r2, 4),
            "F": round(F_val, 4),
            "df_model": df_model,
            "df_resid": df_resid,
            "p_model": round(p_model, 4),
            "predictors": {},
        }
        for name in params.index:
            if str(name).lower() in skip:
                continue
            pred_stats = {
                "beta": round(float(params[name]), 4),
                "t": round(float(tvalues[name]), 4),
                "p": round(float(pvalues[name]), 4),
            }
            if ci is not None:
                pred_stats["ci_lower"] = round(float(ci.loc[name, 0]), 4)
                pred_stats["ci_upper"] = round(float(ci.loc[name, 1]), 4)
            statistics["predictors"][str(name)] = pred_stats

        return Report(text.strip(), statistics, self._warnings)


class LogisticReporter(BaseReporter):
    """
    Report a fitted statsmodels Logit or GLM (logistic) result.

    kwargs
    ------
    outcome_name : str
        Name of the dependent variable (default: "the outcome").
    ci_level : float
        Confidence level (default: 0.95).
    """

    def report(self) -> Report:
        obj = self.obj
        outcome = self.kwargs.get("outcome_name", "the outcome")
        ci_level: float = self.kwargs.get("ci_level", 0.95)

        # McFadden's pseudo-R²
        pr2 = getattr(obj, "prsquared", None)
        llf = float(obj.llf)
        llnull = float(obj.llnull) if hasattr(obj, "llnull") else None

        params = obj.params
        pvalues = obj.pvalues
        tvalues = obj.tvalues

        try:
            ci = obj.conf_int(alpha=1 - ci_level)
        except Exception:
            ci = None

        import math

        # Odds ratios
        betas = {n: float(params[n]) for n in params.index}
        ors = {n: math.exp(b) for n, b in betas.items()}

        skip = {"intercept", "const"}

        header = f"Logistic regression predicting {outcome}"
        if pr2 is not None:
            header += f" (McFadden's R² = {fmt_bound(float(pr2))}, log-likelihood = {fmt_stat(llf, 2)})"
        header += ". "

        lines: List[str] = []
        for name in params.index:
            if str(name).lower() in skip:
                continue
            b = betas[name]
            OR = ors[name]
            p = float(pvalues[name])
            t = float(tvalues[name])
            b_str = fmt_bound(b)
            t_str = fmt_stat(t)
            p_full = fmt_p_full(p)
            sig = "" if p < 0.05 else " (ns)"
            or_str = fmt_stat(OR)
            line = f"{name}: β = {b_str}, OR = {or_str}, t = {t_str}, {p_full}{sig}"
            lines.append(line)

        text = header + "; ".join(lines) + "."

        statistics: dict = {
            "test": "logistic regression",
            "log_likelihood": round(llf, 4),
            "predictors": {},
        }
        if pr2 is not None:
            statistics["mcfadden_r2"] = round(float(pr2), 4)

        for name in params.index:
            if str(name).lower() in skip:
                continue
            pred_stats = {
                "beta": round(float(params[name]), 4),
                "OR": round(float(ors[name]), 4),
                "t": round(float(tvalues[name]), 4),
                "p": round(float(pvalues[name]), 4),
            }
            if ci is not None:
                pred_stats["ci_lower"] = round(float(ci.loc[name, 0]), 4)
                pred_stats["ci_upper"] = round(float(ci.loc[name, 1]), 4)
            statistics["predictors"][str(name)] = pred_stats

        return Report(text, statistics, self._warnings)
