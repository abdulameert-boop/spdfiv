"""Diagnostics for the distribution of unit-specific IV estimates.

Under the random-coefficients assumption of Chen et al. (2025), the
unit-specific estimates theta_i are independent across units, and the
mean-group (MG) estimator is the sample mean of their distribution. This
module provides

* robust summaries of that distribution (trimmed mean, median), with
  inference from a nonparametric bootstrap over units, and
* influence diagnostics: the change in the MG estimate when a unit is
  deleted, and modified z-scores (Iglewicz and Hoaglin 1993) that flag
  outlying unit-specific estimates.

The trimmed mean and the median estimate the centre of the distribution of
theta_i. They coincide with the mean E(theta_i) only when that distribution
is symmetric; they are offered as diagnostics, not as replacements for MG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _require_mg(res):
    if res.unit_params is None:
        raise ValueError("Unit-specific estimates are only available with estimator='mg'.")
    return res.unit_params


def _location(X: np.ndarray, method: str, trim: float) -> np.ndarray:
    if method == "mean":
        return X.mean(axis=0)
    if method == "trimmed":
        return stats.trim_mean(X, trim, axis=0)
    if method == "median":
        return np.median(X, axis=0)
    raise ValueError("method must be 'mean', 'trimmed' or 'median'.")


def mg_robust(res, method: str = "trimmed", trim: float = 0.1, exclude=None,
              bootstrap: int = 999, level: float = 0.95, seed=None) -> pd.DataFrame:
    """Robust location estimates of the unit-specific coefficients.

    Parameters
    ----------
    method : {"trimmed", "median", "mean"}
        Location estimator applied coefficient by coefficient.
    trim : float
        Proportion cut from *each* tail for the trimmed mean.
    exclude : list, optional
        Units removed before computing the estimates (e.g. those flagged by
        :func:`mg_influence`).
    bootstrap : int
        Number of bootstrap resamples of units for standard errors and
        percentile confidence intervals.
    """
    Th = _require_mg(res)
    if exclude is not None:
        Th = Th.drop(index=list(exclude))
    X = Th.values
    Ng = X.shape[0]
    est = _location(X, method, trim)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap, X.shape[1]))
    for b in range(bootstrap):
        boot[b] = _location(X[rng.integers(0, Ng, Ng)], method, trim)
    alpha = (1 - level) / 2
    mg = X.mean(axis=0)
    out = pd.DataFrame({
        "MG": mg,
        "MG_se": X.std(axis=0, ddof=1) / np.sqrt(Ng),
        method: est,
        "se": boot.std(axis=0, ddof=1),
        "lower": np.quantile(boot, alpha, axis=0),
        "upper": np.quantile(boot, 1 - alpha, axis=0),
    }, index=Th.columns)
    out.attrs.update(method=method, trim=trim, n_units=Ng, bootstrap=bootstrap)
    return out


def mg_influence(res, threshold: float = 3.5, cond_threshold: float = 1e12) -> pd.DataFrame:
    """Influence of each unit on the mean-group estimate.

    Returns a long DataFrame with, for every unit and coefficient,

    * ``theta_i`` - the unit-specific estimate,
    * ``mod_z`` - the modified z-score 0.6745 (theta_i - median) / MAD,
    * ``dfmg`` - the change in the MG estimate from deleting the unit,
      MG - MG(-i) = (theta_i - MG) / (N - 1), in units of the MG standard
      error,
    * ``flag`` - ``|mod_z| > threshold`` (3.5 as recommended by Iglewicz
      and Hoaglin 1993),
    * ``cond`` - the condition number of the unit's instrument cross-product
      Z_i'Z_i, and ``ill_conditioned`` - ``cond > cond_threshold``; such units
      have (nearly) collinear instruments, so their estimates depend on
      numerical tolerances.
    """
    Th = _require_mg(res)
    X = Th.values
    Ng = X.shape[0]
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        modz = 0.6745 * (X - med) / mad
    mg = X.mean(axis=0)
    se = X.std(axis=0, ddof=1) / np.sqrt(Ng)
    dfmg = (X - mg) / (Ng - 1) / se
    long = []
    for j, nm in enumerate(Th.columns):
        long.append(pd.DataFrame({"unit": Th.index, "parameter": nm,
                                  "theta_i": X[:, j], "mod_z": modz[:, j],
                                  "dfmg": dfmg[:, j]}))
    out = pd.concat(long, ignore_index=True)
    out["flag"] = out["mod_z"].abs() > threshold
    cond = res.extras.get("unit_cond")
    if cond is not None:
        out["cond"] = out["unit"].map(cond)
        out["ill_conditioned"] = out["cond"] > cond_threshold
    return out


def flagged_units(influence: pd.DataFrame, parameters=None) -> list:
    """Units flagged for at least one of ``parameters`` (default: all)."""
    d = influence
    if parameters is not None:
        d = d[d["parameter"].isin(list(parameters))]
    return pd.unique(d.loc[d["flag"], "unit"]).tolist()
