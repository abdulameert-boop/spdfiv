"""Principal-components estimation of common factors, with missing data.

For a balanced panel the factors are sqrt(T) times the leading eigenvectors
of the T x T matrix (NT)^{-1} sum_i Z_i Z_i', exactly as in Bai (2003) and
Cui, Sarafidis, and Yamagata (2023).  When observations are missing, each
element (s, t) of that matrix is averaged over the units observed in both
periods, the "all-purpose" covariance estimator studied by Xiong and Pelger
(2023).  For a balanced panel the two definitions coincide.
"""

from __future__ import annotations

import numpy as np


def second_moment(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """T x T second-moment matrix of ``Z`` (N, T, k) over available cells ``A``."""
    if Z.ndim == 2:
        Z = Z[:, :, None]
    N, T, _ = Z.shape
    Zm = np.where(A[:, :, None], np.nan_to_num(Z), 0.0)
    num = np.einsum("isk,itk->st", Zm, Zm, optimize=True)
    Af = A.astype(float)
    cnt = Af.T @ Af
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(cnt > 0, num / cnt, 0.0) / T
    return S


def eigenvalue_ratio(evals: np.ndarray, kmax: int, N: int, T: int,
                     zero_factors: bool = False) -> int:
    """Ahn and Horenstein (2013) eigenvalue-ratio estimate of the number of factors.

    The estimate maximizes mu_k / mu_{k+1} over k = 1, ..., kmax, as in
    xtivdfreg / spxtivdfreg. With ``zero_factors=True`` the mock eigenvalue
    mu_0 = sum(mu) / log(min(N, T)) of Ahn and Horenstein is prepended, so
    that zero factors can also be selected.
    """
    if kmax <= 0:
        return 0
    mu = np.clip(np.sort(evals)[::-1], 0.0, None)
    kmax = int(min(kmax, len(mu) - 1))
    if kmax <= 0 or mu[0] <= 0:
        return 0
    tiny = np.finfo(float).tiny
    if zero_factors:
        seq = np.concatenate([[mu.sum() / np.log(min(N, T))], mu[: kmax + 1]])
        return int(np.argmax(seq[:-1] / np.maximum(seq[1:], tiny)))
    if kmax == 1:
        return 1
    ratios = mu[:kmax] / np.maximum(mu[1:kmax + 1], tiny)
    # ties resolved towards the larger number of factors, as in xtivdfreg
    return int(np.flatnonzero(ratios == ratios.max())[-1] + 1)


def standardize_panel(Z: np.ndarray, A: np.ndarray, how: str = "period") -> np.ndarray:
    """Standardize each variable of ``Z`` (N, T, k) over the available cells.

    ``how="period"`` standardizes across units within each period (as the
    option ``std`` of xtivdfreg); ``how="pooled"`` over all cells.
    """
    Zs = np.array(Z, dtype=float)
    for j in range(Zs.shape[2]):
        v = np.where(A, Zs[:, :, j], np.nan)
        if how == "period":
            mu = np.nanmean(v, axis=0, keepdims=True)
            sd = np.nanstd(v, axis=0, ddof=1, keepdims=True)
        elif how == "pooled":
            mu, sd = np.nanmean(v), np.nanstd(v, ddof=1)
        else:
            raise ValueError("std must be False, True, 'period' or 'pooled'.")
        with np.errstate(invalid="ignore", divide="ignore"):
            Zs[:, :, j] = np.where(sd > 0, (v - mu) / sd, 0.0)
    return np.where(A[:, :, None], Zs, 0.0)


def factor_eigen(Z: np.ndarray, A: np.ndarray, standardize=False):
    """Eigenvalues (descending) and sqrt(T)-scaled eigenvectors of the
    T x T second-moment matrix of ``Z`` (N, T, k) on cells ``A``."""
    if Z.ndim == 2:
        Z = Z[:, :, None]
    T = Z.shape[1]
    if standardize:
        Z = standardize_panel(Z, A, "period" if standardize is True else standardize)
    S = second_moment(Z, A)
    evals, evecs = np.linalg.eigh(0.5 * (S + S.T))
    order = np.argsort(evals)[::-1]
    return evals[order], np.sqrt(T) * evecs[:, order]


def estimate_factors(Z: np.ndarray, A: np.ndarray, kmax: int = 4,
                     eigratio: bool = True, standardize=False,
                     zero_factors: bool = False):
    """Estimate factors from ``Z`` (N, T, k) observed on cells ``A`` (N, T).

    Returns ``(F, r)`` with ``F`` of shape (T, r), normalised so F'F / T = I.
    """
    if Z.ndim == 2:
        Z = Z[:, :, None]
    N, T, _ = Z.shape
    if kmax <= 0:
        return np.zeros((T, 0)), 0
    evals, F = factor_eigen(Z, A, standardize)
    r = (eigenvalue_ratio(evals, kmax, N, T, zero_factors) if eigratio
         else int(min(kmax, T - 1)))
    return F[:, :r], r


def project_out(F: np.ndarray, rows: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Apply M_F = I - F(F'F)^+F' to ``V`` using only the factor rows ``rows``."""
    if F.shape[1] == 0:
        return V
    Fi = F[rows]
    coef, *_ = np.linalg.lstsq(Fi, V, rcond=None)
    return V - Fi @ coef
