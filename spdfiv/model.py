"""Defactored IV estimation of spatial dynamic panels with interactive effects.

Implements

* the first-stage and two-stage IV (2SIV) estimators of Cui, Sarafidis,
  and Yamagata (2023) for models with homogeneous slopes, and
* the mean-group IV (MGIV) estimator of Chen, Cui, Sarafidis, and
  Yamagata (2025) for models with heterogeneous slopes,

for the model

    y_it = psi_0 sum_j w_ij y_jt + sum_l rho_l y_i,t-l + sum_l psi_l sum_j w_ij y_j,t-l
           + x_it' beta + sum_j w_ij x_jt' delta + gamma_i' f_t + eps_it.

Missing observations are supported through an explicit option; see
``unbalanced`` in :func:`fit`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .factors import estimate_factors, project_out, factor_eigen, eigenvalue_ratio
from .panel import prepare_panel


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def spatial_lag(W: np.ndarray, V: np.ndarray, avail: np.ndarray,
                renormalize: bool = True) -> np.ndarray:
    """Spatial lag ``W_t v_t`` for every period, using available neighbours only.

    When some neighbours of unit i are unavailable in period t they are
    dropped. With ``renormalize=True`` the remaining weights of row i are
    rescaled so that the row keeps its original sum; for a balanced panel
    the operator is exactly ``W``.
    """
    squeeze = V.ndim == 2
    if squeeze:
        V = V[:, :, None]
    N, T, k = V.shape
    out = np.zeros((N, T, k))
    rs_full = W.sum(axis=1)
    for t in range(T):
        a = avail[:, t]
        vt = np.where(a[:, None], np.nan_to_num(V[:, t, :]), 0.0)
        if a.all():
            out[:, t, :] = W @ vt
            continue
        Wt = W * a[None, :]
        if renormalize:
            rs = Wt.sum(axis=1)
            scale = np.divide(rs_full, rs, out=np.zeros_like(rs), where=rs != 0)
            Wt = Wt * scale[:, None]
        out[:, t, :] = Wt @ vt
    return out[:, :, 0] if squeeze else out


def _gmm(A: np.ndarray, B: np.ndarray, c: np.ndarray):
    """theta = (A'B^{-1}A)^{-1} A'B^{-1} c and the bread (A'B^{-1}A)^{-1}."""
    Binv = np.linalg.pinv(B, hermitian=True)
    H = A.T @ Binv @ A
    Hinv = np.linalg.pinv(H, hermitian=True)
    theta = Hinv @ (A.T @ Binv @ c)
    return theta, Hinv, Binv


def _sandwich(A, Binv, Hinv, Omega):
    meat = A.T @ Binv @ Omega @ Binv @ A
    V = Hinv @ meat @ Hinv
    return 0.5 * (V + V.T)


def _demean(v: np.ndarray) -> np.ndarray:
    return v - v.mean(axis=0, keepdims=True)


# ----------------------------------------------------------------------------
# model specification and results containers
# ----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    depvar: str
    indepvars: list
    splag: bool
    tlags: int
    sptlags: int
    spx: list
    iv: list
    iv_lags: int
    iv_splags: bool
    factmax: int
    eigratio: bool
    std: object
    absorb: bool
    estimator: str
    doubledefact: bool
    unbalanced: str
    renormalize: bool
    zero_factors: bool = False
    common_rank: bool = True
    omega: str = "first"


@dataclass
class SpatialDFIVResults:
    params: pd.Series
    cov: pd.DataFrame
    spec: ModelSpec
    W: np.ndarray
    units: list
    nobs: int
    n_units: int
    obs_per_unit: np.ndarray
    n_instruments: int
    factors_x: list                 # number of factors for each instrument lag block
    factors_u: Optional[int]
    J: Optional[float] = None
    df_J: Optional[int] = None
    sigma_f: Optional[float] = None
    sigma_e: Optional[float] = None
    unit_params: Optional[pd.DataFrame] = None
    unit_se: Optional[pd.DataFrame] = None
    extras: dict = field(default_factory=dict)

    # -- inference --------------------------------------------------------
    @property
    def bse(self) -> pd.Series:
        return pd.Series(np.sqrt(np.clip(np.diag(self.cov.values), 0, None)),
                         index=self.params.index)

    @property
    def zvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2 * stats.norm.sf(np.abs(self.zvalues)), index=self.params.index)

    def conf_int(self, level: float = 0.95) -> pd.DataFrame:
        q = stats.norm.ppf(0.5 + level / 2)
        return pd.DataFrame({"lower": self.params - q * self.bse,
                             "upper": self.params + q * self.bse})

    @property
    def p_J(self) -> Optional[float]:
        if self.J is None or not self.df_J:
            return None
        return float(stats.chi2.sf(self.J, self.df_J))

    @property
    def rho_factor(self) -> Optional[float]:
        if self.sigma_f is None:
            return None
        return self.sigma_f ** 2 / (self.sigma_f ** 2 + self.sigma_e ** 2)

    # -- post-estimation, implemented in sibling modules --------------------
    def impacts(self, *args, **kwargs):
        from .impacts import impacts
        return impacts(self, *args, **kwargs)

    def mg_robust(self, *args, **kwargs):
        from .heterogeneity import mg_robust
        return mg_robust(self, *args, **kwargs)

    def mg_influence(self, *args, **kwargs):
        from .heterogeneity import mg_influence
        return mg_influence(self, *args, **kwargs)

    def summary(self, level: float = 0.95) -> str:
        from .summary import summary_text
        return summary_text(self, level)

    def __repr__(self):  # pragma: no cover
        return self.summary()


# ----------------------------------------------------------------------------
# main estimation routine
# ----------------------------------------------------------------------------

def fit(data: pd.DataFrame, y: str, x: Sequence[str], unit: str, time: str, W,
        *, W_ids=None, assume_ordered: bool = False,
        splag: bool = True, tlags: int = 0, sptlags: int = 0,
        spx: Optional[Sequence[str]] = None,
        iv: Optional[Sequence[str]] = None, iv_lags: int = 0,
        iv_splags: bool = True, factmax: int = 4, eigratio: bool = True,
        std=False, absorb: bool = True, estimator: str = "2siv",
        zero_factors: bool = False, common_rank: bool = True,
        omega: str = "first",
        doubledefact: Optional[bool] = None, unbalanced: str = "none",
        renormalize: bool = True) -> SpatialDFIVResults:
    """Estimate a spatial dynamic panel model with interactive effects.

    Parameters
    ----------
    data : DataFrame
        Long-format panel data.
    y, x : str, list of str
        Dependent variable and covariates. Covariates not listed in ``iv``
        are treated as endogenous with respect to the idiosyncratic error.
    unit, time : str
        Panel and time identifiers.
    W : DataFrame or array
        Spatial weights matrix, labelled by unit (see
        :func:`spdfiv.panel.align_weights`).
    splag : bool
        Include the spatial lag of ``y``.
    tlags, sptlags : int
        Number of time lags and spatial time lags of ``y``.
    spx : list of str
        Covariates whose spatial lags enter the model (spatial Durbin terms).
    iv : list of str
        Exogenous variables used to build defactored instruments. Defaults
        to ``x``.
    iv_lags : int
        Number of time lags of the instruments; factors are estimated
        separately for each lag order.
    iv_splags : bool
        Add spatial lags of the defactored instruments.
    factmax : int
        Maximum number of factors (``0`` gives standard IV estimation).
    eigratio : bool
        Choose the number of factors with the Ahn-Horenstein eigenvalue
        ratio. If ``False`` exactly ``factmax`` factors are used.
    std : bool or {"period", "pooled"}
        Standardize the variables from which factors are extracted. ``True``
        (or ``"period"``) standardizes across units within each period, as
        the option ``std`` of spxtivdfreg; ``"pooled"`` over all cells.
    zero_factors : bool
        Allow the eigenvalue-ratio criterion to select zero factors (mock
        eigenvalue of Ahn and Horenstein 2013). ``False`` (default) follows
        spxtivdfreg, which selects between 1 and ``factmax`` factors.
    omega : {"first", "second"}
        Residuals used for the weight matrix Omega in the variance and the J
        statistic of the 2SIV estimator: ``"first"`` (default) as in the
        equations of Cui et al. (2023) and in spxtivdfreg; ``"second"``
        re-evaluates Omega at the second-stage residuals.
    common_rank : bool
        Use, for all lag blocks of the instruments, the largest number of
        factors selected for any block (as spxtivdfreg). If ``False`` each
        block keeps its own number.
    absorb : bool
        Remove unit fixed effects by within transformation.
    estimator : {"2siv", "fstage", "mg"}
        Two-stage IV, first-stage IV, or mean-group IV.
    doubledefact : bool, optional
        Additionally project the contemporaneous instrument factors out of
        the lagged instruments. Default: ``True`` for ``"mg"``, else ``False``.
    unbalanced : {"none", "staggered", "sporadic"}
        ``"none"`` requires a balanced panel. ``"staggered"`` allows units to
        enter or leave the panel (contiguous observation spells).
        ``"sporadic"`` additionally allows interior gaps. Both are
        experimental: asymptotic theory for these cases is not available.
    renormalize : bool
        With missing neighbours, rescale the remaining weights of each row
        to keep its original row sum.
    """
    x = list(x)
    spx = list(spx or [])
    iv = list(iv) if iv is not None else list(x)
    estimator = estimator.lower()
    if omega not in ("first", "second"):
        raise ValueError("omega must be 'first' or 'second'.")
    if estimator not in ("2siv", "fstage", "mg"):
        raise ValueError("estimator must be '2siv', 'fstage' or 'mg'.")
    if unbalanced not in ("none", "staggered", "sporadic"):
        raise ValueError("unbalanced must be 'none', 'staggered' or 'sporadic'.")
    if doubledefact is None:
        doubledefact = estimator == "mg"
    for v in spx:
        if v not in x:
            raise ValueError(f"spx variable '{v}' must also be listed in x.")

    spec = ModelSpec(y, x, splag, tlags, sptlags, spx, iv, iv_lags, iv_splags,
                     factmax, eigratio, std, absorb, estimator, doubledefact,
                     unbalanced, renormalize, zero_factors, common_rank, omega)

    variables = list(dict.fromkeys([y] + x + iv))
    P = prepare_panel(data, variables, unit, time, W, W_ids, assume_ordered)
    Wm, mask = P.W, P.mask
    N, T0 = mask.shape

    if not mask.all():
        if unbalanced == "none":
            raise ValueError(
                f"The panel is unbalanced ({(~mask).sum()} missing unit-period cells). "
                "Use unbalanced='staggered' or 'sporadic' (experimental).")
        if unbalanced == "staggered":
            for i in range(N):
                obs = np.flatnonzero(mask[i])
                if len(obs) and (obs[-1] - obs[0] + 1) != len(obs):
                    raise ValueError(
                        f"Unit {P.units[i]!r} has interior gaps; use unbalanced='sporadic'.")
        warnings.warn("Estimation with an unbalanced panel is experimental: "
                      "the asymptotic theory of Cui et al. (2023) and Chen et al. "
                      "(2025) assumes a balanced panel.", stacklevel=2)

    p = max(tlags, sptlags, iv_lags)
    S = np.arange(p, T0)
    Ts = len(S)
    if Ts < 3:
        raise ValueError("Too few time periods after accounting for lags.")

    Y = P.arrays[y]
    WY = spatial_lag(Wm, Y, mask, renormalize)

    # ---- regressors ------------------------------------------------------
    cols, names = [], []
    if splag:
        cols.append(WY[:, S]); names.append(f"W.{y}")
    for l in range(1, tlags + 1):
        cols.append(Y[:, S - l]); names.append(f"L{l}.{y}")
    for l in range(1, sptlags + 1):
        cols.append(WY[:, S - l]); names.append(f"L{l}.W.{y}")
    for v in x:
        cols.append(P.arrays[v][:, S]); names.append(v)
    for v in spx:
        WX = spatial_lag(Wm, P.arrays[v], mask, renormalize)
        cols.append(WX[:, S]); names.append(f"W.{v}")
    C = np.stack(cols, axis=2)
    g = C.shape[2]

    # ---- valid observations -----------------------------------------------
    valid = mask[:, S].copy()
    for l in range(1, p + 1):
        valid &= mask[:, S - l]

    # ---- defactored instruments -------------------------------------------
    Zraw = np.stack([P.arrays[v] for v in iv], axis=2)
    raw_blocks, eig_blocks, r_blocks = [], [], []
    for tau in range(iv_lags + 1):
        A = mask[:, S - tau]
        Zb = Zraw[:, S - tau, :].copy()
        if absorb:
            for i in range(N):
                if A[i].any():
                    Zb[i, A[i]] = _demean(Zb[i, A[i]])
        evals, Fe = factor_eigen(Zb, A, std)
        r = (0 if factmax <= 0 else
             eigenvalue_ratio(evals, factmax, N, Ts, zero_factors) if eigratio
             else int(min(factmax, Ts - 1)))
        raw_blocks.append((Zb, A)); eig_blocks.append(Fe); r_blocks.append(r)
    r_raw = list(r_blocks)
    if common_rank:
        # as in xtivdfreg: all lag blocks of one instrument set use the
        # largest number of factors selected for any of them
        r_blocks = [max(r_blocks)] * len(r_blocks)
    blocks, factors_x, F_blocks = [], [], []
    for (Zb, A), Fe, r in zip(raw_blocks, eig_blocks, r_blocks):
        F = Fe[:, :r]
        factors_x.append(r)
        F_blocks.append(F)
        D = np.full_like(Zb, np.nan)
        for i in range(N):
            if A[i].any():
                D[i, A[i]] = project_out(F, A[i], Zb[i, A[i]])
        blocks.append((D, A))
    if doubledefact:
        # second defactorization of all instrument blocks with the factors of
        # the contemporaneous instruments, their number chosen on their own
        # (as in xtivdfreg); for lag blocks this gives M_F0 M_Ftau Z_{-tau}
        F0 = eig_blocks[0][:, :r_raw[0]]
        for D, A in blocks:
            for i in range(N):
                if A[i].any():
                    D[i, A[i]] = project_out(F0, A[i], D[i, A[i]])
    zcols = []
    for D, A in blocks:
        zcols.append(np.nan_to_num(D))
        if iv_splags:
            zcols.append(spatial_lag(Wm, D, A, renormalize))
    Z = np.concatenate(zcols, axis=2)
    m = Z.shape[2]
    if m < g:
        raise ValueError(f"Model is underidentified: {m} instruments for {g} parameters.")

    # ---- stack by unit ----------------------------------------------------
    Ys = Y[:, S]
    yi, Ci, Zi, rows_i, keep_units = [], [], [], [], []
    for i in range(N):
        rws = valid[i]
        if rws.sum() <= (1 if absorb else 0):
            continue
        a, b, c = Ys[i, rws], C[i, rws], Z[i, rws]
        if absorb:
            # y and the regressors are demeaned; the instruments were demeaned
            # before defactoring and are not demeaned again (as spxtivdfreg).
            # Z'y and Z'C are unaffected, but Z'Z, the IV weight, is.
            a, b = _demean(a), _demean(b)
        yi.append(a); Ci.append(b); Zi.append(c); rows_i.append(rws); keep_units.append(i)
    n_i = np.array([len(a) for a in yi])
    n = int(n_i.sum())
    if len(keep_units) < N:
        warnings.warn(f"{N - len(keep_units)} units have no usable observations "
                      "and are dropped.", stacklevel=2)

    common = dict(spec=spec, W=Wm, units=P.units, nobs=n, n_units=len(keep_units),
                  obs_per_unit=n_i, n_instruments=m, factors_x=factors_x)

    if estimator == "mg":
        return _fit_mg(yi, Ci, Zi, names, keep_units, P.units, common)

    # ---- first stage ------------------------------------------------------
    A1 = sum(z.T @ c for z, c in zip(Zi, Ci))
    B1 = sum(z.T @ z for z in Zi)
    c1 = sum(z.T @ a for z, a in zip(Zi, yi))
    theta1, H1inv, B1inv = _gmm(A1, B1, c1)
    u1 = [a - c @ theta1 for a, c in zip(yi, Ci)]

    if estimator == "fstage":
        g1 = [z.T @ u for z, u in zip(Zi, u1)]
        Omega = sum(np.outer(v, v) for v in g1)
        V = _sandwich(A1, B1inv, H1inv, Omega)
        gbar = sum(g1)
        J = float(gbar @ np.linalg.pinv(Omega, hermitian=True) @ gbar)
        sig_e = np.sqrt(sum(u @ u for u in u1) / n)
        return SpatialDFIVResults(
            params=pd.Series(theta1, index=names),
            cov=pd.DataFrame(V, index=names, columns=names),
            factors_u=None, J=J, df_J=m - g, sigma_f=None, sigma_e=float(sig_e),
            **common)

    # ---- second stage -----------------------------------------------------
    U = np.full((len(keep_units), Ts), np.nan)
    Vmask = np.zeros((len(keep_units), Ts), dtype=bool)
    for k_, (rws, u) in enumerate(zip(rows_i, u1)):
        U[k_, rws] = u
        Vmask[k_, rws] = True
    Fy, ry = estimate_factors(U, Vmask, factmax, eigratio, False, zero_factors)

    def M(k_, v):
        return project_out(Fy, rows_i[k_], v)

    MZ = [M(k_, z) for k_, z in enumerate(Zi)]
    A2 = sum(mz.T @ c for mz, c in zip(MZ, Ci))
    gu1 = [mz.T @ u for mz, u in zip(MZ, u1)]
    B2 = sum(np.outer(v, v) for v in gu1)
    c2 = sum(mz.T @ a for mz, a in zip(MZ, yi))
    theta2, H2inv, B2inv = _gmm(A2, B2, c2)

    u2 = [a - c @ theta2 for a, c in zip(yi, Ci)]
    e2 = [M(k_, u) for k_, u in enumerate(u2)]
    g2 = [mz.T @ e for mz, e in zip(MZ, e2)]
    gbar = sum(g2)
    if omega == "first":
        # Equations (10)-(12) of Kripfganz and Sarafidis (2025) and
        # spxtivdfreg: Omega from first-stage residuals, so Omega = B2 and
        # the variance reduces to (A2' B2^{-1} A2)^{-1}
        V = 0.5 * (H2inv + H2inv.T)
        J = float(gbar @ B2inv @ gbar)
    else:
        Omega = sum(np.outer(v, v) for v in g2)
        V = _sandwich(A2, B2inv, H2inv, Omega)
        J = float(gbar @ np.linalg.pinv(Omega, hermitian=True) @ gbar)

    s2u = sum(u @ u for u in u2) / n
    s2e = sum(e @ e for e in e2) / n
    s2f = max(s2u - s2e, 0.0)
    res = SpatialDFIVResults(
        params=pd.Series(theta2, index=names),
        cov=pd.DataFrame(V, index=names, columns=names),
        factors_u=ry, J=J, df_J=m - g, sigma_f=float(np.sqrt(s2f)),
        sigma_e=float(np.sqrt(s2e)), **common)
    res.extras["first_stage_params"] = pd.Series(theta1, index=names)
    return res


def _fit_mg(yi, Ci, Zi, names, keep_units, units, common):
    g = len(names)
    thetas, ses, ok, conds = [], [], [], []
    for a, c, z in zip(yi, Ci, Zi):
        if len(a) <= z.shape[1]:
            thetas.append(np.full(g, np.nan)); ses.append(np.full(g, np.nan))
            ok.append(False); conds.append(np.inf)
            continue
        conds.append(float(np.linalg.cond(z.T @ z)))
        A = z.T @ c
        th, Hinv, Binv = _gmm(A, z.T @ z, z.T @ a)
        u = a - c @ th
        zu = z * u[:, None]
        V = _sandwich(A, Binv, Hinv, zu.T @ zu)
        thetas.append(th); ses.append(np.sqrt(np.clip(np.diag(V), 0, None)))
        ok.append(True)
    ok = np.array(ok)
    if not ok.all():
        warnings.warn(f"{(~ok).sum()} units have fewer observations than instruments "
                      "and are excluded from the mean-group estimate.", stacklevel=3)
    lab = [units[i] for i in keep_units]
    Th = pd.DataFrame(np.array(thetas), index=lab, columns=names)
    Se = pd.DataFrame(np.array(ses), index=lab, columns=names)
    T_ok = Th.values[ok]
    Ng = T_ok.shape[0]
    mg = T_ok.mean(axis=0)
    dev = T_ok - mg
    Sigma = dev.T @ dev / (Ng - 1)
    V = Sigma / Ng
    common = dict(common)
    common["n_units"] = Ng
    res = SpatialDFIVResults(
        params=pd.Series(mg, index=names),
        cov=pd.DataFrame(V, index=names, columns=names),
        factors_u=None, unit_params=Th[ok], unit_se=Se[ok], **common)
    # condition number of each unit's Z_i'Z_i: very large values signal
    # (nearly) collinear instruments and fragile unit-specific estimates
    res.extras["unit_cond"] = pd.Series(np.array(conds), index=lab)[ok]
    return res
