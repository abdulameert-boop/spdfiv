"""Average direct, indirect and total impacts (LeSage and Pace 2009).

For covariate l the matrix of partial derivatives is

    S_l = [a I - b W]^{-1} (beta_l I + delta_l W),

with a = 1, b = psi_0 in the short run and a = 1 - sum(rho), b = sum(psi) in
the long run. The average direct impact is tr(S_l)/N, the average total impact
is 1'S_l 1 / N, and the indirect impact is their difference.

Two inference methods are offered:

* ``"delta"`` - the Delta method, as in Kripfganz and Sarafidis (2025);
* ``"simulation"`` - Krinsky and Robb (1986) simulation from the
  asymptotic normal distribution of the estimator, the approach used for
  impacts by LeSage and Pace (2009). It captures the non-linearity of the
  long-run multipliers, which the Delta method linearizes away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class StabilityError(ValueError):
    pass


class _Spectral:
    """Complex Schur decomposition W = Q R Q*, reused across evaluations.

    The trace of a rational function of W equals the sum of that function
    over the eigenvalues (the diagonal of R), whether or not W is
    diagonalizable, so average direct impacts cost O(N) per evaluation.
    Average total impacts use 1'S1 = (1'Q)(aI - bR)^{-1} Q*(beta 1 + delta W1),
    one triangular solve, O(N^2); when all row sums of W are equal (e.g. row-standardized
    W) the closed form (beta + delta s) / (a - b s) is used instead.
    """

    def __init__(self, W: np.ndarray, method: str = "schur"):
        from scipy.linalg import schur
        self.N = W.shape[0]
        self.W = W
        self.method = method
        R, Q = schur(W.astype(complex), output="complex")
        self.R, self.Q = R, Q
        self.lam = np.diag(R)
        self.omega = float(np.max(np.abs(self.lam)))
        one = np.ones(self.N)
        rs = W.sum(axis=1)
        self.const_rows = bool(np.ptp(rs) < 1e-12 * max(1.0, np.abs(rs).max()))
        self.s = float(rs[0])
        self.q1 = Q.conj().T @ one          # Q* 1
        self.qW1 = Q.conj().T @ (W @ one)   # Q* W 1
        self.p1 = one @ Q                   # 1' Q

    def direct_total(self, a, b, beta, delta):
        """Vectorised over leading dimensions of a, b, beta, delta."""
        a, b, beta, delta = np.broadcast_arrays(*(np.asarray(v, dtype=float)
                                                  for v in (a, b, beta, delta)))
        if self.method == "dense":
            return self._dense(a, b, beta, delta)
        lam = self.lam
        direct = ((beta[..., None] + delta[..., None] * lam)
                  / (a[..., None] - b[..., None] * lam)).mean(axis=-1).real
        if self.const_rows:
            total = (beta + delta * self.s) / (a - b * self.s)
            return direct, total
        from scipy.linalg import solve_triangular
        total = np.empty(a.shape)
        I = np.eye(self.N)
        for idx in np.ndindex(a.shape):
            Rm = a[idx] * I - b[idx] * self.R
            rhs = beta[idx] * self.q1 + delta[idx] * self.qW1
            total[idx] = (self.p1 @ solve_triangular(Rm, rhs)).real / self.N
        return direct, total

    def _dense(self, a, b, beta, delta):
        d = np.empty(a.shape); t = np.empty(a.shape)
        I = np.eye(self.N)
        for idx in np.ndindex(a.shape):
            Smat = np.linalg.solve(a[idx] * I - b[idx] * self.W,
                                   beta[idx] * I + delta[idx] * self.W)
            d[idx] = np.trace(Smat) / self.N
            t[idx] = Smat.sum() / self.N
        return d, t


def _components(res, theta):
    """Map a (possibly 2-D, draws x params) parameter array to model pieces."""
    spec = res.spec
    names = list(res.params.index)
    pos = {nm: j for j, nm in enumerate(names)}
    theta = np.atleast_2d(theta)
    zero = np.zeros(theta.shape[0])
    psi0 = theta[:, pos[f"W.{spec.depvar}"]] if spec.splag else zero
    rho = sum((theta[:, pos[f"L{l}.{spec.depvar}"]] for l in range(1, spec.tlags + 1)), zero)
    psil = sum((theta[:, pos[f"L{l}.W.{spec.depvar}"]] for l in range(1, spec.sptlags + 1)), zero)
    return psi0, rho, psil, pos


def _evaluate(res, theta, variables, kind, spectral):
    spec = res.spec
    psi0, rho, psil, pos = _components(res, theta)
    if kind == "sr":
        a, b = np.ones_like(psi0), psi0
    else:
        a, b = 1.0 - rho, psi0 + psil
    th = np.atleast_2d(theta)
    out_d, out_t = [], []
    for v in variables:
        beta = th[:, pos[v]]
        delta = th[:, pos[f"W.{v}"]] if v in spec.spx else np.zeros_like(beta)
        d, t = spectral.direct_total(a, b, beta, delta)
        out_d.append(d); out_t.append(t)
    D = np.stack(out_d, axis=1); Tt = np.stack(out_t, axis=1)
    return np.concatenate([D, Tt - D, Tt], axis=1)   # draws x (3*k)


def _stable(res, theta, kind, omega):
    psi0, rho, psil, _ = _components(res, theta)
    ok = psi0 * omega < 1
    if kind == "lr":
        ok &= (1.0 - rho) - (psi0 + psil) * omega > 0
    return ok


def impacts(res, kind: str = "sr", method: str = "delta", variables=None,
            draws: int = 2000, seed=None, level: float = 0.95,
            force: bool = False) -> pd.DataFrame:
    """Average direct, indirect and total impacts with standard errors.

    Parameters
    ----------
    kind : {"sr", "lr"}
        Short-run or long-run impacts.
    method : {"delta", "simulation"}
        Delta method or Krinsky-Robb simulation.
    variables : list of str, optional
        Covariates; defaults to all covariates in the model.
    draws : int
        Number of simulation draws (``method="simulation"``).
    force : bool
        Report impacts even if the point estimates violate stability.
    """
    kind = kind.lower()
    if kind not in ("sr", "lr"):
        raise ValueError("kind must be 'sr' or 'lr'.")
    if method not in ("delta", "simulation"):
        raise ValueError("method must be 'delta' or 'simulation'.")
    variables = list(variables) if variables is not None else list(res.spec.indepvars)
    spectral = _Spectral(res.W)
    theta = res.params.values
    Sigma = res.cov.values
    if not _stable(res, theta, kind, spectral.omega)[0] and not force:
        raise StabilityError(
            "The estimated coefficients violate the dynamic stability condition; "
            "impacts are not defined (use force=True to override).")
    point = _evaluate(res, theta, variables, kind, spectral)[0]
    labels = pd.MultiIndex.from_product([["direct", "indirect", "total"], variables],
                                        names=["effect", "variable"])
    q = stats.norm.ppf(0.5 + level / 2)

    if method == "delta":
        k = len(theta)
        J = np.empty((len(point), k))
        for j in range(k):
            h = 1e-6 * max(1.0, abs(theta[j]))
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h; tm[j] -= h
            J[:, j] = (_evaluate(res, tp, variables, kind, spectral)[0]
                       - _evaluate(res, tm, variables, kind, spectral)[0]) / (2 * h)
        se = np.sqrt(np.clip(np.diag(J @ Sigma @ J.T), 0, None))
        out = pd.DataFrame({"impact": point, "se": se}, index=labels)
        out["z"] = out["impact"] / out["se"]
        out["p"] = 2 * stats.norm.sf(np.abs(out["z"]))
        out["lower"] = point - q * se
        out["upper"] = point + q * se
        out.attrs.update(kind=kind, method=method)
        return out

    rng = np.random.default_rng(seed)
    evals, evecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    L = evecs * np.sqrt(np.clip(evals, 0, None))
    sims = theta + rng.standard_normal((draws, len(theta))) @ L.T
    keep = _stable(res, sims, kind, spectral.omega)
    vals = _evaluate(res, sims[keep], variables, kind, spectral)
    alpha = (1 - level) / 2
    out = pd.DataFrame({
        "impact": point,
        "se": vals.std(axis=0, ddof=1),
        "lower": np.quantile(vals, alpha, axis=0),
        "upper": np.quantile(vals, 1 - alpha, axis=0),
    }, index=labels)
    out["z"] = out["impact"] / out["se"]
    out["p"] = 2 * stats.norm.sf(np.abs(out["z"]))
    out = out[["impact", "se", "z", "p", "lower", "upper"]]
    out.attrs.update(kind=kind, method=method, draws=draws,
                     share_stable=float(keep.mean()))
    return out
