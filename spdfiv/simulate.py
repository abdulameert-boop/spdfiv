"""Data-generating process for Monte Carlo experiments and examples.

    y_it = psi_i sum_j w_ij y_jt + rho_i y_i,t-1 + x_it' beta_i + alpha_i
           + gamma_i' f_t + eps_it,
    x_lit = mu_li + Gamma_li' f_t + v_lit,

with AR(1) factors and idiosyncratic covariate errors, loadings of the
covariates correlated with those of the error (so that the covariates are
endogenous with respect to the factor component), and heteroskedastic
errors. Slopes are homogeneous unless ``het > 0``; ``outlier_share`` units
receive a large shift in every slope except psi and rho; ``entry_share`` of
the units enter the panel late (staggered entry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def knn_weights(N: int, k: int = 6, seed=None) -> np.ndarray:
    """Row-standardized k-nearest-neighbour weights for random locations."""
    rng = np.random.default_rng(seed)
    loc = rng.uniform(size=(N, 2))
    d = np.sqrt(((loc[:, None, :] - loc[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    W = np.zeros((N, N))
    nn = np.argsort(d, axis=1)[:, :k]
    W[np.repeat(np.arange(N), k), nn.ravel()] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def simulate_panel(N: int = 100, T: int = 100, psi: float = 0.3, rho: float = 0.4,
                   beta=(1.0, 0.5), n_factors: int = 2, het: float = 0.0,
                   outlier_share: float = 0.0, outlier_shift: float = 3.0,
                   entry_share: float = 0.0, W: np.ndarray | None = None,
                   burn: int = 50, seed=None):
    """Simulate a spatial dynamic panel with interactive effects.

    Returns ``(data, W, truth)``: a long DataFrame with columns ``id``,
    ``time``, ``y``, ``x1``, ..., a labelled weights DataFrame, and a dict
    with the true parameters (``truth["theta_i"]`` holds unit slopes).
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    k, r = len(beta), n_factors
    if W is None:
        W = knn_weights(N, seed=rng.integers(1 << 31))
    TT = T + burn

    f = np.zeros((TT, r))
    for t in range(1, TT):
        f[t] = 0.5 * f[t - 1] + np.sqrt(0.75) * rng.standard_normal(r)

    Gam = rng.normal(0.5, 1.0, size=(N, r, k))             # x loadings
    gam = Gam.mean(axis=2) + rng.normal(0, 0.5, size=(N, r))  # y loadings (correlated)
    mu = rng.normal(0, 1, size=(N, k))
    alpha = rng.normal(0, 1, size=N)
    sig = np.sqrt(rng.uniform(0.5, 1.5, size=N))

    v = np.zeros((N, TT, k))
    for t in range(1, TT):
        v[:, t] = 0.5 * v[:, t - 1] + np.sqrt(0.75) * rng.standard_normal((N, k))
    X = mu[:, None, :] + np.einsum("tr,irk->itk", f, Gam) + v

    psi_i = np.full(N, psi); rho_i = np.full(N, rho)
    beta_i = np.tile(beta, (N, 1))
    if het > 0:
        psi_i = psi_i + rng.uniform(-het, het, N) * 0.5
        rho_i = rho_i + rng.uniform(-het, het, N) * 0.5
        beta_i = beta_i + rng.uniform(-het, het, (N, k)) * 2
    outliers = np.zeros(N, dtype=bool)
    if outlier_share > 0:
        n_out = int(round(outlier_share * N))
        outliers[rng.choice(N, n_out, replace=False)] = True
        beta_i[outliers] += outlier_shift

    eps = rng.standard_normal((N, TT)) * sig[:, None]
    u = (gam @ f.T) + eps

    # staggered entry: a late unit does not exist before it enters, so it
    # neither affects nor is affected by its neighbours in those periods
    present = np.ones((N, TT), dtype=bool)
    if entry_share > 0:
        late = rng.choice(N, int(round(entry_share * N)), replace=False)
        for i in late:
            present[i, : burn + rng.integers(1, T // 2)] = False

    Y = np.zeros((N, TT))
    A_full = np.eye(N) - psi_i[:, None] * W
    rs_full = W.sum(axis=1)
    for t in range(1, TT):
        rhs = rho_i * Y[:, t - 1] + (X[:, t] * beta_i).sum(1) + alpha + u[:, t]
        idx = present[:, t]
        if idx.all():
            Y[:, t] = np.linalg.solve(A_full, rhs)
            continue
        Wt = W[np.ix_(idx, idx)]
        rs = Wt.sum(axis=1)
        Wt = Wt * np.divide(rs_full[idx], rs, out=np.zeros_like(rs), where=rs > 0)[:, None]
        Y[:, t] = 0.0
        Y[idx, t] = np.linalg.solve(np.eye(idx.sum()) - psi_i[idx, None] * Wt, rhs[idx])

    Y, X = Y[:, burn:], X[:, burn:]
    ids = np.arange(1, N + 1)
    obs = present[:, burn:]

    rows = {"id": np.repeat(ids, T), "time": np.tile(np.arange(1, T + 1), N),
            "y": Y.ravel()}
    for j in range(k):
        rows[f"x{j + 1}"] = X[:, :, j].ravel()
    data = pd.DataFrame(rows)
    data = data[obs.ravel()].reset_index(drop=True)
    Wdf = pd.DataFrame(W, index=ids, columns=ids)
    theta_i = pd.DataFrame(np.column_stack([psi_i, rho_i, beta_i]), index=ids,
                           columns=["W.y", "L1.y"] + [f"x{j + 1}" for j in range(k)])
    truth = {"psi": psi, "rho": rho, "beta": beta, "theta_i": theta_i,
             "outliers": ids[outliers], "observed": obs}
    return data, Wdf, truth
