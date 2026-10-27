"""Validation tests for spdfiv.

The central test re-implements the 2SIV estimator of Cui, Sarafidis, and
Yamagata (2023) directly from the equations of Kripfganz and Sarafidis
(2025, Section 2.1) using explicit T x T projection matrices, and checks
that the package reproduces it to machine precision.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

import spdfiv as sd
from spdfiv.factors import eigenvalue_ratio


def _data(N=40, T=30, seed=0, **kw):
    return sd.simulate_panel(N=N, T=T, seed=seed, **kw)


def textbook_2siv(data, W, rx, ry, absorb=True, omega="first"):
    """Equations (5)-(12) of Kripfganz and Sarafidis (2025), written literally."""
    ids = list(W.index)
    Wm = W.to_numpy()
    d = data.set_index(["id", "time"]).sort_index()
    T0 = d.index.get_level_values(1).nunique()
    Y = np.array([d.loc[i, "y"].to_numpy() for i in ids])          # N x T0
    X = np.stack([np.array([d.loc[i, v].to_numpy() for i in ids]) for v in ["x1", "x2"]], 2)
    N = len(ids)
    s = np.arange(1, T0)
    T = len(s)
    dm = (lambda a: a - a.mean(0)) if absorb else (lambda a: a)
    Xi = [dm(X[i, s]) for i in range(N)]
    Xl = [dm(X[i, s - 1]) for i in range(N)]

    def F_of(mats, r):
        S = sum(m @ m.T for m in mats) / (N * T)
        ev, V = np.linalg.eigh(S)
        V = V[:, np.argsort(ev)[::-1]]
        return np.sqrt(T) * V[:, :r]

    def Mof(F):
        return np.eye(T) - F @ np.linalg.inv(F.T @ F) @ F.T

    M0, M1 = Mof(F_of(Xi, rx)), Mof(F_of(Xl, rx))
    D0 = [M0 @ a for a in Xi]
    D1 = [M1 @ a for a in Xl]
    Zi = []
    for i in range(N):
        WD0 = sum(Wm[i, j] * D0[j] for j in range(N))
        WD1 = sum(Wm[i, j] * D1[j] for j in range(N))
        Zi.append(np.hstack([D0[i], WD0, D1[i], WD1]))
    WY = Wm @ Y
    Ci = [dm(np.column_stack([WY[i, s], Y[i, s - 1], X[i, s]])) for i in range(N)]
    yi = [dm(Y[i, s]) for i in range(N)]

    A = sum(z.T @ c for z, c in zip(Zi, Ci))
    B = sum(z.T @ z for z in Zi)
    c = sum(z.T @ y for z, y in zip(Zi, yi))
    Bi = np.linalg.inv(B)
    th1 = np.linalg.solve(A.T @ Bi @ A, A.T @ Bi @ c)
    u1 = [y - C @ th1 for y, C in zip(yi, Ci)]
    My = Mof(F_of([u[:, None] for u in u1], ry))
    At = sum(z.T @ My @ C for z, C in zip(Zi, Ci))
    Bt = sum(z.T @ My @ np.outer(u, u) @ My @ z for z, u in zip(Zi, u1))
    ct = sum(z.T @ My @ y for z, y in zip(Zi, yi))
    Bti = np.linalg.inv(Bt)
    H = np.linalg.inv(At.T @ Bti @ At)
    th2 = H @ At.T @ Bti @ ct
    u2 = [My @ (y - C @ th2) for y, C in zip(yi, Ci)]
    if omega == "first":   # equation (11): first-stage residuals, Omega = B~
        Om = Bt
    else:
        Om = sum(z.T @ My @ np.outer(u, u) @ My @ z for z, u in zip(Zi, u2))
    V = H @ At.T @ Bti @ Om @ Bti @ At @ H
    gbar = sum(z.T @ My @ u for z, u in zip(Zi, u2))
    J = gbar @ np.linalg.inv(Om) @ gbar
    return th1, th2, V, J


class TestTextbookEquivalence(unittest.TestCase):
    def test_2siv_matches_equations(self):
        data, W, _ = _data()
        for omega in ("first", "second"):
            res = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
                         factmax=2, eigratio=False, omega=omega)
            th1, th2, V, J = textbook_2siv(data, W, 2, 2, omega=omega)
            np.testing.assert_allclose(res.extras["first_stage_params"].values, th1, atol=1e-10)
            np.testing.assert_allclose(res.params.values, th2, atol=1e-10)
            np.testing.assert_allclose(res.cov.values, V, atol=1e-12, rtol=1e-8)
            self.assertAlmostEqual(res.J, J, places=8)
            self.assertEqual(res.df_J, 8 - 4)

    def test_no_factors_first_stage_is_2sls(self):
        data, W, _ = _data(seed=2)
        res = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=0, iv_lags=0,
                     factmax=0, estimator="fstage", absorb=False)
        P = sd.prepare_panel(data, ["y", "x1", "x2"], "id", "time", W)
        Wm = P.W
        Y = P.arrays["y"]; X1 = P.arrays["x1"]; X2 = P.arrays["x2"]
        C = np.column_stack([(Wm @ Y).T.ravel(), X1.T.ravel(), X2.T.ravel()])
        Z = np.column_stack([X1.T.ravel(), X2.T.ravel(), (Wm @ X1).T.ravel(), (Wm @ X2).T.ravel()])
        y = Y.T.ravel()
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
        th = np.linalg.solve(C.T @ Pz @ C, C.T @ Pz @ y)
        np.testing.assert_allclose(res.params.values, th, atol=1e-10)


class TestWeightsAlignment(unittest.TestCase):
    def test_permuted_labelled_W_gives_same_estimates(self):
        data, W, _ = _data(seed=3)
        r1 = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1)
        perm = np.random.default_rng(0).permutation(W.index)
        r2 = sd.fit(data, "y", ["x1", "x2"], "id", "time", W.loc[perm, perm],
                    tlags=1, iv_lags=1)
        np.testing.assert_allclose(r1.params.values, r2.params.values, atol=1e-10)

    def test_unlabelled_array_rejected(self):
        data, W, _ = _data(seed=3)
        with self.assertRaises(sd.WeightsAlignmentError):
            sd.fit(data, "y", ["x1", "x2"], "id", "time", W.to_numpy())

    def test_mismatched_units_rejected(self):
        data, W, _ = _data(seed=3)
        with self.assertRaises(sd.WeightsAlignmentError):
            sd.fit(data[data["id"] != 1], "y", ["x1", "x2"], "id", "time", W)

    def test_nonzero_diagonal_rejected(self):
        data, W, _ = _data(seed=3)
        W2 = W.copy(); W2.iloc[0, 0] = 0.1
        with self.assertRaises(sd.WeightsAlignmentError):
            sd.fit(data, "y", ["x1", "x2"], "id", "time", W2)


class TestImpacts(unittest.TestCase):
    def setUp(self):
        data, W, _ = _data(N=60, T=40, seed=4)
        self.res = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1)

    def test_total_closed_form_row_standardized(self):
        p = self.res.params
        for kind, denom in [("sr", 1 - p["W.y"]), ("lr", 1 - p["W.y"] - p["L1.y"])]:
            imp = self.res.impacts(kind)
            for v in ["x1", "x2"]:
                self.assertAlmostEqual(imp.loc[("total", v), "impact"], p[v] / denom, places=10)

    def test_schur_path_equals_dense_solve(self):
        from spdfiv.impacts import _Spectral
        rng = np.random.default_rng(1)
        W_rs = self.res.W                                   # row-standardized kNN
        W_bin = (W_rs > 0).astype(float)                    # binary, unequal row sums
        W_bin[rng.integers(0, 60, 10), rng.integers(0, 60, 10)] = 1.0
        np.fill_diagonal(W_bin, 0.0)
        W_sp = W_bin / np.max(np.abs(np.linalg.eigvals(W_bin)))  # spectral normalization
        for Wm in (W_rs, W_sp):
            sp = _Spectral(Wm)
            a = np.array([1.0, 0.4]); b = np.array([0.3, 0.2])
            d1, t1 = sp.direct_total(a, b, 1.2, 0.5)
            sp.method = "dense"
            d2, t2 = sp.direct_total(a, b, 1.2, 0.5)
            np.testing.assert_allclose(d1, d2, atol=1e-9)
            np.testing.assert_allclose(t1, t2, atol=1e-9)

    def test_simulation_close_to_delta(self):
        a = self.res.impacts("lr")
        b = self.res.impacts("lr", method="simulation", draws=4000, seed=1)
        np.testing.assert_allclose(a["impact"], b["impact"])
        np.testing.assert_allclose(a["se"], b["se"], rtol=0.15)

    def test_instability_raises(self):
        res = self.res
        old = res.params.copy()
        res.params["L1.y"] = 0.9
        with self.assertRaises(sd.StabilityError):
            res.impacts("lr")
        res.params[:] = old


class TestUnbalanced(unittest.TestCase):
    def test_balanced_data_unchanged_by_unbalanced_option(self):
        data, W, _ = _data(seed=5)
        r1 = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1)
        r2 = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
                    unbalanced="sporadic")
        np.testing.assert_allclose(r1.params.values, r2.params.values, atol=1e-12)

    def test_unbalanced_requires_option(self):
        data, W, _ = _data(seed=5, entry_share=0.2)
        with self.assertRaises(ValueError):
            sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
                       unbalanced="staggered")
        self.assertEqual(r.nobs, int(len(data) - 40))  # one lag lost per unit

    def test_gaps_rejected_for_staggered(self):
        data, W, _ = _data(seed=5)
        data = data[~((data["id"] == 3) & (data["time"] == 10))]
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sd.fit(data, "y", ["x1", "x2"], "id", "time", W, unbalanced="staggered")


class TestMeanGroup(unittest.TestCase):
    def test_mg_is_mean_of_unit_estimates_and_robust(self):
        data, W, truth = _data(N=80, T=50, seed=6, het=0.2, outlier_share=0.05)
        r = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
                   estimator="mg")
        np.testing.assert_allclose(r.params.values, r.unit_params.mean().values)
        inf = sd.mg_influence(r)
        dfmg = inf[inf.parameter == "x1"]["dfmg"].to_numpy()
        th = r.unit_params["x1"].to_numpy()
        loo = np.array([np.delete(th, i).mean() for i in range(len(th))])
        np.testing.assert_allclose(dfmg * r.bse["x1"], th.mean() - loo, atol=1e-12)
        flagged = set(sd.flagged_units(inf, ["x1", "x2"]))
        self.assertTrue(set(truth["outliers"]).issubset(flagged))


class TestFactorSelection(unittest.TestCase):
    def test_eigenvalue_ratio(self):
        ev = np.array([50.0, 20.0, 1.0, 0.9, 0.8, 0.7])
        self.assertEqual(eigenvalue_ratio(ev, 4, 100, 100), 2)
        self.assertEqual(eigenvalue_ratio(ev, 4, 100, 100, zero_factors=True), 2)
        # without the mock eigenvalue at least one factor is selected (xtivdfreg)
        self.assertGreaterEqual(eigenvalue_ratio(np.ones(6), 4, 100, 100), 1)
        self.assertEqual(eigenvalue_ratio(np.ones(6), 4, 100, 100, zero_factors=True), 0)


if __name__ == "__main__":
    unittest.main()
