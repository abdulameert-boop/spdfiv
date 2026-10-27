"""Monte Carlo experiments reported in Section 5 of the paper.

Usage:  python replication/monte_carlo.py [--reps 1000] [--workers 2]

Writes CSV summaries to replication/results/.
"""

import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import spdfiv as sd
from spdfiv.impacts import _Spectral

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
X = ["x1", "x2"]
KW = dict(tlags=1, iv_lags=1)


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


# ---------------------------------------------------------------- MC1 -----
def mc1_rep(args):
    N, T, seed = args
    data, W, _ = sd.simulate_panel(N=N, T=T, seed=seed)
    out = []
    for lab, opts in [("2SIV", {}), ("First-stage IV", {"estimator": "fstage"}),
                      ("IV, no factors", {"factmax": 0})]:
        r = _quiet(sd.fit, data, "y", X, "id", "time", W, **KW, **opts)
        rec = {"N": N, "T": T, "seed": seed, "estimator": lab, "J_p": r.p_J,
               "rx_ok": all(f == 2 for f in r.factors_x),
               "ry_ok": (r.factors_u == 2) if r.factors_u is not None else np.nan}
        for nm in ["W.y", "L1.y", "x1"]:
            rec[f"est_{nm}"] = r.params[nm]
            rec[f"se_{nm}"] = r.bse[nm]
        out.append(rec)
    return out


def mc1(reps, pool):
    truth = {"W.y": 0.3, "L1.y": 0.4, "x1": 1.0}
    jobs = [(N, T, 10_000 * k + s) for k, (N, T) in enumerate([(50, 50), (100, 100), (200, 200)])
            for s in range(reps)]
    df = pd.DataFrame([r for rr in pool.map(mc1_rep, jobs, chunksize=10) for r in rr])
    rows = []
    for (N, T, est), g in df.groupby(["N", "T", "estimator"], sort=False):
        rec = {"N": N, "T": T, "estimator": est}
        for nm, th in truth.items():
            e = g[f"est_{nm}"] - th
            rec[f"bias_{nm}"] = e.mean()
            rec[f"rmse_{nm}"] = np.sqrt((e ** 2).mean())
            rec[f"size_{nm}"] = (np.abs(e / g[f"se_{nm}"]) > 1.96).mean()
        rec["J_rej"] = (g["J_p"] < 0.05).mean()
        rec["rx_correct"] = g["rx_ok"].mean()
        rec["ry_correct"] = g["ry_ok"].mean()
        rows.append(rec)
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------- MC2 -----
PSI2, RHO2, B2 = 0.35, 0.5, 1.0


def mc2_rep(args):
    N, T, seed = args
    data, W, _ = sd.simulate_panel(N=N, T=T, psi=PSI2, rho=RHO2, beta=(B2, 0.5), seed=seed)
    r = _quiet(sd.fit, data, "y", X, "id", "time", W, **KW)
    sp = _Spectral(W.to_numpy())
    d, t = sp.direct_total(1 - RHO2, PSI2, B2, 0.0)
    true = {"direct": float(d), "indirect": float(t - d), "total": float(t)}
    out = []
    for method in ["delta", "simulation"]:
        try:
            imp = r.impacts("lr", method=method, variables=["x1"], draws=1000, seed=seed)
        except sd.StabilityError:
            continue
        for eff in ["direct", "indirect", "total"]:
            row = imp.loc[(eff, "x1")]
            out.append({"N": N, "T": T, "seed": seed, "method": method, "effect": eff,
                        "est": row["impact"], "true": true[eff],
                        "cover": row["lower"] <= true[eff] <= row["upper"],
                        "lower_miss": true[eff] < row["lower"],
                        "upper_miss": true[eff] > row["upper"],
                        "width": row["upper"] - row["lower"]})
    return out


def mc2_summarize(df):
    summ = (df.groupby(["N", "T", "effect", "method"], sort=False)
              .agg(coverage=("cover", "mean"), miss_below=("lower_miss", "mean"),
                   miss_above=("upper_miss", "mean"), width=("width", "median"),
                   n=("cover", "size")).reset_index())
    bias = df.assign(err=df["est"] - df["true"]).groupby(
        ["N", "T", "effect", "method"], sort=False)["err"].median()
    summ["median_bias"] = bias.values
    return summ


def mc2(reps, pool):
    jobs = [(N, T, 20_000 + 10_000 * k + s) for k, (N, T) in enumerate([(30, 30), (50, 50), (100, 100)])
            for s in range(reps)]
    df = pd.DataFrame([r for rr in pool.map(mc2_rep, jobs, chunksize=10) for r in rr])
    return df, mc2_summarize(df)


# ---------------------------------------------------------------- MC3 -----
def mc3_rep(args):
    N, T, share, seed = args[:4]
    shift = args[4] if len(args) > 4 else 3.0
    data, W, truth = sd.simulate_panel(N=N, T=T, het=0.2, outlier_share=share,
                                       outlier_shift=shift, seed=seed)
    r = _quiet(sd.fit, data, "y", X, "id", "time", W, estimator="mg", **KW)
    inf = sd.mg_influence(r)
    flagged = sd.flagged_units(inf, ["x1", "x2"])
    out_ids = set(truth["outliers"].tolist())
    clean = [u for u in r.unit_params.index if u not in out_ids]
    tpr = (len(out_ids & set(flagged)) / len(out_ids)) if out_ids else np.nan
    fpr = len(set(flagged) - out_ids) / len(clean)
    res = []
    est = {
        "MG": (r.params["x1"], r.params["x1"] - 1.96 * r.bse["x1"], r.params["x1"] + 1.96 * r.bse["x1"]),
    }
    for m in ["trimmed", "median"]:
        t = sd.mg_robust(r, method=m, trim=0.1, bootstrap=499, seed=seed).loc["x1"]
        est[f"{m}"] = (t[m], t["lower"], t["upper"])
    ex = sd.mg_robust(r, method="mean", exclude=flagged, bootstrap=1, seed=seed).loc["x1"]
    est["MG excl. flagged"] = (ex["mean"], ex["mean"] - 1.96 * ex["MG_se"], ex["mean"] + 1.96 * ex["MG_se"])
    for k, (e, lo, hi) in est.items():
        res.append({"N": N, "T": T, "share": share, "shift": shift, "seed": seed,
                    "method": k, "est": e,
                    "cover": lo <= 1.0 <= hi, "tpr": tpr, "fpr": fpr, "n_flag": len(flagged)})
    return res


def mc3(reps, pool):
    jobs = [(N, T, sh, 50_000 + 1000 * int(100 * sh) + 100_000 * k + s)
            for k, (N, T) in enumerate([(100, 50), (200, 100)])
            for sh in [0.0, 0.05, 0.10] for s in range(reps)]
    df = pd.DataFrame([r for rr in pool.map(mc3_rep, jobs, chunksize=10) for r in rr])
    df["err"] = df["est"] - 1.0
    summ = (df.groupby(["N", "T", "share", "method"], sort=False)
              .agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                   coverage=("cover", "mean"), tpr=("tpr", "mean"), fpr=("fpr", "mean"))
              .reset_index())
    return df, summ


def mc3b(reps, pool):
    """Harder detection problem: smaller shift (1.0) of the aberrant slopes."""
    jobs = [(200, 100, sh, 90_000 + 1000 * int(100 * sh) + s, 1.0)
            for sh in [0.05, 0.10] for s in range(reps)]
    df = pd.DataFrame([r for rr in pool.map(mc3_rep, jobs, chunksize=10) for r in rr])
    df["err"] = df["est"] - 1.0
    summ = (df.groupby(["N", "T", "share", "method"], sort=False)
              .agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                   coverage=("cover", "mean"), tpr=("tpr", "mean"), fpr=("fpr", "mean"))
              .reset_index())
    return df, summ


# ---------------------------------------------------------------- MC4 -----
def mc4_rep(args):
    design, seed = args
    N, T = 100, 100
    truth = {"W.y": 0.3, "L1.y": 0.4, "x1": 1.0}
    out = []
    if design == "staggered":
        data, W, tr = sd.simulate_panel(N=N, T=T, entry_share=0.3, seed=seed)
        obs = tr["observed"]
        late_ids = W.index[~obs.all(axis=1)]
        cand = {
            "Unbalanced (all units)": (data, W, {"unbalanced": "staggered"}),
            "Balanced: drop late units": (data[~data["id"].isin(late_ids)],
                                          W.drop(index=late_ids, columns=late_ids)
                                           .pipe(lambda w: w.div(w.sum(axis=1).replace(0, 1), axis=0)),
                                          {}),
            "Balanced: common period": (data[data["time"] >= data.groupby("id")["time"].min().max()],
                                        W, {}),
        }
    else:
        data, W, tr = sd.simulate_panel(N=N, T=T, seed=seed)
        rng = np.random.default_rng(seed + 7)
        drop = rng.uniform(size=len(data)) < 0.02
        dmiss = data[~drop]
        bad = dmiss.groupby("id")["time"].count() < T
        keep = bad.index[~bad]
        Wk = W.loc[keep, keep]
        Wk = Wk.div(Wk.sum(axis=1).replace(0, 1), axis=0)
        cand = {"Unbalanced (all units)": (dmiss, W, {"unbalanced": "sporadic"}),
                "Balanced: complete units": (dmiss[dmiss["id"].isin(keep)], Wk, {})}
    for lab, (d, w, opt) in cand.items():
        try:
            r = _quiet(sd.fit, d, "y", X, "id", "time", w, **KW, **opt)
        except Exception as exc:  # e.g. too few units left
            out.append({"design": design, "seed": seed, "estimator": lab, "fail": str(exc)[:40]})
            continue
        rec = {"design": design, "seed": seed, "estimator": lab, "nobs": r.nobs, "fail": ""}
        for nm, th in truth.items():
            rec[f"err_{nm}"] = r.params[nm] - th
            rec[f"z_{nm}"] = (r.params[nm] - th) / r.bse[nm]
        out.append(rec)
    return out


def mc4(reps, pool):
    jobs = [(d, 70_000 + 10_000 * k + s) for k, d in enumerate(["staggered", "sporadic"])
            for s in range(reps)]
    df = pd.DataFrame([r for rr in pool.map(mc4_rep, jobs, chunksize=10) for r in rr])
    ok = df[df["fail"] == ""]
    rows = []
    for (des, est), g in ok.groupby(["design", "estimator"], sort=False):
        rec = {"design": des, "estimator": est, "nobs": g["nobs"].mean(), "n_ok": len(g)}
        for nm in ["W.y", "L1.y", "x1"]:
            rec[f"bias_{nm}"] = g[f"err_{nm}"].mean()
            rec[f"rmse_{nm}"] = np.sqrt((g[f"err_{nm}"] ** 2).mean())
            rec[f"size_{nm}"] = (g[f"z_{nm}"].abs() > 1.96).mean()
        rows.append(rec)
    return df, pd.DataFrame(rows)


# ---------------------------------------------------------------- timing --
def timing():
    rows = []
    for N, T in [(100, 50), (250, 50), (500, 50), (1000, 50), (100, 200), (500, 200)]:
        data, W, _ = sd.simulate_panel(N=N, T=T, seed=1)
        best = {}
        for lab, opts in [("2SIV", {}), ("MGIV", {"estimator": "mg"})]:
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                r = _quiet(sd.fit, data, "y", X, "id", "time", W, **KW, **opts)
                ts.append(time.perf_counter() - t0)
            best[lab] = min(ts)
            if lab == "2SIV":
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    r.impacts("lr", method="simulation", draws=1000, seed=1)
                    ts.append(time.perf_counter() - t0)
                best["Impacts (1000 draws)"] = min(ts)
        rows.append({"N": N, "T": T, **best})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="1,2,3,3b,4,t",
                    help="comma-separated experiments to run (default: all)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.workers) as pool:
        for key, fn in [("1", mc1), ("2", mc2), ("3", mc3), ("3b", mc3b), ("4", mc4)]:
            if key in a.only.split(","):
                t0 = time.time()
                raw, summ = fn(a.reps, pool)
                raw.to_csv(os.path.join(OUT, f"mc{key}_raw.csv"), index=False)
                summ.to_csv(os.path.join(OUT, f"mc{key}_summary.csv"), index=False)
                print(f"MC{key} done in {time.time() - t0:.0f}s")
                print(summ.round(3).to_string())
    if "t" in a.only.split(","):
        tm = timing()
        tm.to_csv(os.path.join(OUT, "timing.csv"), index=False)
        print(tm.round(3).to_string())
