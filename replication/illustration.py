"""Code and output of Section 4 (illustration with simulated data).

Every code chunk shown in the paper is reproduced here; the output of each
chunk is written to replication/results/illustration_<k>.txt.
"""

import contextlib
import io
import os
import warnings

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 78)
pd.set_option("display.precision", 4)


@contextlib.contextmanager
def chunk(k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    txt = buf.getvalue()
    with open(os.path.join(OUT, f"illustration_{k}.txt"), "w") as f:
        f.write(txt)
    print(f"--- chunk {k} ---\n{txt}")


# ---- chunk 1: data -----------------------------------------------------------
with chunk(1):
    import spdfiv as sd
    data, W, truth = sd.simulate_panel(N=300, T=40, psi=0.3, rho=0.4,
                                       beta=(1.0, 0.5), seed=2025)
    print(data.head(3))
    print(W.shape, W.index[:5].tolist())

# ---- chunk 2: 2SIV -----------------------------------------------------------
with chunk(2):
    res = sd.fit(data, y="y", x=["x1", "x2"], unit="id", time="time", W=W,
                 splag=True, tlags=1, iv_lags=1, iv_splags=True)
    print(res.summary())

# ---- chunk 3: ignoring factors -------------------------------------------------
with chunk(3):
    res0 = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
                  factmax=0)
    print(pd.DataFrame({"2SIV": res.params, "no factors": res0.params}).round(4))
    print(f"J p-values: 2SIV {res.p_J:.3f}, no factors {res0.p_J:.4f}")

# ---- chunk 4: impacts ------------------------------------------------------------
with chunk(4):
    lr_delta = res.impacts("lr", method="delta")
    lr_sim = res.impacts("lr", method="simulation", draws=5000, seed=1)
    print(lr_sim.round(4))
    print(pd.concat({"delta": lr_delta[["se", "lower", "upper"]],
                     "simulation": lr_sim[["se", "lower", "upper"]]},
                    axis=1).loc["total"].round(4))

# ---- chunk 5: alignment safeguard -------------------------------------------------
with chunk(5):
    try:
        sd.fit(data, "y", ["x1", "x2"], "id", "time", W.to_numpy())
    except sd.WeightsAlignmentError as e:
        import textwrap
        print(textwrap.fill("WeightsAlignmentError: " + str(e), 76))
    perm = W.index[::-1]
    res_p = sd.fit(data, "y", ["x1", "x2"], "id", "time", W.loc[perm, perm],
                   tlags=1, iv_lags=1)
    print("max |difference|:", float((res_p.params - res.params).abs().max()))

# ---- chunk 6: heterogeneous slopes with a few aberrant units -----------------------
with chunk(6):
    dh, Wh, th = sd.simulate_panel(N=300, T=40, het=0.2, outlier_share=0.03,
                                   seed=7)
    mg = sd.fit(dh, "y", ["x1", "x2"], "id", "time", Wh, tlags=1, iv_lags=1,
                estimator="mg")
    print(mg.summary())

with chunk(7):
    infl = sd.mg_influence(mg, threshold=3.5)
    cols = ["unit", "parameter", "theta_i", "mod_z", "dfmg", "flag"]
    print(infl.sort_values("dfmg", key=abs, ascending=False)[cols].head(6))
    flagged = sd.flagged_units(infl, parameters=["x1", "x2"])
    print("flagged:", sorted(flagged))
    print("true aberrant units:", sorted(th["outliers"].tolist()))

with chunk(8):
    print(sd.mg_robust(mg, method="trimmed", trim=0.1, bootstrap=999,
                       seed=1).round(4))
    print(sd.mg_robust(mg, method="mean", exclude=flagged, bootstrap=999,
                       seed=1).round(4))

# ---- chunk 9: staggered entry --------------------------------------------------------
with chunk(9):
    du, Wu, tu = sd.simulate_panel(N=300, T=40, entry_share=0.3, seed=11)
    print(du.groupby("id")["time"].min().value_counts().sort_index().head(4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ru = sd.fit(du, "y", ["x1", "x2"], "id", "time", Wu, tlags=1, iv_lags=1,
                    unbalanced="staggered")
    print("UserWarning:", str(w[0].message))
    print(ru.summary())
