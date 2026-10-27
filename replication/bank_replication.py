"""Section 6: the bank data of Kripfganz and Sarafidis (2025).

Requires the replication materials of that article (Journal of Statistical
Software 113(6), https://doi.org/10.18637/jss.v113.i06): v113i06.dta and W.csv.

Usage:
    python replication/bank_replication.py path/to/v113i06.dta path/to/W.csv

W.csv is an unlabelled matrix; the original Stata code imports it
positionally, i.e. ordered like the sorted bank identifiers ID = 1, ..., 350.
We pass these identifiers explicitly with `W_ids`.

Outputs: replication/results/bank_*.txt and paper/tables/app_bank_*.tex.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

import spdfiv as sd
from latex_out import impacts_table, robust_table, macros, write, _num, _tex_name

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Published Stata results (Kripfganz and Sarafidis 2025): Table 1 (2SIV) and Table 5 (MG).
TABLE1 = {"W.NPL": .3943206, "L1.NPL": .2898521, "INEFF": .4473777, "CAR": .0305078,
          "SIZE": .2225966, "BUFFER": -.0545049, "PROFIT": -.0053351, "QUALITY": .1830412,
          "LIQUIDITY": 2.452391}
TABLE1_SE = {"W.NPL": .0848856, "L1.NPL": .0543794, "INEFF": .1045636, "CAR": .0057852,
             "SIZE": .0941614, "BUFFER": .0118678, "PROFIT": .0018411, "QUALITY": .0307657,
             "LIQUIDITY": .2696471}
TABLE1_J, TABLE1_SIGF, TABLE1_SIGE = 18.8250, .64162366, .90381799
TABLE5 = {"W.NPL": .031593, "L1.NPL": .3005247, "INEFF": .7587664, "CAR": .218054,
          "SIZE": 2.004026, "BUFFER": -.3763774, "PROFIT": -.0179663, "QUALITY": .2872525,
          "LIQUIDITY": 6.330179}
TABLE5_SE = {"W.NPL": .0511028, "L1.NPL": .0148501, "INEFF": .1583511, "CAR": .0262755,
             "SIZE": .3385335, "BUFFER": .0420252, "PROFIT": .005944, "QUALITY": .1386973,
             "LIQUIDITY": .5059499}
# Long-run impacts printed in Table 3.
TABLE3 = {
    "direct": [.6470588, .0441245, .3219497, -.0788324, -.0077164, .2647392, 3.546983],
    "indirect": [.7694677, .0524719, .3828552, -.0937457, -.0091761, .3148218, 4.217992],
    "total": [1.416526, .0965964, .7048049, -.1725781, -.0168925, .579561, 7.764974]}
TABLE3_SE = {
    "direct": [.1593924, .0092325, .1416728, .0183176, .0023773, .0466629, .4454284],
    "indirect": [.3352809, .0237326, .1975749, .0428643, .0046348, .1408165, 1.742264],
    "total": [.4274849, .0291942, .3099048, .0541498, .0063692, .1670612, 1.90367]}
X = ["INEFF", "CAR", "SIZE", "BUFFER", "PROFIT", "QUALITY", "LIQUIDITY"]
IV = ["INTEREST", "CAR", "SIZE", "BUFFER", "PROFIT", "QUALITY", "LIQUIDITY"]
ORDER = list(TABLE1)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    txt = obj if isinstance(obj, str) else obj.to_string()
    with open(os.path.join(OUT, f"bank_{name}.txt"), "w") as f:
        f.write(txt)
    print(f"== {name} ==\n{txt}\n")


def read_weights_file(path):
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path, header=None).to_numpy(dtype=float)
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def code(nm):
    return r"\code{" + _tex_name(nm) + "}"


def compare_table(res, mg):
    rows = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
            r" & \multicolumn{4}{c}{2SIV (Table~1)} & \multicolumn{4}{c}{Mean group (Table~5)} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            r" & \multicolumn{2}{c}{Coefficient} & \multicolumn{2}{c}{Std.\ error}"
            r" & \multicolumn{2}{c}{Coefficient} & \multicolumn{2}{c}{Std.\ error} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r" & \pkg{spdfiv} & \proglang{Stata} & \pkg{spdfiv} & \proglang{Stata}"
            r" & \pkg{spdfiv} & \proglang{Stata} & \pkg{spdfiv} & \proglang{Stata} \\",
            r"\midrule"]
    for nm in ORDER:
        rows.append(" & ".join([code(nm), _num(res.params[nm], 4), _num(TABLE1[nm], 4),
                                _num(res.bse[nm], 4), _num(TABLE1_SE[nm], 4),
                                _num(mg.params[nm], 4), _num(TABLE5[nm], 4),
                                _num(mg.bse[nm], 4), _num(TABLE5_SE[nm], 4)]) + r" \\")
    rows.append(r"\midrule")
    rows.append(f"$J$ statistic & {res.J:.3f} & {TABLE1_J:.3f} & & & & & & \\\\")
    rows.append(f"$\\sigma_f$ & {res.sigma_f:.4f} & {TABLE1_SIGF:.4f} & & & & & & \\\\")
    rows.append(f"$\\sigma_e$ & {res.sigma_e:.4f} & {TABLE1_SIGE:.4f} & & & & & & \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def influence_table(top):
    rows = [r"\begin{tabular}{rlrrrr}", r"\toprule",
            r"Bank & Coefficient & $\hat\theta_i$ & Modified $z$ & Change in MG (s.e.) & cond$(\bm Z_i^\top\bm Z_i)$ \\",
            r"\midrule"]
    for _, r in top.iterrows():
        m, e = f"{r['cond']:.1e}".split("e")
        rows.append(f"{r['unit']} & {code(r['parameter'])} & {_num(r['theta_i'])} & "
                    f"{_num(r['mod_z'], 2)} & {_num(r['dfmg'], 3)} & "
                    f"${m}\\times 10^{{{int(e)}}}$ \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def stagger(data, share=0.3, max_drop=12, seed=2025):
    """Remove the first 1..max_drop quarters of a random share of banks."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(data["ID"].unique()))
    late = rng.choice(ids, int(round(share * len(ids))), replace=False)
    drop = dict(zip(late, rng.integers(1, max_drop + 1, len(late))))
    keep = data["TIME"] > data["ID"].map(drop).fillna(0)
    return data[keep], late


def main(data_path, w_path):
    data = (pd.read_stata(data_path) if data_path.endswith(".dta") else pd.read_csv(data_path))
    W = read_weights_file(w_path)
    ids = sorted(data["ID"].unique())
    common = dict(W_ids=ids, splag=True, tlags=1, iv=IV, iv_lags=1, iv_splags=True,
                  std=True, absorb=True)

    # ---- (1) cross-software check ---------------------------------------------
    res = sd.fit(data, "NPL", X, "ID", "TIME", W, **common)
    mg = sd.fit(data, "NPL", X, "ID", "TIME", W, estimator="mg", **common)
    save("2siv", res.summary())
    save("mg", mg.summary())
    mg.unit_params.join(mg.unit_se, rsuffix="_se").join(
        mg.extras["unit_cond"].rename("cond")).to_csv(os.path.join(OUT, "bank_unit_estimates.csv"))
    cmp = pd.DataFrame({"spdfiv": res.params, "Stata": pd.Series(TABLE1),
                        "se": res.bse, "se Stata": pd.Series(TABLE1_SE),
                        "MG": mg.params, "MG Stata": pd.Series(TABLE5),
                        "MG se": mg.bse, "MG se Stata": pd.Series(TABLE5_SE)}).loc[ORDER]
    cmp["rel diff"] = (cmp["spdfiv"] - cmp["Stata"]) / cmp["Stata"].abs()
    cmp["rel diff se"] = (cmp["se"] - cmp["se Stata"]) / cmp["se Stata"]
    cmp["MG diff / se"] = (cmp["MG"] - cmp["MG Stata"]) / cmp["MG se Stata"]
    save("compare", cmp)
    write("app_bank_compare.tex", compare_table(res, mg))

    # MG without double defactoring, to show its sensitivity
    mg_nodd = sd.fit(data, "NPL", X, "ID", "TIME", W, estimator="mg", doubledefact=False, **common)
    save("mg_nodd", mg_nodd.summary())

    # ---- (2) long-run impacts: Delta vs simulation -------------------------------
    lr_d = res.impacts("lr", method="delta")
    lr_s = res.impacts("lr", method="simulation", draws=10000, seed=1)
    save("impacts_delta", lr_d); save("impacts_sim", lr_s)
    t3 = pd.Series([v for e in ("direct", "indirect", "total") for v in TABLE3[e]], index=lr_d.index)
    t3se = pd.Series([v for e in ("direct", "indirect", "total") for v in TABLE3_SE[e]], index=lr_d.index)
    imp_rel = ((lr_d["impact"] - t3) / t3.abs()).abs().max()
    imp_rel_se = ((lr_d["se"] - t3se) / t3se).abs().max()
    widen = ((lr_s["upper"] - lr_s["lower"]) / (lr_d["upper"] - lr_d["lower"]))
    save("impacts_vs_table3", pd.DataFrame({"spdfiv": lr_d["impact"], "Table 3": t3,
                                            "se": lr_d["se"], "se Table 3": t3se}))
    write("app_bank_impacts.tex", impacts_table(lr_d, lr_s))

    # ---- (3) what drives the MG estimates -----------------------------------------
    infl = sd.mg_influence(mg)
    flagged = sd.flagged_units(infl, parameters=X)
    ill = mg.extras["unit_cond"].index[mg.extras["unit_cond"] > 1e12].tolist()
    top = infl[infl.parameter.isin(["SIZE", "BUFFER"])].sort_values(
        "dfmg", key=np.abs, ascending=False).head(8)
    save("influence_top", top)
    write("app_bank_influence.tex", influence_table(top))
    rob = {"Trimmed (10\\%)": sd.mg_robust(mg, "trimmed", 0.1, bootstrap=999, seed=1),
           "Median": sd.mg_robust(mg, "median", bootstrap=999, seed=1),
           "MG excl.\\ flagged": sd.mg_robust(mg, "mean", exclude=flagged, bootstrap=999, seed=1)}
    for k, v in rob.items():
        save("robust_" + k.split()[0].strip("\\").lower().rstrip("."), v)
    write("app_bank_mg.tex", robust_table(mg, rob))
    excl_ill = sd.mg_robust(mg, "mean", exclude=ill, bootstrap=1, seed=1)
    save("mg_excl_ill", excl_ill)

    # ---- (4) staggered entry on the same data --------------------------------------
    dst, late = stagger(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ru = sd.fit(dst, "NPL", X, "ID", "TIME", W, unbalanced="staggered", **common)
    keep_ids = [i for i in ids if i not in set(late)]
    Wm = pd.DataFrame(W, index=ids, columns=ids).loc[keep_ids, keep_ids]
    Wm = Wm.div(Wm.sum(axis=1).replace(0, 1), axis=0)
    rdrop = sd.fit(data[data["ID"].isin(keep_ids)], "NPL", X, "ID", "TIME", Wm,
                   **{k: v for k, v in common.items() if k != "W_ids"})
    save("2siv_staggered", ru.summary())
    save("2siv_droplate", rdrop.summary())
    stag = pd.DataFrame({"full": res.params, "staggered": ru.params, "drop late": rdrop.params,
                         "se full": res.bse, "se staggered": ru.bse, "se drop": rdrop.bse})
    save("staggered_compare", stag)
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{2}{c}{Full panel} & \multicolumn{2}{c}{Staggered, all banks}"
            r" & \multicolumn{2}{c}{Late banks dropped} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r" & Coef. & s.e. & Coef. & s.e. & Coef. & s.e. \\", r"\midrule"]
    for nm in ORDER:
        rows.append(" & ".join([code(nm), _num(res.params[nm]), _num(res.bse[nm]),
                                _num(ru.params[nm]), _num(ru.bse[nm]),
                                _num(rdrop.params[nm]), _num(rdrop.bse[nm])]) + r" \\")
    rows.append(r"\midrule")
    rows.append("Observations & " + " & ".join(
        r"\multicolumn{2}{c}{" + f"{r.nobs:,}".replace(",", "{,}") + "}" for r in (res, ru, rdrop)) + r" \\")
    rows.append("Banks & " + " & ".join(
        r"\multicolumn{2}{c}{" + str(r.n_units) + "}" for r in (res, ru, rdrop)) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("app_bank_staggered.tex", "\n".join(rows) + "\n")

    # ---- numbers quoted in the text ------------------------------------------------
    gap = lambda est: abs(est - res.params)
    write("app_bank_macros.tex", macros("Bank", {
        "MaxRelCoef": f"{100 * cmp['rel diff'].abs().max():.3f}",
        "MaxRelSE": f"{100 * cmp['rel diff se'].abs().max():.2f}",
        "MaxMGdiffSE": f"{cmp['MG diff / se'].abs().max():.2f}",
        "MedMGdiffSE": f"{cmp['MG diff / se'].abs().median():.2f}",
        "J": f"{res.J:.3f}", "pJ": f"{res.p_J:.3f}",
        "Nflag": len(flagged), "Nunits": mg.n_units,
        "NflagSize": len(sd.flagged_units(infl, ["SIZE"])),
        "NflagBuffer": len(sd.flagged_units(infl, ["BUFFER"])),
        "Nill": len(ill), "NillFlag": len(set(ill) & set(flagged)),
        "MGpsi": _num(mg.params["W.NPL"]), "MGpsiNoDD": _num(mg_nodd.params["W.NPL"]),
        "MGsize": _num(mg.params["SIZE"]), "MGbuffer": _num(mg.params["BUFFER"]),
        "TwoSize": _num(res.params["SIZE"]), "TwoBuffer": _num(res.params["BUFFER"]),
        "TwoPsi": _num(res.params["W.NPL"]),
        "MedSize": _num(rob["Median"].loc["SIZE", "median"]),
        "MedBuffer": _num(rob["Median"].loc["BUFFER", "median"]),
        "MedPsi": _num(rob["Median"].loc["W.NPL", "median"]),
        "TrimSize": _num(rob["Trimmed (10\\%)"].loc["SIZE", "trimmed"]),
        "ExclSize": _num(rob["MG excl.\\ flagged"].loc["SIZE", "mean"]),
        "ExclBuffer": _num(rob["MG excl.\\ flagged"].loc["BUFFER", "mean"]),
        "ExclPsi": _num(rob["MG excl.\\ flagged"].loc["W.NPL", "mean"]),
        "ExclIllSize": _num(excl_ill.loc["SIZE", "mean"]),
        "GapCutSize": f"{100 * (1 - gap(rob['Median'].loc['SIZE', 'median'])['SIZE'] / gap(mg.params['SIZE'])['SIZE']):.0f}",
        "GapCutBuffer": f"{100 * (1 - gap(rob['Median'].loc['BUFFER', 'median'])['BUFFER'] / gap(mg.params['BUFFER'])['BUFFER']):.0f}",
        "Nlate": len(late), "NobsStag": f"{ru.nobs:,}".replace(",", "{,}"),
        "NobsDrop": f"{rdrop.nobs:,}".replace(",", "{,}"),
        "PsiStag": _num(ru.params["W.NPL"]), "PsiDrop": _num(rdrop.params["W.NPL"]),
        "MaxStagDiffSE": f"{((ru.params - res.params).abs() / res.bse).max():.2f}",
        "MaxDropDiffSE": f"{((rdrop.params - res.params).abs() / res.bse).max():.2f}",
        "ImpRel": f"{100 * imp_rel:.2f}", "ImpRelSE": f"{100 * imp_rel_se:.2f}",
        "SimStable": f"{100 * lr_s.attrs['share_stable']:.1f}",
        "WidenInd": f"{widen.loc['indirect'].median():.2f}",
        "WidenDir": f"{widen.loc['direct'].median():.2f}",
        "SizeIndDelta": f"[{_num(lr_d.loc[('indirect', 'SIZE'), 'lower'])}, {_num(lr_d.loc[('indirect', 'SIZE'), 'upper'])}]",
        "SizeIndSim": f"[{_num(lr_s.loc[('indirect', 'SIZE'), 'lower'])}, {_num(lr_s.loc[('indirect', 'SIZE'), 'upper'])}]",
    }))


if __name__ == "__main__":
    main(*sys.argv[1:3])
