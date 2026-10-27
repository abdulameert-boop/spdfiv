"""Turn the Monte Carlo CSV summaries into the LaTeX tables of the paper."""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "..", "paper", "tables")
P = {"W.y": r"$\psi_0$", "L1.y": r"$\rho_1$", "x1": r"$\beta_1$"}


def f3(v):
    t = f"{v:.3f}"
    if t == "-0.000":
        t = "0.000"
    return t.replace("-", "$-$")


def pct(v):
    return f"{100 * v:.1f}"


def write(name, body):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write(body)


def mc1():
    s = pd.read_csv(os.path.join(RES, "mc1_summary.csv"))
    out = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
           r" & & \multicolumn{3}{c}{" + P["W.y"] + r"} & \multicolumn{3}{c}{" + P["L1.y"]
           + r"} & \multicolumn{3}{c}{" + P["x1"] + r"} & \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
           r"$(N,T)$ & Estimator & Bias & RMSE & Size & Bias & RMSE & Size & Bias & RMSE & Size & $J$ \\",
           r"\midrule"]
    for k, ((N, T), g) in enumerate(s.groupby(["N", "T"], sort=False)):
        if k:
            out.append(r"\addlinespace")
        for j, (_, r) in enumerate(g.iterrows()):
            lab = f"$({N},{T})$" if j == 0 else ""
            cells = [lab, r["estimator"]]
            for nm in ["W.y", "L1.y", "x1"]:
                cells += [f3(r[f"bias_{nm}"]), f3(r[f"rmse_{nm}"]), pct(r[f"size_{nm}"])]
            cells.append(pct(r["J_rej"]))
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("mc1.tex", "\n".join(out) + "\n")
    return s


def mc2():
    s = pd.read_csv(os.path.join(RES, "mc2_summary.csv"))
    out = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
           r" & & \multicolumn{4}{c}{Delta method} & \multicolumn{4}{c}{Simulation (Krinsky--Robb)} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
           r"$(N,T)$ & Impact & Cover & Below & Above & Width & Cover & Below & Above & Width \\",
           r"\midrule"]
    for k, ((N, T), g) in enumerate(s.groupby(["N", "T"], sort=False)):
        if k:
            out.append(r"\addlinespace")
        for j, eff in enumerate(["direct", "indirect", "total"]):
            cells = [f"$({N},{T})$" if j == 0 else "", eff]
            for m in ["delta", "simulation"]:
                r = g[(g.effect == eff) & (g.method == m)].iloc[0]
                cells += [pct(r["coverage"]), pct(r["miss_below"]), pct(r["miss_above"]),
                          f"{r['width']:.2f}"]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("mc2.tex", "\n".join(out) + "\n")
    return s


def mc3():
    s = pd.read_csv(os.path.join(RES, "mc3_summary.csv"))
    methods = ["MG", "trimmed", "median", "MG excl. flagged"]
    labels = {"MG": "MG", "trimmed": "Trimmed MG (10\\%)", "median": "Median",
              "MG excl. flagged": "MG excl.\\ flagged"}
    out = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r" & & \multicolumn{3}{c}{No aberrant units} & \multicolumn{3}{c}{5\% aberrant} & \multicolumn{3}{c}{10\% aberrant} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
           r"$(N,T)$ & Estimator & Bias & RMSE & Cover & Bias & RMSE & Cover & Bias & RMSE & Cover \\",
           r"\midrule"]
    det = []
    for k, ((N, T), g) in enumerate(s.groupby(["N", "T"], sort=False)):
        if k:
            out.append(r"\addlinespace")
        for j, m in enumerate(methods):
            cells = [f"$({N},{T})$" if j == 0 else "", labels[m]]
            for sh in [0.0, 0.05, 0.10]:
                r = g[(g.method == m) & (g.share.round(2) == sh)].iloc[0]
                cells += [f3(r["bias"]), f3(r["rmse"]), pct(r["coverage"])]
            out.append(" & ".join(cells) + r" \\")
        cells = ["", r"\emph{Flagging: TPR / FPR}"]
        for sh in [0.0, 0.05, 0.10]:
            r = g[(g.method == "MG") & (g.share.round(2) == sh)].iloc[0]
            tpr = "--" if pd.isna(r["tpr"]) else pct(r["tpr"])
            cells += [r"\multicolumn{3}{c}{" + f"{tpr} / {pct(r['fpr'])}" + "}"]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("mc3.tex", "\n".join(out) + "\n")
    return s


def mc4():
    s = pd.read_csv(os.path.join(RES, "mc4_summary.csv"))
    out = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
           r" & & & \multicolumn{3}{c}{" + P["W.y"] + r"} & \multicolumn{3}{c}{" + P["L1.y"]
           + r"} & \multicolumn{3}{c}{" + P["x1"] + r"} \\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
           r"Design & Estimator & $NT$ & Bias & RMSE & Size & Bias & RMSE & Size & Bias & RMSE & Size \\",
           r"\midrule"]
    names = {"staggered": "Staggered", "sporadic": "Sporadic"}
    for k, (des, g) in enumerate(s.groupby("design", sort=False)):
        if k:
            out.append(r"\addlinespace")
        for j, (_, r) in enumerate(g.iterrows()):
            short = {"Unbalanced (all units)": "Unbalanced, all units",
                     "Balanced: drop late units": "Drop late units",
                     "Balanced: common period": "Common period",
                     "Balanced: complete units": "Complete units only"}
            cells = [names[des] if j == 0 else "", short.get(r["estimator"], r["estimator"]),
                     f"{r['nobs']:.0f}"]
            for nm in ["W.y", "L1.y", "x1"]:
                cells += [f3(r[f"bias_{nm}"]), f3(r[f"rmse_{nm}"]), pct(r[f"size_{nm}"])]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("mc4.tex", "\n".join(out) + "\n")
    return s


def timing():
    s = pd.read_csv(os.path.join(RES, "timing.csv"))
    cols = [c for c in s.columns if c not in ("N", "T")]
    out = [r"\begin{tabular}{rr" + "r" * len(cols) + "}", r"\toprule",
           "$N$ & $T$ & " + " & ".join(cols) + r" \\", r"\midrule"]
    for _, r in s.iterrows():
        out.append(f"{int(r['N'])} & {int(r['T'])} & " + " & ".join(f"{r[c]:.2f}" for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    write("timing.tex", "\n".join(out) + "\n")
    return s


if __name__ == "__main__":
    for fn in (mc1, mc2, mc3, mc4, timing):
        try:
            print(fn().round(3).to_string(), "\n")
        except FileNotFoundError as e:
            print("missing:", e)
