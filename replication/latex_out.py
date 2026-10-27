"""Write estimation results as LaTeX tables for paper/tables/."""

import os

import numpy as np
import pandas as pd

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tables")


def _num(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    t = f"{v:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    return t.replace("-", "$-$")


def _stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def _tex_name(nm):
    return nm.replace("_", r"\_")


def write(name, body):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write(body)


def coef_table(results: dict, labels: dict | None = None, d: int = 3) -> str:
    """Columns = models; rows = coefficients (se), factors, J test, N obs."""
    labels = labels or {}
    names = list(dict.fromkeys(n for r in results.values() for n in r.params.index))
    cols = list(results)
    out = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & " + " & ".join(cols) + r" \\", r"\midrule"]
    for nm in names:
        est, se = [], []
        for c in cols:
            r = results[c]
            if nm in r.params.index:
                est.append(f"${_num(r.params[nm], d)}{_stars(r.pvalues[nm])}$")
                se.append(f"$({_num(r.bse[nm], d)})$")
            else:
                est.append(""); se.append("")
        out.append(labels.get(nm, r"\code{" + _tex_name(nm) + "}") + " & " + " & ".join(est) + r" \\")
        out.append(" & " + " & ".join(se) + r" \\")
    out.append(r"\midrule")
    out.append(r"Factors in instruments & " + " & ".join(
        "/".join(map(str, results[c].factors_x)) for c in cols) + r" \\")
    out.append(r"Factors in residuals & " + " & ".join(
        str(results[c].factors_u) if results[c].factors_u is not None else "--" for c in cols) + r" \\")
    out.append(r"$J$ statistic [$p$ value] & " + " & ".join(
        (f"{results[c].J:.2f} [{results[c].p_J:.3f}]" if results[c].p_J is not None else "--")
        for c in cols) + r" \\")
    out.append(r"Observations & " + " & ".join(f"{results[c].nobs:,}".replace(",", "{,}") for c in cols) + r" \\")
    out.append(r"Units & " + " & ".join(str(results[c].n_units) for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def impacts_table(delta: pd.DataFrame, sim: pd.DataFrame, d: int = 3) -> str:
    out = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r" & & & \multicolumn{2}{c}{Delta method} & \multicolumn{2}{c}{Simulation} \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Impact & Variable & Estimate & s.e. & 95\% interval & s.e. & 95\% interval \\",
           r"\midrule"]
    prev = None
    for (eff, var) in delta.index:
        if prev is not None and eff != prev:
            out.append(r"\addlinespace")
        a, b = delta.loc[(eff, var)], sim.loc[(eff, var)]
        out.append(" & ".join([
            eff if eff != prev else "", r"\code{" + _tex_name(var) + "}", _num(a["impact"], d),
            _num(a["se"], d), f"[{_num(a['lower'], d)}, {_num(a['upper'], d)}]",
            _num(b["se"], d), f"[{_num(b['lower'], d)}, {_num(b['upper'], d)}]"]) + r" \\")
        prev = eff
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def robust_table(mg_res, summaries: dict, d: int = 3) -> str:
    """summaries: label -> DataFrame from mg_robust (point column + lower/upper)."""
    cols = ["MG"] + list(summaries)
    out = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & " + " & ".join(cols) + r" \\", r"\midrule"]
    ci = mg_res.conf_int()
    for nm in mg_res.params.index:
        row = [r"\code{" + _tex_name(nm) + "}", _num(mg_res.params[nm], d)]
        row += [_num(s.loc[nm, s.columns[2]], d) for s in summaries.values()]
        out.append(" & ".join(row) + r" \\")
        row = ["", f"[{_num(ci.loc[nm, 'lower'], d)}, {_num(ci.loc[nm, 'upper'], d)}]"]
        row += [f"[{_num(s.loc[nm, 'lower'], d)}, {_num(s.loc[nm, 'upper'], d)}]"
                for s in summaries.values()]
        out.append(r"\footnotesize " + r" & \footnotesize ".join(row) + r" \\")
    out.append(r"\midrule")
    out.append("Units & " + str(mg_res.n_units) + " & " + " & ".join(
        str(s.attrs.get("n_units", "")) for s in summaries.values()) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def macros(prefix: str, values: dict) -> str:
    """\\newcommand definitions so that the text can quote computed numbers."""
    lines = []
    for k, v in values.items():
        lines.append(f"\\providecommand{{\\{prefix}{k}}}{{}}\\renewcommand{{\\{prefix}{k}}}{{{v}}}")
    return "\n".join(lines) + "\n"
