"""Plain-text coefficient tables."""

from __future__ import annotations

import numpy as np

_TITLES = {
    "2siv": "Second-stage estimator (model with homogeneous slope coefficients)",
    "fstage": "First-stage estimator (model with homogeneous slope coefficients)",
    "mg": "Mean-group estimator (model with heterogeneous slope coefficients)",
}


def summary_text(res, level: float = 0.95) -> str:
    s = res.spec
    ci = res.conf_int(level)
    lines = []
    w = 78
    lines.append("Defactored instrumental variables estimation")
    n = res.obs_per_unit
    lines.append(f"{'Number of obs':<28}= {res.nobs:>10}   {'Number of units':<20}= {res.n_units:>8}")
    lines.append(f"{'Number of instruments':<28}= {res.n_instruments:>10}   "
                 f"{'Obs per unit (min/avg/max)':<20}= {n.min()}/{n.mean():.1f}/{n.max()}")
    fx = "/".join(str(v) for v in res.factors_x)
    lines.append(f"{'Factors in instruments':<28}= {fx:>10}   "
                 f"{'Factors in u':<20}= {res.factors_u if res.factors_u is not None else '-':>8}")
    if s.unbalanced != "none":
        lines.append(f"Unbalanced panel ({s.unbalanced}); experimental.")
    lines.append(_TITLES[s.estimator])
    lines.append("-" * w)
    lines.append(f"{'':>14} {'Coef.':>11} {'Std. err.':>11} {'z':>8} {'P>|z|':>7}"
                 f" {'[' + f'{100 * level:.0f}% conf. interval' + ']':>24}")
    lines.append("-" * w)
    for nm in res.params.index:
        lines.append(f"{nm:>14} {res.params[nm]:>11.6f} {res.bse[nm]:>11.6f} "
                     f"{res.zvalues[nm]:>8.2f} {res.pvalues[nm]:>7.3f} "
                     f"{ci.loc[nm, 'lower']:>11.6f} {ci.loc[nm, 'upper']:>11.6f}")
    lines.append("-" * w)
    if res.sigma_f is not None:
        lines.append(f"{'sigma_f':>14} {res.sigma_f:>11.6f}  (std. dev. of factor error component)")
        lines.append(f"{'sigma_e':>14} {res.sigma_e:>11.6f}  (std. dev. of idiosyncratic error component)")
        lines.append(f"{'rho':>14} {res.rho_factor:>11.6f}  (fraction of variance due to factors)")
        lines.append("-" * w)
    if res.J is not None and res.df_J:
        lines.append(f"Hansen test of the overidentifying restrictions  chi2({res.df_J}) = {res.J:9.4f}")
        lines.append(f"H0: overidentifying restrictions are valid        Prob > chi2 = {res.p_J:9.4f}")
    return "\n".join(lines)
