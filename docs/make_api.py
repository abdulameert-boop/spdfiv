"""Generate docs/api.md from the docstrings of the public functions."""
import inspect, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spdfiv as sd

items = [sd.fit, sd.impacts, sd.mg_influence, sd.mg_robust, sd.flagged_units,
         sd.align_weights, sd.read_weights, sd.prepare_panel, sd.spatial_lag,
         sd.simulate_panel, sd.knn_weights]
out = ["# API reference", "", f"spdfiv {sd.__version__}. Generated from the docstrings by `docs/make_api.py`.", ""]
for f in items:
    out += [f"## `{f.__name__}{inspect.signature(f)}`", "", "```text", inspect.getdoc(f) or "", "```", ""]
out += ["## Results object (`SpatialDFIVResults`)", "",
        "Attributes: `params`, `cov`, `bse`, `zvalues`, `pvalues`, `conf_int(level)`, "
        "`nobs`, `n_units`, `obs_per_unit`, `n_instruments`, `factors_x`, `factors_u`, "
        "`J`, `df_J`, `p_J`, `sigma_f`, `sigma_e`, `rho_factor`, `unit_params`, `unit_se`, "
        "`extras` (`first_stage_params`, `unit_cond`).", "",
        "Methods: `summary()`, `impacts(...)`, `mg_robust(...)`, `mg_influence(...)`.", ""]
open(os.path.join(os.path.dirname(__file__), "api.md"), "w").write("\n".join(out))
print("written")
