# spdfiv

Defactored instrumental variables (IV) estimation of spatial dynamic panel
data models with interactive effects (unobserved common factors), in Python.

Implements

* the two-stage IV (2SIV) estimator of Cui, Sarafidis and Yamagata (2023,
  *The Econometrics Journal*) for homogeneous slopes,
* the mean-group IV (MGIV) estimator of Chen, Cui, Sarafidis and Yamagata
  (2025, arXiv:2501.18467) for heterogeneous slopes,

and adds

* label-based alignment of the spatial weights matrix,
* short- and long-run direct, indirect and total impacts computed through a
  Schur decomposition, with Delta-method or Krinsky–Robb simulation inference,
* influence statistics, outlier flags and robust (trimmed / median) summaries
  of unit-specific estimates with bootstrap inference,
* experimental support for unbalanced panels (staggered entry, sporadic gaps).

## Installation

```
pip install git+https://github.com/abdulameert-boop/spdfiv
```

or, from a local copy of the repository, `pip install .`

Requires Python >= 3.9, NumPy, SciPy and pandas (openpyxl to read Excel
weights files).

## Quick start

```python
import spdfiv as sd

data, W, truth = sd.simulate_panel(N=300, T=40, seed=2025)
res = sd.fit(data, y="y", x=["x1", "x2"], unit="id", time="time", W=W,
             splag=True, tlags=1, iv_lags=1, iv_splags=True)
print(res.summary())
print(res.impacts("lr", method="simulation", draws=5000, seed=1))

mg = sd.fit(data, "y", ["x1", "x2"], "id", "time", W, tlags=1, iv_lags=1,
            estimator="mg")
infl = sd.mg_influence(mg)
print(sd.mg_robust(mg, method="trimmed", trim=0.1))
```

`W` must be a `pandas.DataFrame` labelled by unit on both axes, or an array
with `W_ids=`; unlabelled arrays are rejected unless `assume_ordered=True`.

## Tests

```
python -m unittest discover -s tests
```

The suite includes a literal implementation of the 2SIV equations that the
package must reproduce to machine precision, and (when the environment
variable `SPDFIV_JSS_DATA` points to the replication materials of Kripfganz
and Sarafidis 2025) a check that the published Stata results of their
Tables 1 and 5 are reproduced in every printed digit.

API reference: `docs/api.md` (regenerate with `python docs/make_api.py`).

## Replication

See `replication/README.md`.

## License

GPL-3.0-or-later.
