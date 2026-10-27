# Changelog

## 0.1.0

First release.

* 2SIV estimator of Cui, Sarafidis and Yamagata (2023) and MGIV estimator of
  Chen, Cui, Sarafidis and Yamagata (2025) for spatial dynamic panels with
  interactive effects; first-stage IV estimator.
* Defaults follow the implementation conventions of `spxtivdfreg` 1.4.3; the
  published results of Kripfganz and Sarafidis (2025, Tables 1, 3 and 5)
  are reproduced in every printed digit (`tests/test_stata_replication.py`).
* Label-based alignment of the spatial weights matrix.
* Short- and long-run direct, indirect and total impacts via the complex Schur
  decomposition; Delta-method and Krinsky–Robb simulation inference.
* Mean-group diagnostics: influence, modified z-score flags, conditioning of
  unit-level instruments, trimmed / median summaries with unit bootstrap.
* Experimental support for unbalanced panels (staggered entry, sporadic gaps).
