# API reference

spdfiv 0.1.0. Generated from the docstrings by `docs/make_api.py`.

## `fit(data: 'pd.DataFrame', y: 'str', x: 'Sequence[str]', unit: 'str', time: 'str', W, *, W_ids=None, assume_ordered: 'bool' = False, splag: 'bool' = True, tlags: 'int' = 0, sptlags: 'int' = 0, spx: 'Optional[Sequence[str]]' = None, iv: 'Optional[Sequence[str]]' = None, iv_lags: 'int' = 0, iv_splags: 'bool' = True, factmax: 'int' = 4, eigratio: 'bool' = True, std=False, absorb: 'bool' = True, estimator: 'str' = '2siv', zero_factors: 'bool' = False, common_rank: 'bool' = True, omega: 'str' = 'first', doubledefact: 'Optional[bool]' = None, unbalanced: 'str' = 'none', renormalize: 'bool' = True) -> 'SpatialDFIVResults'`

```text
Estimate a spatial dynamic panel model with interactive effects.

Parameters
----------
data : DataFrame
    Long-format panel data.
y, x : str, list of str
    Dependent variable and covariates. Covariates not listed in ``iv``
    are treated as endogenous with respect to the idiosyncratic error.
unit, time : str
    Panel and time identifiers.
W : DataFrame or array
    Spatial weights matrix, labelled by unit (see
    :func:`spdfiv.panel.align_weights`).
splag : bool
    Include the spatial lag of ``y``.
tlags, sptlags : int
    Number of time lags and spatial time lags of ``y``.
spx : list of str
    Covariates whose spatial lags enter the model (spatial Durbin terms).
iv : list of str
    Exogenous variables used to build defactored instruments. Defaults
    to ``x``.
iv_lags : int
    Number of time lags of the instruments; factors are estimated
    separately for each lag order.
iv_splags : bool
    Add spatial lags of the defactored instruments.
factmax : int
    Maximum number of factors (``0`` gives standard IV estimation).
eigratio : bool
    Choose the number of factors with the Ahn-Horenstein eigenvalue
    ratio. If ``False`` exactly ``factmax`` factors are used.
std : bool or {"period", "pooled"}
    Standardize the variables from which factors are extracted. ``True``
    (or ``"period"``) standardizes across units within each period, as
    the option ``std`` of spxtivdfreg; ``"pooled"`` over all cells.
zero_factors : bool
    Allow the eigenvalue-ratio criterion to select zero factors (mock
    eigenvalue of Ahn and Horenstein 2013). ``False`` (default) follows
    spxtivdfreg, which selects between 1 and ``factmax`` factors.
omega : {"first", "second"}
    Residuals used for the weight matrix Omega in the variance and the J
    statistic of the 2SIV estimator: ``"first"`` (default) as in the
    equations of Cui et al. (2023) and in spxtivdfreg; ``"second"``
    re-evaluates Omega at the second-stage residuals.
common_rank : bool
    Use, for all lag blocks of the instruments, the largest number of
    factors selected for any block (as spxtivdfreg). If ``False`` each
    block keeps its own number.
absorb : bool
    Remove unit fixed effects by within transformation.
estimator : {"2siv", "fstage", "mg"}
    Two-stage IV, first-stage IV, or mean-group IV.
doubledefact : bool, optional
    Additionally project the contemporaneous instrument factors out of
    the lagged instruments. Default: ``True`` for ``"mg"``, else ``False``.
unbalanced : {"none", "staggered", "sporadic"}
    ``"none"`` requires a balanced panel. ``"staggered"`` allows units to
    enter or leave the panel (contiguous observation spells).
    ``"sporadic"`` additionally allows interior gaps. Both are
    experimental: asymptotic theory for these cases is not available.
renormalize : bool
    With missing neighbours, rescale the remaining weights of each row
    to keep its original row sum.
```

## `impacts(res, kind: 'str' = 'sr', method: 'str' = 'delta', variables=None, draws: 'int' = 2000, seed=None, level: 'float' = 0.95, force: 'bool' = False) -> 'pd.DataFrame'`

```text
Average direct, indirect and total impacts with standard errors.

Parameters
----------
kind : {"sr", "lr"}
    Short-run or long-run impacts.
method : {"delta", "simulation"}
    Delta method or Krinsky-Robb simulation.
variables : list of str, optional
    Covariates; defaults to all covariates in the model.
draws : int
    Number of simulation draws (``method="simulation"``).
force : bool
    Report impacts even if the point estimates violate stability.
```

## `mg_influence(res, threshold: 'float' = 3.5, cond_threshold: 'float' = 1000000000000.0) -> 'pd.DataFrame'`

```text
Influence of each unit on the mean-group estimate.

Returns a long DataFrame with, for every unit and coefficient,

* ``theta_i`` - the unit-specific estimate,
* ``mod_z`` - the modified z-score 0.6745 (theta_i - median) / MAD,
* ``dfmg`` - the change in the MG estimate from deleting the unit,
  MG - MG(-i) = (theta_i - MG) / (N - 1), in units of the MG standard
  error,
* ``flag`` - ``|mod_z| > threshold`` (3.5 as recommended by Iglewicz
  and Hoaglin 1993),
* ``cond`` - the condition number of the unit's instrument cross-product
  Z_i'Z_i, and ``ill_conditioned`` - ``cond > cond_threshold``; such units
  have (nearly) collinear instruments, so their estimates depend on
  numerical tolerances.
```

## `mg_robust(res, method: 'str' = 'trimmed', trim: 'float' = 0.1, exclude=None, bootstrap: 'int' = 999, level: 'float' = 0.95, seed=None) -> 'pd.DataFrame'`

```text
Robust location estimates of the unit-specific coefficients.

Parameters
----------
method : {"trimmed", "median", "mean"}
    Location estimator applied coefficient by coefficient.
trim : float
    Proportion cut from *each* tail for the trimmed mean.
exclude : list, optional
    Units removed before computing the estimates (e.g. those flagged by
    :func:`mg_influence`).
bootstrap : int
    Number of bootstrap resamples of units for standard errors and
    percentile confidence intervals.
```

## `flagged_units(influence: 'pd.DataFrame', parameters=None) -> 'list'`

```text
Units flagged for at least one of ``parameters`` (default: all).
```

## `align_weights(W, units: 'Sequence', W_ids: 'Optional[Sequence]' = None, assume_ordered: 'bool' = False) -> 'np.ndarray'`

```text
Return ``W`` as an array whose rows and columns follow ``units``.

Parameters
----------
W : pandas.DataFrame or array_like
    Weights matrix. A ``DataFrame`` must carry unit labels on its index
    and columns. An array must be accompanied by ``W_ids``.
units : sequence
    Panel unit identifiers in the order used internally.
W_ids : sequence, optional
    Unit identifiers corresponding to the rows (and columns) of an
    unlabelled array ``W``.
assume_ordered : bool
    Accept an unlabelled array and assume it is ordered like ``units``.
    Discouraged; provided for compatibility with other software.
```

## `read_weights(path: 'str', ids_column: 'Optional[str]' = None, **kwargs) -> 'pd.DataFrame'`

```text
Read a labelled spatial weights matrix from a CSV or Excel file.

The file must contain unit identifiers as the first column (or in
``ids_column``) and as the header row, so that the resulting
``DataFrame`` carries labels on both axes.
```

## `prepare_panel(data: 'pd.DataFrame', variables: 'Sequence[str]', unit: 'str', time: 'str', W, W_ids=None, assume_ordered=False) -> 'PanelArrays'`

```text
Convert long-format panel data into ``(N, T)`` arrays.
```

## `spatial_lag(W: 'np.ndarray', V: 'np.ndarray', avail: 'np.ndarray', renormalize: 'bool' = True) -> 'np.ndarray'`

```text
Spatial lag ``W_t v_t`` for every period, using available neighbours only.

When some neighbours of unit i are unavailable in period t they are
dropped. With ``renormalize=True`` the remaining weights of row i are
rescaled so that the row keeps its original sum; for a balanced panel
the operator is exactly ``W``.
```

## `simulate_panel(N: 'int' = 100, T: 'int' = 100, psi: 'float' = 0.3, rho: 'float' = 0.4, beta=(1.0, 0.5), n_factors: 'int' = 2, het: 'float' = 0.0, outlier_share: 'float' = 0.0, outlier_shift: 'float' = 3.0, entry_share: 'float' = 0.0, W: 'np.ndarray | None' = None, burn: 'int' = 50, seed=None)`

```text
Simulate a spatial dynamic panel with interactive effects.

Returns ``(data, W, truth)``: a long DataFrame with columns ``id``,
``time``, ``y``, ``x1``, ..., a labelled weights DataFrame, and a dict
with the true parameters (``truth["theta_i"]`` holds unit slopes).
```

## `knn_weights(N: 'int', k: 'int' = 6, seed=None) -> 'np.ndarray'`

```text
Row-standardized k-nearest-neighbour weights for random locations.
```

## Results object (`SpatialDFIVResults`)

Attributes: `params`, `cov`, `bse`, `zvalues`, `pvalues`, `conf_int(level)`, `nobs`, `n_units`, `obs_per_unit`, `n_instruments`, `factors_x`, `factors_u`, `J`, `df_J`, `p_J`, `sigma_f`, `sigma_e`, `rho_factor`, `unit_params`, `unit_se`, `extras` (`first_stage_params`, `unit_cond`).

Methods: `summary()`, `impacts(...)`, `mg_robust(...)`, `mg_influence(...)`.
