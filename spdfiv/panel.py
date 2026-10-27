"""Panel data preparation and safe alignment of the spatial weights matrix.

The functions in this module turn a long-format ``pandas.DataFrame`` into
dense ``(N, T)`` arrays and align the spatial weights matrix with the panel
units by *label*, never by position.  Misaligned weights matrices are a
silent and common source of error in applied spatial econometrics; here a
mismatch is always an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


class WeightsAlignmentError(ValueError):
    """Raised when the spatial weights matrix cannot be matched to the panel."""


@dataclass
class PanelArrays:
    units: list
    times: list
    arrays: dict          # variable name -> (N, T) float array (nan = missing)
    mask: np.ndarray      # (N, T) bool, True if every requested variable observed
    W: np.ndarray         # (N, N) weights, rows/cols ordered as ``units``


def read_weights(path: str, ids_column: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Read a labelled spatial weights matrix from a CSV or Excel file.

    The file must contain unit identifiers as the first column (or in
    ``ids_column``) and as the header row, so that the resulting
    ``DataFrame`` carries labels on both axes.
    """
    if str(path).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    key = ids_column if ids_column is not None else df.columns[0]
    df = df.set_index(key)
    df.columns = _coerce_labels(df.columns, df.index)
    return df.astype(float)


def _coerce_labels(columns, index):
    """Header labels are read as strings; convert them to the index dtype."""
    try:
        return pd.Index(columns).astype(pd.Index(index).dtype)
    except (TypeError, ValueError):
        return pd.Index(columns)


def align_weights(W, units: Sequence, W_ids: Optional[Sequence] = None,
                  assume_ordered: bool = False) -> np.ndarray:
    """Return ``W`` as an array whose rows and columns follow ``units``.

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
    """
    units = list(units)
    if isinstance(W, pd.DataFrame):
        idx, cols = list(W.index), list(W.columns)
        if set(idx) != set(cols):
            raise WeightsAlignmentError(
                "Row and column labels of the weights matrix differ.")
        if len(set(idx)) != len(idx):
            raise WeightsAlignmentError("Duplicate unit labels in the weights matrix.")
        missing_in_W = [u for u in units if u not in set(idx)]
        extra_in_W = [u for u in idx if u not in set(units)]
        if missing_in_W or extra_in_W:
            raise WeightsAlignmentError(
                "Units in the data and in the weights matrix do not match. "
                f"In data but not in W: {missing_in_W[:5]}"
                f"{' ...' if len(missing_in_W) > 5 else ''}; "
                f"in W but not in data: {extra_in_W[:5]}"
                f"{' ...' if len(extra_in_W) > 5 else ''}.")
        Wa = W.loc[units, units].to_numpy(dtype=float)
    else:
        Wa = np.asarray(W, dtype=float)
        if Wa.ndim != 2 or Wa.shape[0] != Wa.shape[1]:
            raise WeightsAlignmentError("The weights matrix must be square.")
        if W_ids is not None:
            W_ids = list(W_ids)
            if len(W_ids) != Wa.shape[0]:
                raise WeightsAlignmentError("len(W_ids) must equal the dimension of W.")
            return align_weights(pd.DataFrame(Wa, index=W_ids, columns=W_ids),
                                 units)
        if not assume_ordered:
            raise WeightsAlignmentError(
                "An unlabelled weights array was supplied. Pass a labelled "
                "DataFrame, or the unit labels via `W_ids`, or set "
                "`assume_ordered=True` to accept positional matching.")
        if Wa.shape[0] != len(units):
            raise WeightsAlignmentError(
                f"W is {Wa.shape[0]}x{Wa.shape[0]} but the panel has {len(units)} units.")
    if np.isnan(Wa).any():
        raise WeightsAlignmentError("The weights matrix contains missing values.")
    if np.any(np.diag(Wa) != 0):
        raise WeightsAlignmentError("All diagonal elements of W must be zero.")
    return Wa


def prepare_panel(data: pd.DataFrame, variables: Sequence[str], unit: str,
                  time: str, W, W_ids=None, assume_ordered=False) -> PanelArrays:
    """Convert long-format panel data into ``(N, T)`` arrays."""
    for v in list(variables) + [unit, time]:
        if v not in data.columns:
            raise KeyError(f"Variable '{v}' not found in data.")
    if data.duplicated([unit, time]).any():
        raise ValueError("Duplicate (unit, time) pairs in the data.")

    if isinstance(W, pd.DataFrame):
        units = list(W.index)
    elif W_ids is not None:
        units = list(W_ids)
    else:
        units = sorted(data[unit].unique())
    data_units = set(data[unit].unique())
    if isinstance(W, pd.DataFrame) or W_ids is not None:
        absent = [u for u in units if u not in data_units]
        extra = [u for u in data_units if u not in set(units)]
        if absent or extra:
            raise WeightsAlignmentError(
                "Units in the data and in the weights matrix do not match. "
                f"In W but never in data: {absent[:5]}; in data but not in W: {extra[:5]}.")
    times = sorted(data[time].unique())
    Wa = align_weights(W, units, W_ids=W_ids, assume_ordered=assume_ordered)

    u_pos = pd.Index(units).get_indexer(data[unit])
    t_pos = pd.Index(times).get_indexer(data[time])
    N, T = len(units), len(times)
    arrays = {}
    mask = np.ones((N, T), dtype=bool)
    for v in variables:
        a = np.full((N, T), np.nan)
        a[u_pos, t_pos] = data[v].to_numpy(dtype=float)
        arrays[v] = a
        mask &= ~np.isnan(a)
    return PanelArrays(units=units, times=times, arrays=arrays, mask=mask, W=Wa)
