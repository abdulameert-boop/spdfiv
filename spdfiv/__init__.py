"""spdfiv: defactored IV estimation of spatial dynamic panels with interactive effects."""

from .model import fit, spatial_lag, SpatialDFIVResults
from .panel import align_weights, read_weights, prepare_panel, WeightsAlignmentError
from .impacts import impacts, StabilityError
from .heterogeneity import mg_robust, mg_influence, flagged_units
from .simulate import simulate_panel, knn_weights

__version__ = "0.1.0"

__all__ = ["fit", "spatial_lag", "SpatialDFIVResults", "align_weights", "read_weights",
           "prepare_panel", "WeightsAlignmentError", "impacts", "StabilityError",
           "mg_robust", "mg_influence", "flagged_units", "simulate_panel",
           "knn_weights", "__version__"]
