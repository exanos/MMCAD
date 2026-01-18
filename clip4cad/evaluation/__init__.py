"""
Evaluation utilities for CLIP4CAD.
"""

from .retrieval import (
    compute_retrieval_metrics,
    text_to_geometry_retrieval,
    geometry_to_text_retrieval,
    cross_modal_retrieval,
)
from .rotation_robustness import (
    evaluate_rotation_robustness,
    compute_rotation_invariance_score,
)

__all__ = [
    "compute_retrieval_metrics",
    "text_to_geometry_retrieval",
    "geometry_to_text_retrieval",
    "cross_modal_retrieval",
    "evaluate_rotation_robustness",
    "compute_rotation_invariance_score",
]
