"""Segmentation models for VBSegm2Step."""

from .spine import SpinePredictor
from .local import LocalPredictor

__all__ = [
    "SpinePredictor",
    "LocalPredictor",
]
