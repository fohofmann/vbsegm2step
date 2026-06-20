"""
VBSegm2Step: Vertebral Body Segmentation Pipeline

A pipeline for automated vertebral body segmentation using dual nnU-Net models:
- VertebralBodiesCT (Task 601): Whole spine segmentation
- VertebralBodiesCT-Neighbors (Task 602): Iterative neighbor prediction
"""

__version__ = "0.0.2"
__author__ = "Felix O. Hofmann"

from .config import Config
from .segmentation import SpinePredictor, LocalPredictor
from .vertebrae import VertebraInfo, VertebraCanvas, VertebraQueue
from .pipeline import VBSegm2StepPipeline
from .io import load_image, save_segmentation, save_probabilities

__all__ = [
    "Config",
    "SpinePredictor", 
    "LocalPredictor",
    "VertebraInfo",
    "VertebraCanvas", 
    "VertebraQueue",
    "VBSegm2StepPipeline",
    "load_image",
    "save_segmentation",
    "save_probabilities",
]
