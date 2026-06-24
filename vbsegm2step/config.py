"""Configuration settings for VBSegm2Step pipeline."""

import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np


class Config:
    """Configuration class for VBSegm2Step pipeline."""

    MODEL601_VARIANT_ENV = "VBSEGM2STEP_MODEL601_VARIANT"
    MODEL601_VARIANTS = {
        "ResEncL": {
            "path": Path("./models/Dataset601_VertebralBodies_ResEncL/"),
            "hf": "fhofmann/VertebralBodiesCT-ResEncL",
            "trainer": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
            "folds": (0, 1, 2, 3, 4),
        },
        "ResEncM": {
            "path": Path("./models/Dataset601_VertebralBodies/"),
            "hf": "fhofmann/VertebralBodiesCT-ResEncM",
            "trainer": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
            "folds": "all",
        },
    }
    DEFAULT_MODEL601_VARIANT = "ResEncL"
    MODEL601_VARIANT = DEFAULT_MODEL601_VARIANT

    # Model paths - these should be updated for your system
    PATH_NNUNET601 = MODEL601_VARIANTS[DEFAULT_MODEL601_VARIANT]["path"]
    PATH_NNUNET602 = Path("./models/Dataset602_VertebralBodiesNeighbors/")
    HF_NNUNET601 = MODEL601_VARIANTS[DEFAULT_MODEL601_VARIANT]["hf"]
    HF_NNUNET602 = "fhofmann/VertebralBodiesCT-Neighbors"
    MODEL601_ENV = "VBSEGM2STEP_MODEL601"
    MODEL602_ENV = "VBSEGM2STEP_MODEL602"
    TRAINER_FOLDER_NNUNET601 = MODEL601_VARIANTS[DEFAULT_MODEL601_VARIANT]["trainer"]
    TRAINER_FOLDER_NNUNET602 = "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
    TRAINER_FOLDER = TRAINER_FOLDER_NNUNET602
    FOLDS_NNUNET601: Union[str, Tuple[int, ...]] = MODEL601_VARIANTS[DEFAULT_MODEL601_VARIANT]["folds"]
    FOLDS_NNUNET602: Union[str, Tuple[int, ...]] = "all"

    # nnUNet verbose
    NNUNET_VERBOSE = False
    NNUNET_TQDM = False

    # Padding settings (from q95 analysis)
    PAD_X = 8   # mm
    PAD_Y = 22  # mm  
    PAD_Z = 76  # mm
    
    # Border expansion settings
    EXPANSION_MM = 10  # mm for adaptive expansion
    MAX_EXPANSION_ITERATIONS = 6
    BORDER_MARGIN = 1  # voxels
    
    # Released-model label contract: Task 601 probability channels use these
    # anatomical integer labels directly. Do not change this to "renumber"
    # pretrained model outputs; use a different config only with matching models.
    VERTEBRA_LABELS: Dict[str, int] = {
        'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5': 5, 'T6': 6,
        'T7': 7, 'T8': 8, 'T9': 9, 'T10': 10, 'T11': 11, 'T12': 12,
        'L1': 13, 'L2': 14, 'L3': 15, 'L4': 16, 'L5': 17, 'L6': 18,
        'sacrum': 19, 'coccyx': 20, 'T13': 21
    }
    
    # Volume validation in cm³ for plausibility check
    MIN_VERTEBRA_VOLUME_CM3 = 2 * 2 * 2  # cm³
    MAX_VERTEBRA_VOLUME_CM3 = 6 * 6 * 6  # cm³
    
    # Anatomical regions
    VERTEBRAE_THORAX: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21]
    VERTEBRAE_LUMBAR: List[int] = [13, 14, 15, 16, 17, 18]

    # Preferred anchor regions
    VERTEBRAE_THORAX_PREFERRED: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # T2-T11
    VERTEBRAE_LUMBAR_PREFERRED: List[int] = [14, 15, 16]  # L2-L4
    
    # Fusion memory settings
    # Z-slab size for streaming fusion in `VertebraCanvas.export_fusion`
    # Lower this if you hit RAM limits on very large volumes.
    FUSION_SLAB_Z = 16

    # Canvas dtype for accumulation buffers. Using float16 halves RAM usage.
    # Valid options: np.float32, np.float16
    CANVAS_DTYPE = np.float16

    # Device configuration
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        DEVICE = torch.device('cuda')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    else:
        torch.set_num_threads(multiprocessing.cpu_count())
        DEVICE = torch.device('cpu')
    
    @staticmethod
    def volume_voxels_to_cm3(volume_voxels: int, spacing: tuple) -> float:
        """Convert voxel volume to cm³."""
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]  # mm³
        volume_mm3 = volume_voxels * voxel_volume_mm3
        volume_cm3 = volume_mm3 / 1000
        return volume_cm3
    
    def is_valid_vertebra_volume(self, volume_voxels: int, spacing: tuple) -> bool:
        """Check if vertebra volume is within valid range."""
        volume_cm3 = self.volume_voxels_to_cm3(volume_voxels, spacing)
        return self.MIN_VERTEBRA_VOLUME_CM3 <= volume_cm3 <= self.MAX_VERTEBRA_VOLUME_CM3

    def processable_vertebra_labels(self) -> List[int]:
        """Return labels that can be processed by the neighbor-refinement queue."""
        # Task 602 is trained for vertebral-body neighbor walking, so terminal or
        # ambiguous anatomy outside that contract must not be selected as anchors.
        excluded = {
            self.VERTEBRA_LABELS.get('sacrum'),
            self.VERTEBRA_LABELS.get('coccyx'),
            self.VERTEBRA_LABELS.get('T13'),
        }
        return [
            int(label)
            for label in self.VERTEBRA_LABELS.values()
            if label not in excluded
        ]
    
    def configure_model601_variant(self, variant: str) -> None:
        """Select the released Task 601 model variant and matching nnU-Net layout."""
        if variant not in self.MODEL601_VARIANTS:
            valid = ", ".join(sorted(self.MODEL601_VARIANTS))
            raise ValueError(f"Unsupported model601 variant {variant!r}. Choose one of: {valid}.")

        model = self.MODEL601_VARIANTS[variant]
        self.MODEL601_VARIANT = variant
        self.PATH_NNUNET601 = model["path"]
        self.HF_NNUNET601 = model["hf"]
        self.TRAINER_FOLDER_NNUNET601 = model["trainer"]
        self.FOLDS_NNUNET601 = model["folds"]

    @classmethod
    def from_overrides(cls, model601: Optional[Path] = None,
                       model602: Optional[Path] = None,
                       model601_variant: Optional[str] = None) -> "Config":
        """Create config from defaults, environment variables, and CLI overrides."""
        config = cls()
        config.configure_model601_variant(cls.DEFAULT_MODEL601_VARIANT)
        env_model601_variant = os.environ.get(cls.MODEL601_VARIANT_ENV)
        env_model601 = os.environ.get(cls.MODEL601_ENV)
        env_model602 = os.environ.get(cls.MODEL602_ENV)

        if env_model601_variant:
            config.configure_model601_variant(env_model601_variant)
        if model601_variant is not None:
            config.configure_model601_variant(model601_variant)
        if env_model601:
            config.PATH_NNUNET601 = Path(env_model601)
        if env_model602:
            config.PATH_NNUNET602 = Path(env_model602)
        if model601 is not None:
            config.PATH_NNUNET601 = Path(model601)
        if model602 is not None:
            config.PATH_NNUNET602 = Path(model602)
        return config

    def nnunet601_trainer_path(self) -> Path:
        """Return expected nnU-Net 601 trainer folder under the model root."""
        return Path(self.PATH_NNUNET601) / self.TRAINER_FOLDER_NNUNET601

    def nnunet602_trainer_path(self) -> Path:
        """Return expected nnU-Net 602 trainer folder under the model root."""
        return Path(self.PATH_NNUNET602) / self.TRAINER_FOLDER_NNUNET602

    def _fold_dirs(self, folds: Union[str, Tuple[int, ...]]) -> List[str]:
        if folds == "all":
            return ["fold_all"]
        return [f"fold_{fold}" for fold in folds]

    def _validate_model_root(self, title: str, root: Path, trainer_path: Path,
                             folds: Union[str, Tuple[int, ...]]) -> bool:
        ok = True
        if not root.exists():
            print(f"❌ {title} model root does not exist: {root}")
            ok = False
        if not trainer_path.exists():
            print(f"❌ {title} trainer folder does not exist: {trainer_path}")
            ok = False
        for metadata_file in ("plans.json", "dataset.json"):
            metadata_path = trainer_path / metadata_file
            if not metadata_path.exists():
                print(f"❌ {title} metadata file does not exist: {metadata_path}")
                ok = False
        for fold_dir in self._fold_dirs(folds):
            checkpoint_path = trainer_path / fold_dir / "checkpoint_final.pth"
            if not checkpoint_path.exists():
                print(f"❌ {title} checkpoint does not exist: {checkpoint_path}")
                ok = False
        return ok

    def validate_model_paths(self) -> bool:
        """Validate expected nnU-Net model roots and checkpoint files."""
        nnunet601_ok = self._validate_model_root(
            "nnU-Net 601",
            Path(self.PATH_NNUNET601),
            self.nnunet601_trainer_path(),
            self.FOLDS_NNUNET601,
        )
        nnunet602_ok = self._validate_model_root(
            "nnU-Net 602",
            Path(self.PATH_NNUNET602),
            self.nnunet602_trainer_path(),
            self.FOLDS_NNUNET602,
        )

        return nnunet601_ok and nnunet602_ok

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"Configuration:\n"
            f"  Device: {self.DEVICE}\n"
            f"  Volume range: {self.MIN_VERTEBRA_VOLUME_CM3}-{self.MAX_VERTEBRA_VOLUME_CM3} cm³\n"
            f"  Border expansion: {self.EXPANSION_MM}mm, max {self.MAX_EXPANSION_ITERATIONS} iterations\n"
            f"  Padding: {self.PAD_X}/{self.PAD_Y}/{self.PAD_Z}mm (x/y/z)\n"
            f"  nnUNet601: {self.PATH_NNUNET601} ({self.MODEL601_VARIANT}, folds={self.FOLDS_NNUNET601})\n"
            f"  nnUNet602: {self.PATH_NNUNET602}\n"
        )
