# VBSegm2Step — Two‑Step Vertebral Body Segmentation

VBSegm2Step provides automated vertebral body segmentation for CT using a robust two‑step pipeline built on nnU‑Net v2.

## Overview
- Purpose: End‑to‑end vertebra segmentation with per‑vertebra integer labels (background = 0).
- Method: A global whole‑spine model (Task 601) predicts all vertebrae; a local neighbor model (Task 602) iteratively refines labels around an automatically selected anchor with adaptive crop expansion and probability fusion.
- Output: 3D NIfTI labelmap preserving input spacing/origin/direction. See Labels for mapping.

## Features
- Two‑step global → local pipeline for robustness and detail.
- Confidence‑based anchor selection from whole‑spine prediction.
- Iterative neighbor discovery (cranial/caudal) with dynamic crop expansion.
- Probability fusion over crops with conservative decision rules.
- Simple CLI and Python API.

## Installation
- Requirements: Python 3.8+, PyTorch, SimpleITK, nnUNet v2, NumPy, SciPy, tqdm, huggingface‑hub.
- Local development:
  - `uv sync --extra dev`
  - run commands through `uv run ...` from the repository root.
- Standard package install:
  - `pip install .`
  - use this for package-style installs outside the development checkout.

## Model Weights (Hugging Face)
This package uses two trained nnU‑Net v2 models:
- Task 601 (whole spine): https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM
- Task 602 (neighbors): https://huggingface.co/fhofmann/VertebralBodiesCT-Neighbors

Ways to provide models:
- Use the built‑in downloader (Hugging Face Hub) to populate the configured folders:
  - `vbsegm2step downloadmodels`
- Place existing model snapshot roots at the paths in `vbsegm2step/config.py`.
- Override paths with CLI arguments, environment variables, or Python `Config()` attributes (see Configuration).

## Quickstart (CLI)
- Validate model paths and model initialization: `uv run vbsegm2step validate`
- Download models: `uv run vbsegm2step downloadmodels`
- Predict a single file: `uv run vbsegm2step predict -i /path/ct.nii.gz -o /path/out_seg.nii.gz`
- Batch over a directory: `uv run vbsegm2step batch -i /data/ct_scans -o /data/out_segs -p "*.nii.gz"`
- Recursive batch over nested directories: `uv run vbsegm2step batch -i /data/ct_scans -o /data/out_segs -p "**/*.nii.gz"`

## Quickstart (Python)
```python
from pathlib import Path
from vbsegm2step import Config, VBSegm2StepPipeline

# Option A: use default paths from config.py
config = Config()

# Option B: override model paths programmatically (instance attributes)
# config = Config()
# config.PATH_NNUNET601 = Path("/models/Dataset601_VertebralBodies")
# config.PATH_NNUNET602 = Path("/models/Dataset602_VertebralBodiesNeighbors")

pipeline = VBSegm2StepPipeline(config)

in_file = Path("/data/ct.nii.gz")
out_file = Path("/data/ct_vbseg.nii.gz")

# Default: returns segmentation; probabilities=None
seg, prob = pipeline.process_file(in_file, out_file)

# Also return fused probabilities (larger memory footprint)
seg, prob = pipeline.process_file(in_file, out_file, return_probabilities=True)
```

## Input / Output
- Input: 3D CT volumes (`.nii.gz`). Loader re‑orients to RAS internally.
- Output: 3D NIfTI labelmap with integer vertebra labels (background = 0). Spacing/origin/direction are preserved.
- Batch output paths preserve relative input subdirectories when recursive patterns are used.
- Probabilities: The pipeline fuses per‑label probabilities. By default `prob` is `None`; pass `return_probabilities=True` to receive fused probabilities. For optional debugging/QA heatmaps, save a probability channel via `vbsegm2step.io.save_probabilities(prob, channel_idx, path, image_props)`.

## Labels
Mapping (`vbsegm2step.config.Config.VERTEBRA_LABELS`):
- Thoracic: T1..T12 → 1..12 (T13 → 21)
- Lumbar: L1..L6 → 13..18
- Sacrum: 19, Coccyx: 20

Background is 0.
For the released models, these label values are part of the model contract: Task 601 probability channel indices match the anatomical label values. The local fusion canvas can store labels densely internally, but changing `VERTEBRA_LABELS` does not remap pretrained model output channels.

## Configuration
- File: `vbsegm2step/config.py`
- Key settings:
  - `PATH_NNUNET601`, `PATH_NNUNET602`: model snapshot roots containing the expected nnU-Net v2 trainer folder.
  - Device auto‑selection (CUDA if available, else CPU).
  - Padding for initial crops (`PAD_X/Y/Z` in mm) and adaptive crop expansion (`EXPANSION_MM`).
  - Conservative fusion thresholds and volume plausibility checks.
  - Fusion uses Dirichlet smoothing by default in `VertebraCanvas.export_fusion(use_dirichlet=True, alpha_bg=0.2, alpha_vert=0.05)`. The smoothed posterior is computed from accumulated channel evidence plus a larger background pseudo-count, then the conservative top-2 rule (`theta_min=0.40`, `delta=0.15`) decides the final label. Set `use_dirichlet=False` only when explicitly comparing against plain weighted-average fusion.
  - Fusion memory control: `FUSION_SLAB_Z` sets Z‑slab size for streaming fusion (default 16). Lower if you see OOMs on very large volumes.

Changing model locations:
- Default: `downloadmodels`, `validate`, `predict`, and `batch` work with the package defaults under `./models/` when the model snapshots are present there.
- CLI: pass `--model601 /path/to/Dataset601_VertebralBodies` and `--model602 /path/to/Dataset602_VertebralBodiesNeighbors`.
- Environment: set `VBSEGM2STEP_MODEL601` and `VBSEGM2STEP_MODEL602`.
- Python: set instance attributes on `Config()` before building the pipeline (see example above). The same config instance is used for model paths, crop padding/expansion, label mapping, canvas dtype, and fusion slab size.

Model path precedence is: CLI arguments, then environment variables, then `Config()` defaults. Model paths point to the dataset model roots that contain `nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth`.

## How It Works (High‑Level)
1. Whole‑spine segmentation (Task 601) predicts vertebrae across the entire scan and estimates per‑label probabilities.
2. The pipeline selects a processable anchor vertebra based on a confidence score favoring typical thoracic/lumbar anchors. Sacrum, coccyx, and T13 are not used as local-refinement anchors.
3. A queue grows cranially and caudally. For each center vertebra, the neighbor model (Task 602) predicts center/above/below within a crop. If predictions touch crop borders, the crop expands adaptively and re‑predicts.
4. Each crop contributes weighted probabilities into a full‑volume canvas. The default fusion path applies Dirichlet smoothing to the accumulated probabilities and then uses a conservative top‑2 decision rule to form the final labelmap.

## Performance Notes
- GPU recommended; CPU fallback works but is slower.
- Memory use scales with volume size and the probability canvas; logs print an estimate at runtime.

## Intended Use and Limitations
- This package and the released model weights are intended for research use.
- They are not intended for clinical decision-making or unsupervised clinical deployment.
- Performance may degrade on uncommon anatomy, severe pathology, unusual acquisition protocols, or scans outside the released models' training distribution.

## Development
- Run the synthetic regression suite with `uv run pytest -q`.
- Model-dependent inference checks are separate from the default tests and require local nnU-Net weights.

## Troubleshooting
- “Model path does not exist”: Update `PATH_NNUNET601/602` in `vbsegm2step/config.py` or run `vbsegm2step downloadmodels`.
- “nnU‑Net predictor not initialized”: Ensure nnU‑Net v2 is installed and the model roots contain `nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth`.
- Input orientation/spacing: SimpleITK loader casts to float32, re‑orients to RAS, and preserves metadata on save.

## License
- Code in this repository: MIT (see `LICENSE` and `pyproject.toml`).
- Model weights: distributed separately on Hugging Face under CC BY-SA 4.0; use them under the terms stated on each model page.
- VertebralBodiesCT-Labels dataset: distributed separately on Hugging Face under CC BY-SA 4.0.
- Source images and upstream labels: not included in this repository; follow the terms, approvals, and citation requirements of the relevant TotalSegmentator, VerSe, and other source data.
- Dependencies: retain their own licenses. In particular, nnU-Net v2 must be cited and used according to its upstream terms.

## Citation
If you use this software, cite this repository using `CITATION.cff`. A DOI is pending and should be added here once available.

Please also cite separately:
- Task 601 model weights: https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM
- Task 602 model weights: https://huggingface.co/fhofmann/VertebralBodiesCT-Neighbors
- VertebralBodiesCT-Labels dataset: https://huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels
- nnU-Net v2 according to its upstream citation instructions,
- any datasets or source cohorts according to their respective requirements.
