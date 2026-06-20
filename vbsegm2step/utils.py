"""Utility functions image processing."""

from pathlib import Path
from typing import Optional
import numpy as np
from scipy.ndimage import label


def nifti_stem(path: Path) -> str:
    """Return the filename stem, treating .nii.gz as one compound suffix."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def nifti_output_path(input_path: Path, output_dir: Path,
                      input_root: Optional[Path] = None) -> Path:
    """Build a .nii.gz output path without duplicating NIfTI suffixes.

    When `input_root` is provided, preserve the input file's relative parent
    directories under `output_dir`. This avoids filename collisions for
    recursive batch inputs.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_root is None:
        output_parent = output_dir
    else:
        try:
            relative_parent = input_path.relative_to(Path(input_root)).parent
        except ValueError:
            relative_parent = Path()
        output_parent = output_dir / relative_parent

    return output_parent / f"{nifti_stem(input_path)}.nii.gz"


def same_path(left: Path, right: Path) -> bool:
    """Compare paths before they necessarily exist."""
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def extract_largest_component(prediction: np.ndarray, target_label: Optional[int] = None) -> Optional[np.ndarray]:
    """Extract largest connected component for target label.
    
    Args:
        prediction: Input prediction array
        target_label: Specific label to extract, if None use prediction as binary mask
        
    Returns:
        Binary mask of largest component or None if no components found
    """
    # Extract binary mask for target label
    if target_label is not None:
        prediction = (prediction == target_label).astype(np.uint8)

    if np.sum(prediction) == 0:
        return None

    # Find connected components using scipy.ndimage.label
    cc_array, num_features = label(prediction)

    if num_features == 0:
        return None

    # Find largest component
    component_sizes = np.bincount(cc_array.flat)[1:]  # Skip background
    if len(component_sizes) == 0:
        return None

    largest_component_id = np.argmax(component_sizes) + 1
    largest_mask = (cc_array == largest_component_id).astype(np.uint8)

    return largest_mask
