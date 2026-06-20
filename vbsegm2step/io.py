"""Utility functions for file I/O."""

import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import SimpleITK as sitk


def load_directory(input_dir: Union[str, Path], pattern: str = "*.nii.gz") -> List[Path]:
    """Load all CT scans from directory using glob pattern.
    
    Args:
        input_dir: Directory containing input files
        pattern: Glob pattern for file matching. Use patterns such as
            "**/*.nii.gz" to match recursively.
        
    Returns:
        List of Path objects for matching files
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return []
    
    pattern_path = input_dir / pattern
    file_paths = glob.glob(str(pattern_path), recursive=True)
    file_paths = [Path(p) for p in file_paths]
    file_paths.sort()
    
    print(f"✓ Found {len(file_paths)} files")
    return file_paths


def load_image(file_path: Path) -> Tuple[Optional[sitk.Image], Optional[np.ndarray], Optional[dict]]:
    """Load CT scan and return image + properties.
    
    Args:
        file_path: Path to the CT scan file
        
    Returns:
        Tuple of (SimpleITK image, numpy array, properties dict)
    """
    
    try:
        # Load with SimpleITK (keep native orientation metadata)
        image_native = sitk.ReadImage(str(file_path))

        # Reorient to RAS for inference
        image_sitk = sitk.DICOMOrient(image_native, 'RAS')
        image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
        
        # Keep both coordinate systems: inference runs in RAS, while saved
        # outputs must be transformed back to the input scan's native metadata.
        properties = {
            # RAS-oriented properties
            'spacing': image_sitk.GetSpacing(),
            'origin': image_sitk.GetOrigin(), 
            'direction': image_sitk.GetDirection(),
            'size': image_sitk.GetSize(),
            'filename': file_path.name,
    
            # Native properties
            'spacing_native': image_native.GetSpacing(),
            'origin_native': image_native.GetOrigin(),
            'direction_native': image_native.GetDirection(),
            'size_native': image_native.GetSize(),
            'orientation_native': sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                image_native.GetDirection()
            )
        }
        
        # Convert to numpy for nnU-Net (z,y,x order!)
        image_np = sitk.GetArrayFromImage(image_sitk)
        properties['shape'] = image_np.shape

        print(f"  image loaded (RAS): np(z,y,x)={properties['shape']}, "
              f"spacing(x,y,z)={properties['spacing']}, "
              f"size(x,y,z)={properties['size']} ")
        return image_sitk, image_np, properties
        
    except Exception as e:
        print(f"  ❌ Failed to load {file_path.name}: {e}")
        return None, None, None


def save_segmentation(segmentation_np: np.ndarray,
                      output_path: Union[str, Path],
                      image_props: dict) -> None:
    """Save numpy segmentation array as NIfTI file with metadata.
    
    Args:
        segmentation_np: Segmentation as numpy array (z,y,x)
        output_path: Output file path
        image_props: Image properties dict with spacing, origin, direction
    """
    # Convert numpy array to SimpleITK image (RAS orientation used during inference)
    segmentation_sitk = sitk.GetImageFromArray(segmentation_np)
    segmentation_sitk.SetSpacing(image_props['spacing'])
    segmentation_sitk.SetOrigin(image_props['origin'])
    segmentation_sitk.SetDirection(image_props['direction'])

    # Reorient back to native orientation and then restore native metadata.
    # This keeps the saved labelmap spatially aligned with the original file.
    segmentation_sitk = sitk.DICOMOrient(segmentation_sitk, image_props['orientation_native'])
    segmentation_sitk.SetSpacing(image_props.get('spacing_native', image_props['spacing']))
    segmentation_sitk.SetOrigin(image_props.get('origin_native', image_props['origin']))
    segmentation_sitk.SetDirection(image_props.get('direction_native', image_props['direction']))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(segmentation_sitk, str(output_path))
    print(f"  ✓ Saved segmentation: {output_path}")


def save_probabilities(probabilities_np: np.ndarray,
                       channel_idx: int,
                       output_path: Union[str, Path], 
                       image_props: dict) -> None:
    """Save one probability channel as a NIfTI file with metadata.

    This helper is not called by the default pipeline. It is exported for
    optional debugging/QA workflows, for example writing per-label heatmaps to
    inspect the fused probability distribution.
    
    Args:
        probabilities_np: Probabilities as numpy array (channels,z,y,x)
        channel_idx: Channel index to save
        output_path: Output file path
        image_props: Image properties dict with spacing, origin, direction
    """

    if channel_idx < 0 or channel_idx >= probabilities_np.shape[0]:
        print(f"❌ Invalid channel index {channel_idx} for probabilities with shape {probabilities_np.shape}")
        return
    
    # Extract specific channel
    probabilities_np = probabilities_np[channel_idx]

    # Convert numpy array to SimpleITK image (RAS orientation used during inference)
    probabilities_sitk = sitk.GetImageFromArray(probabilities_np)
    probabilities_sitk.SetSpacing(image_props['spacing'])
    probabilities_sitk.SetOrigin(image_props['origin'])
    probabilities_sitk.SetDirection(image_props['direction'])

    # set format to float32
    probabilities_sitk = sitk.Cast(probabilities_sitk, sitk.sitkFloat32)

    # Match segmentation saving: probability channels are stored in the input
    # scan's native orientation, not the internal RAS inference orientation.
    probabilities_sitk = sitk.DICOMOrient(probabilities_sitk, image_props['orientation_native'])
    probabilities_sitk.SetSpacing(image_props.get('spacing_native', image_props['spacing']))
    probabilities_sitk.SetOrigin(image_props.get('origin_native', image_props['origin']))
    probabilities_sitk.SetDirection(image_props.get('direction_native', image_props['direction']))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(probabilities_sitk, str(output_path))
    print(f"✓ Saved probabilities: {output_path}")
    
