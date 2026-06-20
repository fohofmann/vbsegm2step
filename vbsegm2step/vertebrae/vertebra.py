from typing import Union, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation

from ..config import Config

class VertebraInfo:
    """Class to store information about a single vertebra."""
    
    def __init__(self, label: int, segmentation: np.ndarray, image_props: dict, 
                 bbox: Optional[dict] = None, meta: Optional[dict] = None,
                 config: Optional[Config] = None):
        """Initialize VertebraInfo.
        
        Args:
            label: Vertebra label (integer)
            segmentation: Binary segmentation array (z,y,x)
            image_props: Image properties dict
            bbox: Optional bounding box for crop placement, only used for orientation within full image
            meta: Optional metadata dict
            config: Optional configuration object
        """
        # Validation
        if label is None or not isinstance(label, int):
            raise ValueError("Label must be an integer")
        if image_props is None or not isinstance(image_props, dict):
            raise ValueError("Image properties must be a dictionary")
        if segmentation is None or not isinstance(segmentation, np.ndarray):
            raise ValueError("Segmentation must be a 3D numpy array (z,y,x)")
        if not np.array_equal(segmentation, segmentation.astype(bool)):
            raise ValueError("Segmentation must be binary (0,1)")
        
        self.label = label
        self.image_props = image_props
        self.meta = meta or {}
        self.config = config or Config()
        self._bbox = None
        
        # Handle segmentation placement
        if bbox is None and segmentation.shape == image_props['shape']:
            self._segmentation = segmentation
        elif (bbox is not None and 
              all(k in bbox for k in ('z0','z1','y0','y1','x0','x1')) and 
              segmentation.shape == (bbox['z1']-bbox['z0'], bbox['y1']-bbox['y0'], bbox['x1']-bbox['x0'])):
            self._segmentation = np.zeros(image_props['shape'], dtype=np.uint8)
            self._segmentation[bbox['z0']:bbox['z1'], bbox['y0']:bbox['y1'], bbox['x0']:bbox['x1']] = segmentation
        else:
            raise ValueError("Invalid segmentation shape or bbox configuration")

    def bbox_initialize(self) -> Optional[dict]:
        """Initialize bounding box around segmentation with padding."""
        if not np.any(self._segmentation):
            return None
        
        segmentation = self._segmentation
        spacing = self.image_props['spacing']
        size = self.image_props['size']

        # Get bounding box around segmentation. The max edge is exclusive so
        # bbox values can be used directly in Python slices.
        z_coords, y_coords, x_coords = np.where(segmentation > 0)
        x_min, x_max = np.min(x_coords), np.max(x_coords) + 1
        y_min, y_max = np.min(y_coords), np.max(y_coords) + 1
        z_min, z_max = np.min(z_coords), np.max(z_coords) + 1

        # Convert padding from mm to voxels
        config = self.config
        pad_x_vox = int(np.ceil(config.PAD_X / spacing[0]))
        pad_y_vox = int(np.ceil(config.PAD_Y / spacing[1]))
        pad_z_vox = int(np.ceil(config.PAD_Z / spacing[2]))

        # Add padding, limit to image size
        x_min = max(0, x_min - pad_x_vox)
        x_max = min(size[0], x_max + pad_x_vox)
        y_min = max(0, y_min - pad_y_vox)
        y_max = min(size[1], y_max + pad_y_vox)
        z_min = max(0, z_min - pad_z_vox)
        z_max = min(size[2], z_max + pad_z_vox)

        bbox = {
            'z0': int(z_min), 'z1': int(z_max),
            'y0': int(y_min), 'y1': int(y_max),
            'x0': int(x_min), 'x1': int(x_max)
        }

        print(f"  → {self.label}: Initialized bbox with padding {config.PAD_X}/{config.PAD_Y}/{config.PAD_Z}mm")
        return bbox
    
    def bbox_expand(self, direction: Union[dict, list] = None) -> dict:
        """Expand bounding box in specified directions."""
        if direction is None or not isinstance(direction, (dict, list)):
            raise ValueError("Direction must be provided as dict or list")
        if isinstance(direction, list):
            direction = {d: True for d in direction}

        spacing = self.image_props['spacing']
        size = self.image_props['size']
        bbox = self.bbox

        # Calculate expansion amounts in voxels
        config = self.config
        pad_x_vox = int(np.ceil(config.EXPANSION_MM / spacing[0]))
        pad_y_vox = int(np.ceil(config.EXPANSION_MM / spacing[1]))
        pad_z_vox = int(np.ceil(config.EXPANSION_MM / spacing[2]))

        # Expand in specified directions
        if direction.get('x0', False):
            bbox['x0'] = max(0, bbox['x0'] - pad_x_vox)
        if direction.get('x1', False):
            bbox['x1'] = min(size[0], bbox['x1'] + pad_x_vox)
        if direction.get('y0', False):
            bbox['y0'] = max(0, bbox['y0'] - pad_y_vox)
        if direction.get('y1', False):
            bbox['y1'] = min(size[1], bbox['y1'] + pad_y_vox)
        if direction.get('z0', False):
            bbox['z0'] = max(0, bbox['z0'] - pad_z_vox)
        if direction.get('z1', False):
            bbox['z1'] = min(size[2], bbox['z1'] + pad_z_vox)

        self._bbox = bbox
        directions = [k for k, v in direction.items() if v]
        print(f"  → {self.label}: Expanded bbox by {config.EXPANSION_MM}mm in {directions}")
        return bbox

    @property
    def bbox(self) -> dict:
        """Get bounding box, initializing if needed."""
        if self._bbox is None:
            self._bbox = self.bbox_initialize()     
        return self._bbox

    @property
    def segmentation(self) -> np.ndarray:
        """Get segmentation array, cropped to bbox if bbox is set."""
        if self._bbox is None:
            return self._segmentation
        else:
            return self._segmentation[self._bbox['z0']:self._bbox['z1'],
                                      self._bbox['y0']:self._bbox['y1'],
                                      self._bbox['x0']:self._bbox['x1']]

    @property
    def center(self) -> Optional[Tuple[int, int, int]]:
        """Get center coordinates of the vertebra."""
        if not np.any(self._segmentation):
            print(f"⚠ {self.label}: No segmentation found for center calculation")
            return None
        coords = np.where(self._segmentation > 0)
        if len(coords[0]) == 0:
            return None
        center_z = int(round(np.mean(coords[0])))
        center_y = int(round(np.mean(coords[1]))) 
        center_x = int(round(np.mean(coords[2])))
        return (center_x, center_y, center_z)

    @property
    def marker(self) -> np.ndarray:
        """Create marker image for 602 input."""

        center_x, center_y, center_z = self.center

        # Create empty array, place marker at center
        marker_np = np.zeros(self.image_props['shape'], dtype=np.uint8)
        marker_np[center_z, center_y, center_x] = 1
        
        # Create marker using binary dilation
        structure = np.ones((3, 3, 3), dtype=bool)
        marker_np = binary_dilation(marker_np, structure=structure).astype(np.uint8)

        print(f"  → {self.label}: Created marker at center ({center_z},{center_y},{center_x})")
        if self._bbox is None:
            return marker_np
        else:
            return marker_np[self._bbox['z0']:self._bbox['z1'],
                             self._bbox['y0']:self._bbox['y1'],
                             self._bbox['x0']:self._bbox['x1']]
