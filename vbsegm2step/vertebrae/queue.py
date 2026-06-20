from collections import deque
from typing import List, Optional, Tuple
import numpy as np
from ..config import Config

from ..utils import extract_largest_component
from .vertebra import VertebraInfo


class VertebraQueue:
    """Queue-based vertebra processing for iterative neighbor prediction."""
    
    def __init__(self, image_props: dict, vertebra_first: VertebraInfo,
                 vertebra_all: Optional[List[int]] = None,
                 config: Optional[Config] = None):
        """Initialize vertebra queue."""
        self.config = config or getattr(vertebra_first, "config", None) or Config()
        self.image_props = image_props
        self.image_size = image_props['size']
        self.image_shape = image_props['shape']

        self.queue = deque()
        self.tracker_all = vertebra_all or self._default_tracker_all()
        
        if vertebra_first.label in self.tracker_all:
            self.queue.append(vertebra_first)
            print(f"  Queue: Starting pipeline from {vertebra_first.label}")
        else:
            print(f"⚠ Queue: {vertebra_first.label} not in tracking list {self.tracker_all}")

        self.tracker_done_z = []
        self.tracker_done = []
        self.settings_counter_max = len(self.tracker_all)

    def _default_tracker_all(self) -> List[int]:
        return self.config.processable_vertebra_labels()

    @staticmethod
    def _get_median_position(segmentation: np.ndarray, label: int, axis: int = 0) -> Optional[int]:
        """Get median position for a given label along specified axis."""
        if segmentation is None or not isinstance(segmentation, np.ndarray) or segmentation.ndim != 3:
            return None
        coords = np.where(segmentation == label)[axis]
        return int(np.median(coords)) if len(coords) > 0 else None

    @property
    def i_label(self) -> Optional[int]:
        """Get current vertebra label."""
        return self.queue[0].label if self.queue else None

    @property
    def queue_size(self) -> int:
        """Get number of vertebrae in queue."""
        return len(self.queue)

    @property
    def tracker_queue(self) -> List[int]:
        """Get labels currently in queue."""
        return [vertebra.label for vertebra in self.queue]

    def is_processed_or_queued(self, label: int) -> bool:
        """Check if label is already processed or in queue."""
        return label in self.tracker_done or label in self.tracker_queue

    def __bool__(self) -> bool:
        """Check if queue processing should continue."""
        return (len(self.queue) > 0 and len(self.tracker_done) < self.settings_counter_max)

    def get_next(self) -> Optional[VertebraInfo]:
        """Get next vertebra for processing."""
        if not self.queue:
            return None
        i_vertebra = self.queue[0]
        print(f"  Processing vertebra {i_vertebra.label}")
        return i_vertebra

    def get_neighbors(self) -> Tuple[Optional[int], Optional[int]]:
        """Get cranial and caudal neighbors of current vertebra."""
        i_label_idx = self.tracker_all.index(self.i_label)
        cranial_neighbor = self.tracker_all[i_label_idx - 1] if i_label_idx > 0 else None
        caudal_neighbor = self.tracker_all[i_label_idx + 1] if i_label_idx < len(self.tracker_all) - 1 else None
        return cranial_neighbor, caudal_neighbor

    @property
    def labelmap(self) -> dict:
        """Get labelmap for current vertebra and neighbors."""
        if self.i_label not in self.tracker_all:
            raise ValueError(f"Invalid label: {self.i_label}. Must be one of {self.tracker_all}.")
        cranial_neighbor, caudal_neighbor = self.get_neighbors()

        # Task 602 emits local channels: 0=background, 2=center, 3=cranial,
        # 4=caudal. At anatomical boundaries absent neighbor channels are
        # omitted, not mapped to background, so stray votes cannot reinforce 0.
        labelmap = {
            0: 0,  # background
            2: self.i_label,  # center vertebra
        }
        if cranial_neighbor is not None:
            labelmap[3] = cranial_neighbor  # above
        if caudal_neighbor is not None:
            labelmap[4] = caudal_neighbor  # below
        return labelmap

    def process_segmentation(self, segmentation: np.ndarray, bbox: Optional[dict] = None) -> bool:
        """Validate a local segmentation and update queue state.

        On success this may enqueue newly discovered cranial/caudal neighbors
        and records the current vertebra in `tracker_done` / `tracker_done_z`.
        """

        # 1. Valid Array
        if segmentation is None or not isinstance(segmentation, np.ndarray):
            print(f"❌ {self.i_label}: Invalid segmentation array")
            return False

        # 2. Shape
        if bbox is None:
            x0, x1 = 0, self.image_size[0]
            y0, y1 = 0, self.image_size[1]
            z0, z1 = 0, self.image_size[2]
        else:
            z0, z1 = bbox['z0'], bbox['z1']
            y0, y1 = bbox['y0'], bbox['y1'] 
            x0, x1 = bbox['x0'], bbox['x1']
        expected_shape = (z1-z0, y1-y0, x1-x0)  # numpy order: z,y,x
        if segmentation.shape != expected_shape:
            print(f"❌ {self.i_label}: Shape mismatch {segmentation.shape} vs expected {expected_shape}")
            return False

        # 3. Internal Monotonicity: z position caudal < center < cranial
        # Orientation: caudal = 0, cranial = +++
        # Reframe crop predictions into full-image coordinates before comparing
        # with already processed vertebrae from other crops.
        segmentation_abs = np.zeros(self.image_shape, dtype=np.uint8)
        segmentation_abs[z0:z1, y0:y1, x0:x1] = segmentation.copy() # reframe segmentation
        segmentation_abs[segmentation_abs == 1] = 0 # ignore "any"

        z_center = self._get_median_position(segmentation_abs, 2, axis=0)
        z_cranial = self._get_median_position(segmentation_abs, 3, axis=0)
        z_caudal = self._get_median_position(segmentation_abs, 4, axis=0)

        if extract_largest_component(segmentation_abs, target_label=2) is None:
            print(f"❌ {self.i_label}: Missing center vertebra label 2")
            return False

        if z_center is not None and z_cranial is not None:
            if z_cranial < z_center:
                print(f"❌ Internal Monotonicity Error: Current cranial vertebra (label 3) below center (label 2).")
                return False
        if z_center is not None and z_caudal is not None:
            if z_caudal > z_center:
                print(f"❌ Internal Monotonicity Error: Current caudal vertebra (label 4) above center (label 2).")
                return False

        # 4. External Monotonicity: z position center above / below neighbor
        cranial_neighbor, caudal_neighbor = self.get_neighbors()
        if cranial_neighbor in self.tracker_done:
            cranial_neighbor_idx = self.tracker_done.index(cranial_neighbor)
            cranial_neighbor_z = self.tracker_done_z[cranial_neighbor_idx]
            if cranial_neighbor_z is not None and cranial_neighbor_z < z_center:
                print(f"❌ External Monotonicity Error: Already processed cranial neighbor (center) below current center.")
                return False
        if caudal_neighbor in self.tracker_done:
            caudal_neighbor_idx = self.tracker_done.index(caudal_neighbor)
            caudal_neighbor_z = self.tracker_done_z[caudal_neighbor_idx]
            if caudal_neighbor_z is not None and caudal_neighbor_z > z_center:
                print(f"❌ External Monotonicity Error: Already processed caudal neighbor (center) above current center.")
                return False
            
        print(f"  ✓ {self.i_label}: Segmentation passed validation checks")

        # 5. get unprocessed neighbors
        cranial_neighbor, caudal_neighbor = self.get_neighbors()
        if self.is_processed_or_queued(cranial_neighbor):
            cranial_neighbor = None
        if self.is_processed_or_queued(caudal_neighbor):
            caudal_neighbor = None

        # 6. check if segmentation includes vertebra
        if cranial_neighbor is not None and np.any(segmentation == 3):
            n_vertebra = VertebraInfo(label=cranial_neighbor,
                                      segmentation=extract_largest_component(segmentation, target_label=3),
                                      image_props=self.image_props,
                                      bbox=bbox,
                                      meta={'source': 'nnUNet602'},
                                      config=self.config)
            self.queue.append(n_vertebra)
            print(f"  → Found new vertebra {n_vertebra.label} in cranial direction")

        if caudal_neighbor is not None and np.any(segmentation == 4):
            n_vertebra = VertebraInfo(label=caudal_neighbor,
                                      segmentation=extract_largest_component(segmentation, target_label=4),
                                      image_props=self.image_props,
                                      bbox=bbox,
                                      meta={'source': 'nnUNet602'},
                                      config=self.config)
            self.queue.append(n_vertebra)
            print(f"  → Found new vertebra {n_vertebra.label} in caudal direction")

        # 7. tracking
        self.tracker_done.append(self.i_label)
        self.tracker_done_z.append(z_center)
        return True

    def track(self):
        """Mark current vertebra as finished and remove from queue."""
        print(f"  ✓ Finished processing vertebra {self.i_label}, {self.queue_size-1} left in queue\n")
        self.queue.popleft()
