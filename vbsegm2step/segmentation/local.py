
"""nnU-Net Task 602 predictor for neighbor segmentation."""

from typing import Optional, Tuple
import numpy as np
import traceback

from ..config import Config
from ..vertebrae.vertebra import VertebraInfo


class LocalPredictor:
    """Wrapper for nnU-Net Task 602 - Neighbor segmentation."""
    
    def __init__(self, config: Config):
        """Initialize nnU-Net Task 602 predictor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.predictor = None
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Initialize the nnU-Net predictor."""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            print("Initializing nnU-Net Task 602 - Neighbor segmentation")

            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=self.config.DEVICE,
                verbose=self.config.NNUNET_VERBOSE,
                verbose_preprocessing=self.config.NNUNET_VERBOSE,
                allow_tqdm=self.config.NNUNET_TQDM
            )
            
            self.predictor.initialize_from_trained_model_folder(
                str(self.config.nnunet602_trainer_path()),
                use_folds='all',
                checkpoint_name='checkpoint_final.pth',
            )

            print("✓ initialization completed")

        except Exception as e:
            print(f"❌ Failed to initialize nnU-Net Task 602 - Neighbor segmentation: {e}, traceback: {traceback.format_exc()}")
            self.predictor = None
    
    def predict(self, image_0000_np: np.ndarray, image_0001_np: np.ndarray, 
                image_props: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run nnU-Net Task 602 prediction to get neighbor segmentation.
        
        Args:
            image_0000_np: CT image crop as numpy array (z,y,x)
            image_0001_np: Center vertebra mask as numpy array (z,y,x)
            image_props: Image properties including spacing
            
        Returns:
            Tuple of (segmentation, probabilities) or (None, None) if failed
        """
        if self.predictor is None:
            print("❌ nnU-Net Task 602 predictor not initialized")
            return None, None

        # Stack channels: Channel 0 = CT, Channel 1 = Center mask
        image_np = np.stack([image_0000_np, image_0001_np], axis=0)

        try:
            print("  → Running nnU-Net Task 602 prediction")

            # Predict
            # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
            segmentation, probabilities = self.predictor.predict_single_npy_array(
                input_image=image_np,
                image_properties={'spacing': tuple(image_props['spacing'][::-1])}, # expects (z,y,x)
                save_or_return_probabilities=True
            )

            # Region-based nnU-Net returns "any vertebra" plus atomic regions.
            # Fusion expects an explicit background channel and local channels:
            # 0=background, 1=any, 2=center, 3=cranial, 4=caudal.
            probabilities_bg = 1 - probabilities[0]  # 1-any
            probabilities = np.stack([
                probabilities_bg, probabilities[0], probabilities[1], 
                probabilities[2], probabilities[3]
            ], axis=0)

            # Reduce memory: keep probabilities in float16
            if probabilities is not None and probabilities.dtype != np.float16:
                probabilities = probabilities.astype(np.float16, copy=False)

            print(f"  ✓ nnU-Net Task 602 completed. Found labels: {np.unique(segmentation)}")
            return segmentation, probabilities

        except Exception as e:
            print(f"❌ nnU-Net Task 602 prediction failed: {e}, traceback: {traceback.format_exc()}")
            return None, None


    def check_expansion_required(self, segmentation: np.ndarray, bbox: dict, 
                                image_props: dict, margin: Optional[int] = None) -> Tuple[bool, dict]:
        """Check if bounding box expansion is required."""
        if margin is None:
            margin = self.config.BORDER_MARGIN
        size = image_props['size']
        
        # Only labels center=2, above=3, below=4 are relevant
        segmentation = segmentation > 1
        
        # Check borders touched
        borders_touched = [
            segmentation[:, :, :margin].any(),   # x0
            segmentation[:, :, -margin:].any(),  # x1
            segmentation[:, :margin, :].any(),   # y0
            segmentation[:, -margin:, :].any(),  # y1
            segmentation[:margin, :, :].any(),   # z0
            segmentation[-margin:, :, :].any(),  # z1
        ]
        
        # Check borders expandable
        borders_expandable = [
            bbox['x0'] > 0,         # x0
            bbox['x1'] < size[0],   # x1
            bbox['y0'] > 0,         # y0
            bbox['y1'] < size[1],   # y1
            bbox['z0'] > 0,         # z0
            bbox['z1'] < size[2],   # z1
        ]
        
        # Combine checks
        expand = {dim: bool(borders_touched[i] and borders_expandable[i])
                  for i, dim in enumerate(['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])}
        
        if not any(expand.values()):
            print("  → No expansion required")
            return False, expand
        else:
            print("  → Expansion required in direction:", [k for k, v in expand.items() if v])
            return True, expand

    def apply_bbox_to_array(self, img: np.ndarray, bbox: dict) -> np.ndarray:
        """Apply bounding box to extract crop from array."""
        return img[bbox['z0']:bbox['z1'], bbox['y0']:bbox['y1'], bbox['x0']:bbox['x1']]

    def predict_with_expansion(self, image_np: np.ndarray, image_props: dict,
                               vertebra: VertebraInfo) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Predict neighbors with adaptive expansion."""
        bbox = vertebra.bbox  # Initialize bounding box
        image_0000_np = self.apply_bbox_to_array(image_np, bbox)
        image_0001_np = vertebra.marker  # bbox integrated, binary mask
        
        expansion_count = 0
        while True:
            # Run segmentation
            segmentation, probabilities = self.predict(
                image_0000_np, image_0001_np, image_props
            )
            
            if segmentation is None:
                break
            
            # Check if expansion needed
            expansion_required, expansion_direction = self.check_expansion_required(
                segmentation, bbox, image_props
            )
            
            if not expansion_required:
                break

            if expansion_count >= self.config.MAX_EXPANSION_ITERATIONS:
                # Stop before mutating bbox again so returned arrays still
                # describe exactly the returned crop coordinates.
                print("  → Expansion limit reached")
                break
            
            # Update for next iteration
            bbox = vertebra.bbox_expand(direction=expansion_direction)
            image_0000_np = self.apply_bbox_to_array(image_np, bbox)
            image_0001_np = vertebra.marker
            
            expansion_count += 1
        
        return segmentation, probabilities, bbox
