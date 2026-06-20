"""nnU-Net model wrappers for nnU-Net Task 601 and nnU-Net Task 602."""

import traceback
from typing import Optional, Tuple
import numpy as np
from ..config import Config
from ..vertebrae.vertebra import VertebraInfo
from ..utils import extract_largest_component

class SpinePredictor:
    """Wrapper for nnU-Net Task 601 - Whole spine segmentation."""
    
    def __init__(self, config: Config):
        """Initialize nnU-Net Task 601 predictor.
        
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

            print("Initializing nnU-Net Task 601 - Whole spine segmentation")

            # Initialize predictor
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
            
            # Load model
            self.predictor.initialize_from_trained_model_folder(
                str(self.config.nnunet601_trainer_path()),
                use_folds='all',
                checkpoint_name='checkpoint_final.pth',
            )
            
            print("✓ initialization completed")
            
        except Exception as e:
            print(f"❌ Failed to initialize nnU-Net Task 601: {e}, traceback: {traceback.format_exc()}")
            self.predictor = None
    
    def predict(self, image_np: np.ndarray, image_props: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run nnUNet Task 601 prediction to get initial vertebra segmentation.

        Args:
            image_np: Input image as numpy array (z,y,x)
            image_props: Image properties including spacing
            
        Returns:
            Tuple of (segmentation, probabilities) or (None, None) if failed
        """
        if self.predictor is None:
            print("❌ nnU-Net Task 601 predictor not initialized")
            return None, None
            
        try:
            # Predict
            # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
            segmentation, probabilities = self.predictor.predict_single_npy_array(
                input_image=image_np[None], # only one channel
                image_properties={'spacing': tuple(image_props['spacing'][::-1])}, # expects (z,y,x)
                save_or_return_probabilities=True
            )
            # Reduce memory: keep probabilities in float16
            if probabilities is not None and probabilities.dtype != np.float16:
                probabilities = probabilities.astype(np.float16, copy=False)
            
            print(f"  nnU-Net Task 601 completed. Found labels: {np.unique(segmentation)}")
            return segmentation, probabilities

        except Exception as e:
            print(f"❌ nnU-Net Task 601 prediction failed: {e}, traceback: {traceback.format_exc()}")
            return None, None

    def calculate_vertebra_confidence(self, segmentation: np.ndarray, probabilities: np.ndarray, 
                                        label: int, image_props: dict) -> Tuple[Optional[VertebraInfo], Optional[float]]:
        """Calculate confidence score for a vertebra."""
        
        # Extract binary mask for this vertebra
        mask_binary = (segmentation == label).astype(np.uint8)
        mask_voxels = np.sum(mask_binary)
        
        if mask_voxels == 0:
            print(f"⚠ {label}: Not found in segmentation")
            return None, None
            
        # Validate volume
        if not self.config.is_valid_vertebra_volume(mask_voxels, image_props['spacing']):
            print(f"⚠ {label}: Invalid volume - {mask_voxels} voxels")
            return None, None
            
        # Task 601 uses anatomical label values as probability channel indices.
        if label >= probabilities.shape[0]:
            print(f"⚠ {label}: Label exceeds probability channels")
            return None, None
        
        # Calculate confidence metrics
        mask_probability = probabilities[label][mask_binary == 1]
        mean_probability = np.mean(mask_probability)
        std_probability = np.std(mask_probability)
        
        high_conf_ratio = (mask_probability > 0.8).sum() / mask_voxels
        
        # Position score
        position_score = 1.0
        if label in self.config.VERTEBRAE_THORAX_PREFERRED:
            position_score = 1.5
        elif label in self.config.VERTEBRAE_LUMBAR_PREFERRED:
            position_score = 1.5
        
        # Comprehensive confidence score
        # Weight: 40% mean probability, 30% high confidence ratio, 20% inverse std deviation, 10% position score
        confidence = (mean_probability * 0.4 + high_conf_ratio * 0.3 + 
                        (1.0 - (std_probability * 0.5)) * 0.2 + position_score * 0.1)
        
        # Create vertebra info object
        vertebra = VertebraInfo(
            label=label,
            segmentation=extract_largest_component(mask_binary, target_label=1),
            image_props=image_props,
            bbox=None,
            meta={'source': 'nnUNet601', 'confidence': confidence},
            config=self.config
        )
        
        return vertebra, confidence

    def select_best_anchor(self, segmentation: np.ndarray, probabilities: np.ndarray, 
                            image_props: dict) -> Optional[VertebraInfo]:
        """Select vertebra with highest confidence as anchor."""
        print("Analyzing vertebrae for confidence-based selection...")
        
        # Anchor and queue must use the same label contract; otherwise the
        # pipeline could select a label that Task 602 will never process.
        unique_labels = np.unique(segmentation)
        processable_labels = set(self.config.processable_vertebra_labels())
        vertebra_labels = [int(label) for label in unique_labels 
                            if int(label) in processable_labels]
        
        if len(vertebra_labels) == 0:
            print("❌ No vertebrae found in segmentation")
            return None
        
        print(f"  All vertebrae: {vertebra_labels}")
        
        # Analyze each vertebra
        vertebrae = []
        confidence_scores = []
        
        for label in vertebra_labels:
            vertebra, confidence = self.calculate_vertebra_confidence(
                segmentation, probabilities, label, image_props
            )
            
            if vertebra is None:
                continue
                
            vertebrae.append(vertebra)
            confidence_scores.append(confidence)
        
        if not vertebrae:
            print("❌ No valid vertebra candidates found")
            return None
        
        # Select best vertebra
        best_vertebra_idx = confidence_scores.index(max(confidence_scores))
        best_vertebra = vertebrae[best_vertebra_idx]
        print(f"  Selected anchor: {best_vertebra.label} (confidence: {max(confidence_scores):.3f})")
        
        return best_vertebra
