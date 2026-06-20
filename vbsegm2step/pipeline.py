"""Main pipeline class for VBSegm2Step."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import traceback

from .config import Config
from .segmentation import SpinePredictor, LocalPredictor
from .vertebrae import VertebraCanvas, VertebraQueue
from .io import load_image, save_segmentation
from .utils import nifti_output_path, same_path


class VBSegm2StepPipeline:
    """Two-step vertebral body segmentation (global + sliding-window)."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the pipeline.
        
        Args:
            config: Configuration object, uses default if None
        """
        
        self.config = config or Config()
        
        # Validate model paths
        if not self.config.validate_model_paths():
            raise ValueError("Model paths are not valid. Please check configuration.")
        
        # Initialize predictors
        print("Loading models to memory")
        self.spine_predictor = SpinePredictor(self.config)
        self.local_predictor = LocalPredictor(self.config)

        if self.spine_predictor.predictor is None:
            raise RuntimeError("Failed to initialize SpinePredictor")
        if self.local_predictor.predictor is None:
            raise RuntimeError("Failed to initialize LocalPredictor")    
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None,
                     return_probabilities: bool = False) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], bool]:
        """Process a single CT scan file.
        
        Args:
            input_path: Path to input CT scan
            output_path: Path for output segmentation (if None, no file is saved)
            return_probabilities: If True, also returns fused probabilities
            
        Returns:
            Tuple `(segmentation, probabilities_or_None)` if successful, False otherwise
        """
        input_path = Path(input_path)
        if output_path is not None:
            output_path = Path(output_path)

        if output_path is not None and same_path(input_path, output_path):
            print("❌ Input and output paths cannot be the same")
            return False

        try:
            print(f"\n🔄 Processing: {str(input_path)}")

            # Load CT scan
            print("- Loading image")
            _, image_np, image_props = load_image(input_path)
            if image_np is None:
                return False

            # Step 1: nnU-Net 601 prediction
            print("- Running nnU-Net 601 (Whole spine segmentation)")
            spine_segmentation, spine_probabilities = self.spine_predictor.predict(
                image_np, image_props
            )

            if spine_segmentation is None:
                print("❌ nnU-Net 601 prediction failed")
                return False
            
            # Step 2: Select best anchor vertebra
            print("- Selecting best anchor vertebra")
            anchor_vertebra = self.spine_predictor.select_best_anchor(
                spine_segmentation, spine_probabilities, image_props
            )

            if anchor_vertebra is None:
                print("  no suitable anchor vertebra found")
                if output_path is not None:
                    # Keep the tool useful on weak scans: if no safe local
                    # anchor exists, return/save the global model result.
                    print("- Saving results based on nnU-Net 601 only")
                    save_segmentation(spine_segmentation, output_path, image_props)
                probabilities = spine_probabilities if return_probabilities else None
                return spine_segmentation, probabilities
            
            # Step 3: Initialize queue and canvas
            print("- Initializing processing queue and canvas")
            vertebra_queue = VertebraQueue(image_props, anchor_vertebra, config=self.config)
            canvas = VertebraCanvas(image_props, config=self.config)
            canvas.add(spine_probabilities, weights=1.0) # Add initial probabilities from nnU-Net 601

            # Step 4: Process queue with nnU-Net 602
            print("- Processing vertebra queue with nnU-Net 602 (Neighbor segmentation)")
            while vertebra_queue:
                try:
                    i_vertebra = vertebra_queue.get_next()

                    print(f"  → Starting nnU-Net 602 prediction with dynamic expansion")
                    local_segmentation, local_probabilities, bbox = self.local_predictor.predict_with_expansion(
                        image_np, image_props, i_vertebra
                    )

                    # Validate and add to queue
                    if vertebra_queue.process_segmentation(local_segmentation, bbox):
                        # Add to canvas with confidence weighting
                        print(f"  → Adding to canvas with confidence weighting")
                        weights = canvas.weights(local_segmentation, local_probabilities)
                        canvas.add(
                            probabilities=local_probabilities,
                            weights=weights,
                            labelmap=vertebra_queue.labelmap,
                            bbox=bbox
                        )
                    
                    vertebra_queue.track()

                except Exception as e:
                    print(f"❌ Error processing vertebra {vertebra_queue.i_label}: {e}, traceback: {traceback.format_exc()}")
                    vertebra_queue.track()

            print(f"  finished vertebrae queue: {sorted(vertebra_queue.tracker_done)}")
            
            # Step 5: Generate final segmentation
            print("- Generating final segmentation")
            fused_segmentation, fused_probabilities = canvas.export_fusion(
                return_probabilities=return_probabilities
            )
            
            # Step 6: Save results
            if output_path is not None:
                print("- Saving results")
                save_segmentation(fused_segmentation, output_path, image_props)

            print(f"✅ Successfully processed {input_path.name}\n\n")
            return fused_segmentation, fused_probabilities
            
        except Exception as e:
            print(f"❌ Failed to process {input_path.name}: {e}")
            return False
    
    
    def process_directory(self, input_dir: Path, output_dir: Path, pattern: str = "*.nii.gz") -> int:
        """Process all files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match
            
        Returns:
            Number of successfully processed files
        """
        from .io import load_directory
        
        # Get file list
        print(f"\n🔄 Processing directory: {str(input_dir)}")
        file_paths = load_directory(input_dir, pattern)
        
        if not file_paths:
            print("❌ No files found for processing")
            return 0
        
        # Process each file
        success_count = 0
        for file_path in file_paths:
            output_path = nifti_output_path(file_path, output_dir, input_root=input_dir)
            
            if self.process_file(file_path, output_path) is not False:
                success_count += 1
            
            print(f"Progress: {success_count}/{len(file_paths)} files completed\n")
        
        print(f"✅ Batch processing completed: {success_count}/{len(file_paths)} files successful")
        return success_count
