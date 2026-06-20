from typing import Union, Optional, Tuple
import numpy as np
from ..config import Config

class VertebraCanvas:
    """Canvas for fusing multiple vertebra predictions with probability weighting."""
    
    def __init__(self, image_props: dict, config: Optional[Config] = None):
        """Initialize storage arrays for vertebra fusion."""
        self.image_size = image_props['size']  # SimpleITK format [x,y,z]
        self.image_shape = image_props['shape']  # numpy format [z,y,x]
        
        # Get all possible vertebra labels + background
        self.config = config or Config()
        self.vertebra_labels = list(self.config.VERTEBRA_LABELS.values())
        self.vertebra_labels = [0] + self.vertebra_labels  # Include background (0)
        self.n_labels = len(self.vertebra_labels)
        # The canvas stores channels densely for memory, but exports the original
        # anatomical label values so non-contiguous label contracts still work.
        self.label_to_index = {label: idx for idx, label in enumerate(self.vertebra_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.vertebra_labels)}

        # Storage arrays: probability sums and weights
        # Use dtype from config (default float16) to reduce memory usage.
        self._dtype = getattr(self.config, 'CANVAS_DTYPE', np.float16)
        self.probability_sum = np.zeros((self.n_labels,) + self.image_shape, dtype=self._dtype)
        self.probability_wts = np.zeros((self.n_labels,) + self.image_shape, dtype=self._dtype)

        print(f"✓ VertebraCanvas: Initialized for {self.image_shape} with {self.n_labels} labels")
        print(f"  ℹ Memory allocated: {2 * self.probability_sum.nbytes / 1024**2:.1f}MB (dtype={self.probability_sum.dtype})")

    def weights(self, segmentation: np.ndarray, probabilities: np.ndarray, *,
                tau_vert: float = 0.20, tau_any_bg: float = 0.25, tau_bg: float = 0.60,
                min_w: float = 0.50, max_w: float = 1.00, w_bg_scale: float = 0.50) -> np.ndarray:
        """Build per-channel weights for a crop."""
        weights = np.zeros_like(probabilities, dtype=np.float32)

        # Predicted vertebra voxels from segmentation (atomic labels 2,3,4)
        vert_mask = np.isin(segmentation, (2, 3, 4))

        # Task 602 uses channel 1 for "any vertebra"; only the atomic center and
        # neighbor channels are reliable enough to vote for specific labels.
        fg = probabilities[2:5, ...]  # (3, Z, Y, X)
        voxel_conf = fg.max(axis=0)   # (Z, Y, X)

        # per-crop scalar confidence from vertebra voxels if present; otherwise from (1 - any)
        any_low = probabilities[1, ...] < tau_any_bg
        if vert_mask.any():
            crop_w = float(np.clip(voxel_conf[vert_mask].mean(), min_w, max_w))
        else:
            # no vertebra predicted → derive a neutral-ish weight from bg-ish evidence in the crop
            # use low "any" as a proxy; fall back to min_w if there is no
            # background-supported voxel.
            mean_bg_like = (1.0 - probabilities[1, ...])[any_low].mean() if any_low.any() else min_w
            crop_w = float(np.clip(mean_bg_like, min_w, max_w))

        # Vertebra channels (2..4): once Task 602 decodes this voxel as
        # vertebra and at least one atomic channel is plausible, keep all
        # atomic probabilities. Fusion decides later whether evidence is
        # decisive enough for a final label.
        vert_gate = vert_mask & (voxel_conf >= tau_vert)   # (Z, Y, X) bool
        weights[2:5, ...] = crop_w * vert_gate.astype(np.float32)[None, ...]

        # Background votes are deliberately conservative: a local crop must say
        # both "not any vertebra" and "high background" before erasing evidence.
        bg_high = probabilities[0, ...] >= tau_bg
        bg_gate = (~vert_mask) & any_low & bg_high
        weights[0, ...] = (crop_w * w_bg_scale) * bg_gate.astype(np.float32)

        # channel 1 ("any") is not written into fusion; keep its weights 0
        return weights

    def add(self, probabilities: np.ndarray, weights: Union[np.ndarray, float] = 1.0,
            labelmap: Optional[dict] = None, bbox: Optional[dict] = None):
        """Add crop-based probabilities to the canvas."""

        # Handle weights
        weights_scalar: Optional[float] = None
        if isinstance(weights, (int, float)):
            weights_scalar = float(weights) # Avoid allocating a full-sized weight array for scalar weights.
        elif isinstance(weights, np.ndarray) and weights.shape != probabilities.shape:
            print(f"❌ Canvas: Weights shape {weights.shape} ≠ probability shape {probabilities.shape}")
            return

        # Handle labelmap
        if labelmap is None:
            if probabilities.shape[0] != self.n_labels:
                print(f"❌ Canvas: Probability map has {probabilities.shape[0]} channels, expected {self.n_labels}")
                return
            # Full-volume Task 601 channels already match the configured
            # anatomical labels. Local Task 602 crops pass an explicit labelmap.
            labelmap = {i: self.index_to_label[i] for i in range(0, self.n_labels)}
        else:
            for ch, lbl in labelmap.items():
                if lbl not in self.label_to_index:
                    print(f"❌ Canvas: Label {lbl} not in supported labels")
                    return
                if ch < 0 or ch >= probabilities.shape[0]:
                    print(f"❌ Canvas: Channel {ch} exceeds probability map channels {probabilities.shape[0]}")
                    return

        # Handle bbox boundaries
        if bbox is None:
            x0, x1 = 0, self.image_size[0]
            y0, y1 = 0, self.image_size[1]
            z0, z1 = 0, self.image_size[2]   
        else:
            x0, x1 = bbox['x0'], bbox['x1']
            y0, y1 = bbox['y0'], bbox['y1'] 
            z0, z1 = bbox['z0'], bbox['z1']

        # Check crop dimensions match
        expected_shape = (z1-z0, y1-y0, x1-x0) # 3ch: z, y, x
        if probabilities[0].shape != expected_shape:  # use 0 as representative example
            print(f"❌ Canvas: Probability shape {probabilities.shape} ≠ bbox {expected_shape}")
            return
        
        # Add probabilities to the specified region
        for ch, lbl in labelmap.items():
            lbl_idx = self.label_to_index[lbl]
            if weights_scalar is not None:
                # Scalar fast-path without creating a large temporary
                self.probability_sum[lbl_idx, z0:z1, y0:y1, x0:x1] += (probabilities[ch] * weights_scalar).astype(self._dtype, copy=False)
                self.probability_wts[lbl_idx, z0:z1, y0:y1, x0:x1] += np.asarray(weights_scalar, dtype=self._dtype)
            else:
                self.probability_sum[lbl_idx, z0:z1, y0:y1, x0:x1] += (probabilities[ch] * weights[ch]).astype(self._dtype, copy=False)
                self.probability_wts[lbl_idx, z0:z1, y0:y1, x0:x1] += weights[ch].astype(self._dtype, copy=False)

        # Success message
        region_info = f"crop {(z1-z0, y1-y0, x1-x0)}" if bbox else "full image"
        labels_info = ", ".join([f"{lbl}" for lbl in labelmap.values() if lbl > 0])
        print(f"  ✓ Added probabilities: {labels_info} to canvas within bounding box {region_info}")

    def export_fusion(self, use_dirichlet: bool = True, alpha_bg: float = 0.2,
                      alpha_vert: float = 0.05, theta_min: float = 0.40,
                      delta: float = 0.15, fallback: str = 'bg',
                      return_probabilities: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fuse accumulated probabilities with optional posterior export.

        By default, fusion uses the established Dirichlet-smoothed sums and
        then applies the conservative top-2 acceptance rule (`theta_min` and
        `delta`). Set `use_dirichlet=False` to use weighted per-channel
        averages before cross-channel normalization.
        """
        # Work on the existing buffers; convert per-slab to float32 to avoid a full-volume upcast.
        sums = self.probability_sum  # (C,Z,Y,X)
        wts  = self.probability_wts  # (C,Z,Y,X)
        C, Z, Y, X = sums.shape

        # Prepare optional posterior output
        post: Optional[np.ndarray] = None
        if return_probabilities:
            # Keep same dtype as canvas to avoid doubling memory; callers can upcast if needed.
            post = np.zeros_like(sums, dtype=sums.dtype)

        # Prepare segmentation output
        segmentation = np.zeros((Z, Y, X), dtype=np.uint8)
        label_lookup = np.asarray(self.vertebra_labels, dtype=segmentation.dtype)

        if use_dirichlet:
            alphas = np.full((C,), alpha_vert, dtype=np.float32)
            alphas[0] = alpha_bg

        # Process in slabs along Z to reduce peak memory
        slab_cfg = getattr(self.config, 'FUSION_SLAB_Z', 16)
        slab = max(1, min(int(slab_cfg), Z))

        # Counters for reporting
        total_vertebra_present = 0
        total_need_fallback = 0

        for z0 in range(0, Z, slab):
            z1 = min(Z, z0 + slab)

            # Views to avoid copies
            sums_s = sums[:, z0:z1, :, :].astype(np.float32, copy=False)
            wts_s  = wts[:,  z0:z1, :, :].astype(np.float32, copy=False)

            # vertebra evidence (any channel > 0 excluding bg) — build iteratively
            vertebra_present_s = np.zeros((z1 - z0, Y, X), dtype=bool)

            # Prepare top-2 trackers for this slab
            p1_s = np.zeros((z1 - z0, Y, X), dtype=np.float32)
            c1_s = np.zeros((z1 - z0, Y, X), dtype=np.uint8)
            p2_s = np.zeros((z1 - z0, Y, X), dtype=np.float32)

            if use_dirichlet:
                den_s = sums_s[0].copy()
                present_c = wts_s[0] > 0
                if alphas[0] != 0:
                    den_s[present_c] += alphas[0]
                for c in range(1, C):
                    present_c = wts_s[c] > 0
                    vertebra_present_s |= present_c
                    den_s += sums_s[c]
                    if alphas[c] != 0:
                        den_s[present_c] += alphas[c]

                den_nz = den_s > 0
                for c in range(C):
                    tmp = sums_s[c].copy()
                    present_c = wts_s[c] > 0
                    if alphas[c] != 0:
                        tmp[present_c] += alphas[c]
                    np.divide(tmp, den_s, out=tmp, where=den_nz)

                    better = tmp > p1_s
                    p2_s = np.where(better, p1_s, p2_s)
                    p1_s = np.where(better, tmp, p1_s)
                    c1_s = np.where(better, np.uint8(c), c1_s)
                    between = (~better) & (tmp > p2_s)
                    p2_s = np.where(between, tmp, p2_s)

                    if return_probabilities and post is not None:
                        post[c, z0:z1, :, :] = tmp.astype(post.dtype, copy=False)
            else:
                # Average each channel before cross-channel normalization so
                # repeated overlapping crop votes do not beat single global
                # background evidence merely by being counted more often.
                den_s = np.zeros_like(p1_s, dtype=np.float32)
                tmp = np.zeros_like(p1_s, dtype=np.float32)
                for c in range(C):
                    valid = wts_s[c] > 0
                    if c > 0:
                        vertebra_present_s |= valid
                    if not np.any(valid):
                        continue
                    tmp.fill(0)
                    np.divide(sums_s[c], wts_s[c], out=tmp, where=valid)
                    den_s += tmp

                # Compute normalized per-channel posteriors and track the top two
                # without materializing a full slab-sized probability copy per class.
                den_nz = den_s > 0
                for c in range(C):
                    valid = wts_s[c] > 0
                    if not np.any(valid):
                        if return_probabilities and post is not None:
                            post[c, z0:z1, :, :] = 0
                        continue
                    tmp.fill(0)
                    np.divide(sums_s[c], wts_s[c], out=tmp, where=valid)
                    np.divide(tmp, den_s, out=tmp, where=den_nz)

                    better = tmp > p1_s
                    p2_s = np.where(better, p1_s, p2_s)
                    p1_s = np.where(better, tmp, p1_s)
                    c1_s = np.where(better, np.uint8(c), c1_s)
                    between = (~better) & (tmp > p2_s)
                    p2_s = np.where(between, tmp, p2_s)

                    if return_probabilities and post is not None:
                        post[c, z0:z1, :, :] = tmp.astype(post.dtype, copy=False)

            # Decision rule & fallback on this slab
            accept_s = (vertebra_present_s & (p1_s >= theta_min) & ((p1_s - p2_s) >= delta))
            segmentation[z0:z1, :, :][accept_s] = label_lookup[c1_s[accept_s]]

            need_fallback_s = vertebra_present_s & (~accept_s)
            if fallback == 'argmax':
                segmentation[z0:z1, :, :][need_fallback_s] = label_lookup[c1_s[need_fallback_s]]
            elif fallback == 'bg':
                # already background (0)
                pass
            else:
                raise ValueError("Fallback must be 'argmax' or 'bg'")

            # Accumulate stats
            total_vertebra_present += int(vertebra_present_s.sum())
            total_need_fallback += int(need_fallback_s.sum())

        # Stats
        unique_labels = np.unique(segmentation)
        vertebra_found = [int(l) for l in unique_labels if int(l) in self.label_to_index and l > 0]
        print(f"  ✓ Canvas fusion completed")
        print(f"    voxels: {total_vertebra_present:,}, fallback applied: {total_need_fallback:,}")
        print(f"    final labels: {sorted(vertebra_found)}")

        return segmentation, post
