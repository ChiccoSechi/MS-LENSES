import os
import logging

import ants
import numpy as np
import nibabel as nib
from collections import deque

from utils import (
    WORK_DIR,
    PATH_BINARY,
    PATH_PROBABILITY,
    PATH_ORIG_BINARY,
    PATH_ORIG_PROBS,
    PATH_HYSTERESIS,
    PATH_TRANSFORM_0,
    PATH_TRANSFORM_1,
    PATH_FLAIR_FINAL
)

logger = logging.getLogger(__name__)


class PostprocessingPipeline:
    """
    Manages the full MRI postprocessing pipeline:
    - Back-transformation of masks from MNI152 space to original image space
    - Adaptive hysteresis thresholding on the probability map
    """

    def __init__(self, 
                 original_flair: str, 
                 work_dir: str = WORK_DIR):
        """
        Args:
            original_flair (str): path to the original (non-preprocessed) FLAIR image.
            work_dir (str): directory where intermediate and final files will be saved.
        """
    
        self.original_flair = original_flair
        self.work_dir = work_dir

        # Intermediate file paths
        self.path_binary      = os.path.join(work_dir, PATH_BINARY)
        self.path_probability = os.path.join(work_dir, PATH_PROBABILITY)
        self.path_orig_binary = os.path.join(work_dir, PATH_ORIG_BINARY)
        self.path_orig_probs  = os.path.join(work_dir, PATH_ORIG_PROBS)
        self.path_hysteresis  = os.path.join(work_dir, PATH_HYSTERESIS)
        self.path_transform_0 = os.path.join(work_dir, PATH_TRANSFORM_0)
        self.path_transform_1 = os.path.join(work_dir, PATH_TRANSFORM_1)

    def to_original_space(self):
        """
        Maps segmentation masks back from MNI152 space to the original 
        FLAIR space using the inverse transforms saved during SyN registration.
        """

        binary_mask = ants.image_read(self.path_binary)
        probability_mask = ants.image_read(self.path_probability)
        inv_transforms = [self.path_transform_0, self.path_transform_1]
        orig_flair = ants.image_read(self.original_flair)

        # Apply inverse transform to the binary mask using genericLabel
        # interpolation to preserve discrete label values
        binary_original = ants.apply_transforms(
            fixed=orig_flair,
            moving=binary_mask,
            transformlist=inv_transforms,
            interpolator="genericLabel"
        )

        # Apply inverse transform to the probability map using linear
        # interpolation to preserve continuous probability values
        probability_original = ants.apply_transforms(
            fixed=orig_flair,
            moving=probability_mask,
            transformlist=inv_transforms,
            interpolator="linear"
        )

        # Save both masks in original FLAIR space
        ants.image_write(binary_original, self.path_orig_binary)
        ants.image_write(probability_original, self.path_orig_probs)
        logger.info(
            f"Binary and Probability masks saved in:\n"
            f"                                - {self.path_orig_binary}\n"
            f"                                - {self.path_orig_probs}"
        )

    def adaptive_hysteresis_threshold(self,
                                      low_threshold: float = 0.3,
                                      high_threshold: float = 0.6,
                                      sigma: float = 0.1,
                                      connectivity: int = 6
    ):
        """
        Adaptive hysteresis thresholding on the probability map.

        Algorithm:
        1. Normalize FLAIR intensities within the brain mask.
        2. Identify high-confidence seeds (prob > high_threshold).
        3. Grow regions via BFS with an adaptive threshold:
               threshold_adaptive = low + (high - low) * (1 - w)
           where w = exp(-(FLAIR_i - FLAIR_j)^2 / (2 * sigma^2))

        Args:
            low_threshold (float): minimum probability threshold for region growing. Default 0.3.
            high_threshold (float): probability threshold for high-confidence seeds. Default 0.6.
            sigma (float): FLAIR intensity similarity bandwidth. Default 0.1.
            connectivity (int): voxel neighbourhood size (6, 18, or 26). Default 6.
        """
        
        # Load probability map and FLAIR image
        probability_mask_nii = nib.load(self.path_orig_probs)
        probability_mask = probability_mask_nii.get_fdata()
        flair = ants.image_read(self.original_flair)
        flair_image = flair.numpy()

        # Compute brain mask and normalize FLAIR intensities within it
        brain_mask = ants.get_mask(flair).numpy().astype(bool)
        flair_brain = flair_image[brain_mask]
        flair_norm = np.zeros_like(flair_image, dtype=float)
        flair_norm[brain_mask] = (flair_brain - flair_brain.min()) / (flair_brain.max() - flair_brain.min() + 1e-10)

        # Initialize seed voxels and BFS data structures
        seeds_high = (probability_mask > high_threshold) & brain_mask
        result = np.zeros_like(probability_mask, dtype=bool)
        visited = np.zeros_like(probability_mask, dtype=bool)

        # Define neighbourhood offsets based on connectivity
        if connectivity == 6:
            offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        elif connectivity == 18:
            offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                       if 0 < abs(dx)+abs(dy)+abs(dz) <= 2]
        elif connectivity == 26:
            offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                       if not (dx==0 and dy==0 and dz==0)]

        for seed in np.argwhere(seeds_high):
            if visited[tuple(seed)]:
                continue
            # Initialise BFS queue from the current seed voxel
            queue = deque([seed])
            visited[tuple(seed)] = True
            result[tuple(seed)]  = True
            
            while queue:
                current = queue.popleft()
                x, y, z = current
                flair_i = flair_norm[x, y, z]
                
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    # Skip out-of-bounds voxels
                    if not (0 <= nx < probability_mask.shape[0] and
                            0 <= ny < probability_mask.shape[1] and
                            0 <= nz < probability_mask.shape[2]):
                        continue
                    # Skip already-visited voxels or those outside the brain mask
                    if visited[nx, ny, nz] or not brain_mask[nx, ny, nz]:
                        continue
                    
                    flair_j = flair_norm[nx, ny, nz]
                    prob_j = probability_mask[nx, ny, nz]
                    
                    # Compute FLAIR intensity similarity weight
                    w = np.exp(-(flair_i - flair_j)**2 / (2 * sigma**2))
                    # Lower the threshold when the neighbour is intensity-similar to the seed
                    threshold_adaptive = low_threshold + (high_threshold - low_threshold) * (1 - w)
                    
                    # Include the voxel if its probability exceeds the adaptive threshold
                    if prob_j >= threshold_adaptive:
                        visited[nx, ny, nz] = True
                        result[nx, ny, nz]  = True
                        queue.append(np.array([nx, ny, nz]))

        result_nii = nib.Nifti1Image(result, probability_mask_nii.affine, probability_mask_nii.header)
        nib.save(result_nii, self.path_hysteresis)
        logger.info(
            f"Hysteresis output saved in\n"
            f"                                - {self.path_hysteresis}."
        )

    def cleanup(self):
        """
        Removes intermediate transform files from the working directory.
        """
        
        for path in [self.path_transform_0, self.path_transform_1]:
            if os.path.exists(path):
                os.remove(path)