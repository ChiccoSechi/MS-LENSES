import os
import ants
import numpy as np
import nibabel as nib
from collections import deque

def from_prediction_to_orig_space(
    image: str,
    input_path_binary: str = "flair_preprocessed_binary.nii.gz",
    input_path_probability: str = "flair_preprocessed_probs.nii.gz",
    work_directory:str = "work_dir"
):
    """Transform segmentation masks from MNI152 space back to original patient space.
    
    This function applies inverse transformations (saved during preprocessing) to bring
    both binary and probability masks from standardized MNI152 space back to the original
    anatomical space.
    
    Args:
        image (str): Path to the original FLAIR image.
        input_path_binary (str, optional): Filename of binary mask in MNI space. 
                                           Default "flair_preprocessed_binary.nii.gz".
        input_path_probability (str, optional): Filename of probability map in MNI space.
                                                Default "flair_preprocessed_probs.nii.gz".
        work_directory (str, optional): Working directory containing masks and transforms.
                                        Default "work_dir".
    
    Returns:
        str: Path to the probability map in original space.
    """
    # Load masks from MNI space
    binary_mask = ants.image_read(f"{work_directory}/{input_path_binary}")
    probability_mask = ants.image_read(f"{work_directory}/{input_path_probability}")
    # Load inverse transformation matrices (from preprocessing)
    inv_transforms = [
        f"{work_directory}/transform_0.mat",
        f"{work_directory}/transform_1.nii.gz"
    ]
    # Load original FLAIR as reference space
    orig_flair = ants.image_read(image)
    # Apply inverse transforms to binary mask (nearest neighbor for labels)
    binary_original = ants.apply_transforms(
        fixed=orig_flair,
        moving=binary_mask,
        transformlist=inv_transforms,
        interpolator="genericLabel"
    )
    # Apply inverse transforms to probability map (linear interpolation)
    probability_original = ants.apply_transforms(
        fixed=orig_flair,
        moving=probability_mask,
        transformlist=inv_transforms,
        interpolator="linear"
    )
    # Save transformed masks in original space
    ants.image_write(binary_original, f"{work_directory}/flair_orig_binary.nii.gz")
    ants.image_write(probability_original, f"{work_directory}/flair_orig_probs.nii.gz")
    if os.path.exists(f"{work_directory}/transform_0.mat"):
        os.remove(f"{work_directory}/transform_0.mat")
    if os.path.exists(f"{work_directory}/transform_1.nii.gz"):
        os.remove(f"{work_directory}/transform_1.nii.gz")
    return f"{work_directory}/flair_orig_probs.nii.gz"
    
def adaptive_hysteresis_threshold(
    input_path: str,
    image: str,
    work_directory: str = "work_dir",
    low_threshold: float = 0.3,
    high_threshold: float = 0.6,
    sigma: float = 0.1,
    connectivity: int = 6
):
    """Apply FLAIR-adaptive hysteresis thresholding for lesion segmentation refinement.
    
    This function implements an hysteresis algorithm that adapts the probability
    threshold based on local FLAIR intensity similarity. Voxels similar to seed regions 
    (in FLAIR space) require lower probability thresholds for inclusion, while dissimilar 
    voxels require higher thresholds.
    
    Algorithm:
        1. Normalize FLAIR intensities within brain mask
        2. Identify high-confidence seeds (prob > high_threshold)
        3. Grow regions using BFS with adaptive thresholding:
           threshold_adaptive = low + (high - low) * (1 - w)
           where w = exp(-(FLAIR_i - FLAIR_j) ^ 2 / (2 * sigma ^ 2))
    
    Args:
        input_path (str): Path to probability map (.nii.gz file).
        image (str): Path to original flair image.
        work_directory (str, optional): Directory for output. 
                                        Default "work_dir".
        low_threshold (float, optional): Minimum probability threshold. 
                                         Default 0.3.
        high_threshold (float, optional): High confidence seed threshold. 
                                          Default 0.6.
        sigma (float, optional): FLAIR similarity bandwidth parameter. 
                                 Default 0.1.
        connectivity (int, optional): Voxel neighborhood (6, 18, or 26). 
                                      Default 6.
    """
    # Validate connectivity parameter
    if connectivity not in [6, 18, 26]:
        raise ValueError(f"connectivity must be 6, 18, or 26, got {connectivity}")
    # Validate threshold values
    if  low_threshold > high_threshold:
        raise ValueError(
            f"Invalid threshold values: low={low_threshold}, high={high_threshold}."
            "\nLow threshold must be less than or equal to high threshold."
        )
    # Load probability map from file
    probability_mask_nii = nib.load(input_path) 
    probability_mask = probability_mask_nii.get_fdata()
    # Convert FLAIR ANTs image to numpy array
    flair = ants.image_read(image)
    flair_image = flair.numpy()
    # Create brain mask using ANTs
    brain_mask_ants = ants.get_mask(flair)
    brain_mask = brain_mask_ants.numpy().astype(bool)
    # Normalize FLAIR intensities ONLY within brain (excludes background)
    flair_brain = flair_image[brain_mask]
    flair_min = flair_brain.min()
    flair_max = flair_brain.max()
    flair_norm = np.zeros_like(flair_image, dtype=float)
    flair_norm[brain_mask] = (flair_image[brain_mask] - flair_min) / (flair_max - flair_min + 1e-10)
    # Initialize high-confidence seeds (only within brain)
    seeds_high = (probability_mask > high_threshold) & brain_mask
    # Initialize result and visited masks
    result = np.zeros_like(probability_mask, dtype=bool)
    visited = np.zeros_like(probability_mask, dtype=bool)
    # Define neighborhood offsets based on connectivity
    if connectivity == 6:
        # Face neighbors only
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    elif connectivity == 18:
        # Face + edge neighbors
        offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                   if 0 < abs(dx)+abs(dy)+abs(dz) <= 2]
    elif connectivity == 26:
        # All 26 neighbors (face + edge + corner)
        offsets = [(dx,dy,dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                   if not (dx==0 and dy==0 and dz==0)]
    # Get coordinates of all seed voxels
    seeds = np.argwhere(seeds_high)
    # Breadth-first search from each seed
    for seed in seeds:
        if visited[tuple(seed)]:
            continue
        # Initialize queue with current seed
        queue = deque([seed])
        visited[tuple(seed)] = True
        result[tuple(seed)] = True
        # Expand region from seed
        while queue:
            current = queue.popleft()
            x, y, z = current
            flair_i = flair_norm[x, y, z]
            # Check all neighbors
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                # Check image boundaries
                if not (0 <= nx < probability_mask.shape[0] and
                       0 <= ny < probability_mask.shape[1] and
                       0 <= nz < probability_mask.shape[2]):
                    continue
                # Skip if already visited
                if visited[nx, ny, nz]:
                    continue
                # Skip if outside brain mask
                if not brain_mask[nx, ny, nz]:
                    continue
                # Get neighbor values
                flair_j = flair_norm[nx, ny, nz]
                prob_j = probability_mask[nx, ny, nz]
                # Calculate FLAIR similarity weight
                w = np.exp(-(flair_i - flair_j)**2 / (2 * sigma**2))
                # Calculate adaptive threshold based on FLAIR similarity
                threshold_adaptive = low_threshold + (high_threshold - low_threshold) * (1 - w)
                # Include voxel if probability exceeds adaptive threshold
                if prob_j >= threshold_adaptive:
                    visited[nx, ny, nz] = True
                    result[nx, ny, nz] = True
                    queue.append(np.array([nx, ny, nz]))
    # Save binary result as NIfTI
    result_nii = nib.Nifti1Image(
            result,
            probability_mask_nii.affine,
            probability_mask_nii.header
        )
    nib.save(result_nii, f"{work_directory}/flair_probs_hysteresis.nii.gz")