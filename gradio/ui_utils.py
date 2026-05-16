import os
import sys
import numpy as np
import nibabel as nib
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# Make mslenses/ importable. 
_MSLENSES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mslenses"))
if _MSLENSES_DIR not in sys.path:
    sys.path.insert(0, _MSLENSES_DIR)

from utils import (
    PATH_FLAIR_FINAL,
    PATH_BINARY,
    PATH_PROBABILITY,
    PATH_ORIG_BINARY,
    PATH_ORIG_PROBS,
    PATH_HYSTERESIS,
)

# UI Constants
OUTPUT_LABELS = [
    "FLAIR preprocessed (MNI152)",
    "Binary Mask (MNI152)",
    "Probability Mask (MNI152)",
    "FLAIR (Original Space)",
    "Binary Mask (Original Space)",
    "Probability Mask (Original Space)",
    "Hysteresis Mask (Original Space)",
]

OUTPUT_FILENAMES = [
    PATH_FLAIR_FINAL,
    PATH_BINARY,
    PATH_PROBABILITY,
    None,
    PATH_ORIG_BINARY,
    PATH_ORIG_PROBS,
    PATH_HYSTERESIS,
]

MODE_PREPROC_ONLY = "Only Preprocessing: N4 + HD-BET + MNI152 registration (no Inference)"

N_OUTPUTS = len(OUTPUT_LABELS)  # 7

# CSS: makes the preprocessing radio group stack vertically.
RADIO_VERTICAL_CSS = ".radio-vertical .wrap { flex-direction: column !important; }"

# NIfTI Montage Rendering
def compute_center(nii_path: str):
    """
    Return integer center-of-mass voxel indices of non-zero data.
    Reorients the volume to canonical RAS orientation before computing the
    center-of-mass.
    
    Args:
        nii_path (str): path to the .nii.gz file.
        
    Returns:
        np.ndarray | None: length-3 integer array (cx, cy, cz) with the
            center-of-mass voxel indices, or None on failure.
    """
    volume = np.squeeze(nib.as_closest_canonical(nib.load(nii_path)).get_fdata())
    if volume.ndim != 3:
        return None
    mask = volume > 0
    if not mask.any():
        return None
    return np.round(center_of_mass(mask)).astype(int)


def render_nifti_montage(nii_path: str, center=None, vrange=None) -> np.ndarray | None:
    """
    Render a 1 x 3 RGB montage (axial / sagittal / coronal) from a NIfTI file.
    The image is reoriented to canonical RAS before slicing. 

    Args:
        nii_path (str): Absolute path to the .nii.gz file.
        center: Optional (cx, cy, cz) voxel indices for slice location.
                If None, the center-of-mass of non-zero voxels is used, 
                falling back to the geometric centre for all-zero volumes.
        vrange: Optional (vmin, vmax) tuple for display normalisation.
                If None, per-slice min-max is used.

    Returns:
        np.ndarray | None: H x W x 3 numpy array ready for gr.Image, 
        or None on failure.
    """
    img = nib.as_closest_canonical(nib.load(nii_path))
    volume = img.get_fdata()
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        return None
    sx, sy, sz = (float(z) for z in img.header.get_zooms()[:3])
    dim_x, dim_y, dim_z = volume.shape

    if center is not None:
        cx = int(np.clip(center[0], 0, dim_x - 1))
        cy = int(np.clip(center[1], 0, dim_y - 1))
        cz = int(np.clip(center[2], 0, dim_z - 1))
    else:
        mask = volume > 0
        if mask.any():
            com = center_of_mass(mask)
            cx, cy, cz = int(round(com[0])), int(round(com[1])), int(round(com[2]))
        else:
            cx, cy, cz = dim_x // 2, dim_y // 2, dim_z // 2

    anatomical_slices = [
        np.rot90(volume[:, :, cz]),              # axial
        np.flipud(np.rot90(volume[cx, :, :])),   # sagittal
        np.flipud(np.rot90(volume[:, cy, :])),   # coronal
    ]

    aspects = [sy / sx, sz / sy, sz / sx]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="black")
    for axis, slice_data, asp in zip(axes, anatomical_slices, aspects):
        vmin = vrange[0] if vrange is not None else slice_data.min()
        vmax = vrange[1] if vrange is not None else slice_data.max()
        # +1e-8 prevents division by zero on flat (all-zero) volumes.
        axis.imshow(
            (slice_data - vmin) / (vmax - vmin + 1e-8),
            cmap="gray", origin="lower", aspect=asp,
        )
        axis.axis("off")

    plt.tight_layout(pad=0.5)
    figure.canvas.draw()
    montage_rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return montage_rgb

def distribute_results(pipeline_results_state) -> tuple:
    """
    Unpack the pipeline results state into a flat tuple for Gradio's .then().

    Args:
        pipeline_results_state: Tuple (output_file_paths, output_montages) set by run_pipeline.

    Returns:
        tuple: flat tuple of 13 values: 6 download file paths (index 3
               excluded) followed by 7 montage arrays.
    """
    if pipeline_results_state is None:
        # 6 download paths (index 3 excluded) + 7 montages
        return (None,) * (N_OUTPUTS * 2 - 1)
    output_file_paths, output_montages = pipeline_results_state
    download_paths = [fp for i, fp in enumerate(output_file_paths) if i != 3]
    return (*download_paths, *output_montages)


# Preprocessing Mode Change Callback
def on_preprocessing_mode_change(preprocessing_methods):
    """
    React to a preprocessing mode selection change in the radio widget.
    When MODE_PREPROC_ONLY is selected, the 4 hysteresis parameter widgets
    become non-interactive and the 11 output components that only exist in
    the full pipeline are hidden.

    Args:
        preprocessing_methods: the currently selected radio value.

    Returns:
        tuple: 15 gr.update() objects in the order declared in the
            .change() call: 4 interactivity updates for hysteresis widgets
            followed by 11 visibility updates for output components.
    """
    is_preprocessing_only = preprocessing_methods == MODE_PREPROC_ONLY
    h = gr.update(interactive=not is_preprocessing_only)  # hysteresis sliders/widgets
    v = gr.update(visible=not is_preprocessing_only)      # output file/image slots
    return h, h, h, h, v, v, v, v, v, v, v, v, v, v, v
