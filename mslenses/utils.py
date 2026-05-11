import os
import logging
import argparse

# HARDCODED PATHS & VARIABLES

# --- PREPROCESSING ---
PATH_MNI152_TEMPLATE = "MNI152/mni_icbm152_t1_tal_nlin_asym_09a.nii"
PATH_MNI152_BRAIN_MASK = "MNI152/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii"
PATH_MNI152_BRAIN_EXTRACTED = "MNI152/mni152_brain_extracted.nii.gz"

PATH_FLAIR_N4 = "flair_n4.nii.gz"
PATH_FALIR_HD_BET = "flair_hdbet.nii.gz"
PATH_FLAIR_FINAL = "flair_preprocessed.nii.gz"

# --- INFERENCE ---
PATH_NNUNET_DIR = "nnUNet_work_dir"
PATH_NNUNET_RAW = "nnUNet/nnUNet_raw"
PATH_NNUNET_PREPROCESSED = "nnUNet/nnUNet_preprocessed"
PATH_NNUNET_RESULTS = "nnUNet/nnUNet_results"

DIMENSION = (160,192,160)
MARGIN = 10
K_VALUE = 16

PATH_UNET = "models/UNet.pth"
PATH_SWINUNETR = "models/SwinUNETR.pth"
PATH_SEGRESNETDS = "models/SegResNetDS.pth"
PATH_NNUNET = "nnUNet/nnUNet_results/Dataset300_FLAIR/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"

# --- POSTPROCESSING ---
PATH_BINARY = PATH_FLAIR_FINAL.replace(".nii.gz", "_binary.nii.gz")
PATH_PROBABILITY = PATH_FLAIR_FINAL.replace(".nii.gz", "_probability.nii.gz")

PATH_ORIG_BINARY = "flair_orig_binary.nii.gz"
PATH_ORIG_PROBS  = "flair_orig_probability.nii.gz"

PATH_HYSTERESIS  = "flair_orig_hysteresis.nii.gz"

PATH_TRANSFORM_0 = "transform_0.mat"
PATH_TRANSFORM_1 = "transform_1.nii.gz"

# --- UTILS ---
DSC = [0.8424, 0.8233, 0.8550]
WORK_DIR = "directory"

logger = logging.getLogger(__name__)

def _validate_input(filepath: str) -> str:
    """
    Argument type validator for argparse: checks file extension and existence.

    Args:
        filepath (str): path to the input file provided by the user.

    Raises:
        argparse.ArgumentTypeError: if the file does not have a .nii.gz extension.
        argparse.ArgumentTypeError: if the file does not exist on disk.

    Returns:
        str: the validated input file path.
    """
    
    if not filepath.endswith(".nii.gz"): 
        raise argparse.ArgumentTypeError("The input file must be *.nii.gz!")
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    return filepath

def input_parser()-> argparse.Namespace:
    """
    Parses and validates command-line arguments.

    Returns:
        argparse.Namespace: validated arguments with attributes:
            - input (str): path to the input FLAIR image.
            - preprocessed (bool): whether the input is already preprocessed (N4 e BE).
            - full_preprocessed (bool): whether the input is full preprocessed (N4, BE and MNI152).
            - only_preprocessing (bool): run only preprocessing, skip inference and postprocessing.
            - low_threshold (float): low probability threshold for hysteresis.
            - high_threshold (float): high probability threshold for hysteresis.
            - sigma (float): FLAIR intensity similarity bandwidth.
            - connectivity (int): voxel neighbourhood connectivity (6, 18, or 26).
    """
    
    parser = argparse.ArgumentParser(
        prog="mslenses",
        description="MS-Lenses: automated multiple slerosis lesion segmentation from FLAIR MRI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required input FLAIR image
    parser.add_argument("-i", "--input", type=_validate_input, required=True, 
                        help="Input file: a FLAIR image with *.nii.gz extension.\nMust exist on disk.")
    # Preprocessing flags — mutually exclusive
    group = parser.add_mutually_exclusive_group()
    # Skip N4 correction and brain extraction if the image is already preprocessed
    group.add_argument("-p", "--preprocessed", action="store_true",
                        help="Flag: skip preprocessing if the input is already preprocessed:\n-> N4 and Brain Extraction")
    # Skip N4 correction, brain extraction and MNI152 syn Registration if the image is already preprocessed
    group.add_argument("-fp", "--full_preprocessed", action="store_true",
                        help="Flag: skip preprocessing if the input is already preprocessed:\n-> N4, Brain Extraction and SyN MNI152 registration")
    # Run only preprocessing
    group.add_argument("-op", "--only_preprocessing", action="store_true",
                        help="Flag: run only the preprocessing pipeline, skip inference and postprocessing.")
    # Adaptive hysteresis thresholding parameters
    parser.add_argument("-lt", "--low_threshold", type=float, default=0.3, 
                        help="Low threshold for adaptive hysteresis thresholding\n- Default = 0.3")
    parser.add_argument("-ht", "--high_threshold", type=float, default=0.6, 
                        help="High threshold for adaptive hysteresis thresholding\n- Default = 0.6")
    parser.add_argument("-s", "--sigma", type=float, default=0.1, 
                        help="FLAIR intensity similarity bandwidth for adaptive thresholding\n- Default = 0.1")
    parser.add_argument("-c", "--connectivity", type=int, default=6, choices=[6, 18, 26], 
                        help="Voxel neighbourhood connectivity for region growing [6, 18, 26]\n- Default = 6")

    args = parser.parse_args()

    # Validate threshold range and ordering
    if not (0 <= args.low_threshold <= 1 and 0 <= args.high_threshold <= 1):
        parser.error("Thresholds must be between 0 and 1.")
    if args.low_threshold > args.high_threshold:
        parser.error(f"Low threshold ({args.low_threshold}) must be ≤ high threshold ({args.high_threshold}).")
    if args.sigma <= 0:
        parser.error(f"Sigma ({args.sigma}) must be positive.")

    return args
        
def check_if_models_exists():
    """
    Checks that all model weights required by the ensemble are present on disk.
    Logs a warning for each missing file and prints installation instructions.
    """
    
    required_models = [
        PATH_UNET,
        PATH_SWINUNETR,
        PATH_SEGRESNETDS,
        PATH_NNUNET
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    
    if missing_models:
        for model in missing_models:
            logger.warning(
                f"{model} not found." 
            )
        logger.info(
            "Please install from Zenodo: https://zenodo.org/records/18208365/files/models.zip"    
            "See README.md (Installation) for detailed instructions."
        )

def _models_weights(k: int = 1, 
                   dsc: list = DSC) -> list[float]:
    """
    Computes normalized ensemble weights from per-model validation DSC scores.

    Each weight is proportional to DSC^k, so higher k widens the gap between
    strong and weak models. Default DSC values reflect validation performance of:
        - UNet:        0.8424
        - SwinUNETR:   0.8233
        - SegResNetDS: 0.8550

    Args:
        k (int): exponent applied to each DSC score before normalization.
                 k=1 gives linear weighting; higher values favor better models more
                 aggressively. Default: 1.
        dsc (list): Dice Similarity Coefficients for each ensemble model.
                    Defaults to the hardcoded validation results above.

    Returns:
        list: normalized weights summing to 1.0, one per model.
    """
    
    # Rise each DSC score to the power k
    dice_squared = [d**k for d in dsc]
    # Normalize so that weights sum to 1
    total_sq = sum(dice_squared)
    weights = [d/total_sq for d in dice_squared]
    
    return weights