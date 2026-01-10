import os
import sys
import ants
import torch
import shutil
import argparse
import subprocess

# PREPROCESSING PIPELINE FUNCTIONS
        
def brain_extraction(
    input_file: str, 
    output_file: str, 
    device: str = "cuda"
    ):
    """Brain Extraction using HD-BET.

    Args:
        input_file (str): input file path.
        output_file (str): output file path.
        device (str): Device to use ('cuda' or 'cpu'). 
                      Default 'cuda'.
    """
    # Check if CUDA is available 
    if device == "cuda" and torch.cuda.is_available():
        # Total VRAM available
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total < 16:
            print(f"\n          WARNING! GPU has only {total:.2f} GB VRAM.")
            print("          (HD-BET needs >16 GB). Switching to CPU... ", end="")
            # Change device to CPU
            device = "cpu"  
    # HD-BET command
    cmd = ["hd-bet", 
           "-i", input_file, 
           "-o", output_file,
           "-device", device,
           "--disable_tta"]
    # Running cmd
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    # Error Handling
    except subprocess.CalledProcessError as e:
        print(f"ERROR: HD-BET failed (device: {device})")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: HD-BET not found. Install with: pip install HD-BET")
        sys.exit(1)

def MNI152():
    """Function for MNI152 brain extracted template
    """
    # Reading MNI152 template
    mni152 = ants.image_read("MNI152/mni_icbm152_t1_tal_nlin_asym_09a.nii")
    # Reading MNI152 brain mask
    mni152_brain_mask = ants.image_read("MNI152/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii")
    # MNI152 brain extracted
    mni152_brain = ants.mask_image(mni152, mni152_brain_mask)
    # Saving image
    ants.image_write(mni152_brain, "MNI152/mni152_brain_extracted.nii.gz")
    
def flair_preprocessing(
    image: str, 
    mni152_brain: ants.core.ants_image.ANTsImage, 
    work_directory: str = "work_dir", 
    device: str = "cpu", 
    preprocessed: bool = False,
    verbose: bool = False):
    """Preprocessing pipeline for FLAIR MRI standardization to MNI152 space.
    This function can perform either complete preprocessing (N4 bias correction, brain extraction, 
    and MNI152 registration) or registration-only preprocessing for images that are already 
    preprocessed.
    Complete pipeline (preprocessed=False):
        1. N4 bias field correction via ANTsPy
        2. Brain extraction using HD-BET (GPU-accelerated when available)
        3. Non-linear registration to MNI152 template space (SyN algorithm)
    Registration-only pipeline (preprocessed=True):
        1. Non-linear registration to MNI152 template space (SyN algorithm)

    Args:
        image (str): path to FLAIR image
        mni152_brain (ants.core.ants_image.ANTsImage): MNI152 brain extracted template
        work_directory (str, optional): working directory path
        device (str, optional): Device for HD-BET ('cuda' or 'cpu')
        preprocessed (bool, optional): Skip N4 and HD-BET (for already preprocessed images)
        verbose (bool, optional): ANTs verbose output
        
    Returns:
        str: filename of the preprocessed image (e.g., "flair_preprocessed.nii.gz")
    """
    # Working directory creation
    os.makedirs(work_directory, exist_ok=True)
    title = "PREPROCESSING"
    width = 60
    print()
    print("|" + "=" * (width - 2) + "|")
    print(f"|{title.center(width - 2)}|")
    print("|" + "=" * (width - 2) + "|")
    print("Starting Preprocessing:")
    # Reading image 
    flair = ants.image_read(image)
    # If the FLAIR image is raw and need preprocessing
    if not preprocessed:
        print("    -> ANTS preprocessing (via antspyx)...")
        # Application of N4 algorithm
        print("        - N4 bias field correction... ", end="")
        flair = ants.n4_bias_field_correction(flair)
        print("Done.")
        # Image writing for HD-BET execution
        ants.image_write(flair, f"{work_directory}/flair.nii.gz")
        # HD-BET execution
        print("    -> Brain Extraction (via HD-BET)...")
        if device == "cpu":
            print("        - HD-BET via CPU... ", end="")
            brain_extraction(f"{work_directory}/flair.nii.gz", f"{work_directory}/flair_hdbet.nii.gz", device)
        else:
            print("        - HD-BET via GPU... ", end="")
            brain_extraction(f"{work_directory}/flair.nii.gz", f"{work_directory}/flair_hdbet.nii.gz", device)
        # FLAIR brain extracted
        flair = ants.image_read(f"{work_directory}/flair_hdbet.nii.gz")
        print("Done!")
    # ANTS PREPROCESSING
    # FLAIR registration on MNI152 template
    print("    -> Registration (MNI152 template)... ", end="")
    reg = ants.registration(
        fixed=mni152_brain,
        moving=flair,
        type_of_transform="SyN",
        aff_metric="mattes",
        syn_metric="mattes",
        verbose=verbose
    )
    # Saving registered image
    flair = reg["warpedmovout"]
    print("Done.")
    # Saving inverted transforms (useful later) as .mat files
    inv_transforms = reg["invtransforms"]
    for i, transform in enumerate(inv_transforms):
        if transform.endswith('.nii.gz'):
            output_filename = f"transform_{i}.nii.gz"
        else:
            output_filename = f"transform_{i}.mat"
        path = os.path.join(work_directory, output_filename)
        shutil.copy(transform, path)
        print(f"    -> Saved transform: {output_filename}")
    # Define output filename
    output_filename = "flair_preprocessed.nii.gz"
    output_path = os.path.join(work_directory, output_filename)
    # Writing image
    ants.image_write(flair, output_path)
    # Temp file removal
    if not preprocessed:
        os.remove(f"{work_directory}/flair.nii.gz")
        os.remove(f"{work_directory}/flair_hdbet.nii.gz")
    print(f"You can find the preprocessed file in\n{output_path}")
    return output_filename

# ARGPARSE FUNCTIONS 
def validate_input(filepath: str):
    if not filepath.endswith(".nii.gz"): 
        raise argparse.ArgumentTypeError("The input files must be *.nii.gz!")
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    return filepath

# CHECK MODELS

def check_models():
    required_models = [
        "models/UNet.pth",
        "models/SwinUNETR.pth", 
        "models/SegResNetDS.pth",
        "nnUNet/nnUNet_results/Dataset300_FLAIR/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    
    if missing_models:
        print("\n" + "="*60)
        print("ERROR: Pre-trained models not found!")
        print("="*60)
        print("\nMissing files:")
        for model in missing_models:
            print(f"  - {model}")
        print()
        print("Please download the models from Zenodo:")
        print("https://zenodo.org/records/18208365/files/models.zip")
        print()
        print("Extract models.zip in the mslenses/ directory.")
        print("See README.md (Installation) for detailed instructions.")
        sys.exit(1)