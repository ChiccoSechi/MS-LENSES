from monai.transforms import (
    Compose, 
    LoadImage, LoadImaged, 
    EnsureChannelFirst, EnsureChannelFirstd,
    CropForeground, CropForegroundd, 
    NormalizeIntensityd,
    DivisiblePad, DivisiblePadd,
    CenterSpatialCrop, CenterSpatialCropd,
    ToTensord,
    SpatialResample
)
from monai.data import (
    MetaTensor, 
    Dataset
)
import os
import torch
import shutil
import subprocess
import numpy as np
from torch import nn
import nibabel as nib

# SETTING, INFERENCE AND ENSEMBLE FUNCTIONS

def load_models(
    unet: nn.Module, 
    swinunetr: nn.Module, 
    segresnetds: nn.Module
    ):
    """Load pre-trained ensemble models for MS segmentation.

    Args:
        unet (nn.Module): UNet neural network with residual units.
        swinunetr (nn.Module): SwinUNETR (Swin Transformer-based UNETR) neural network.
        segresnetds (nn.Module): SegResNetDS (Segmentation Residual Network with Deep Supervision) neural network.

    Returns:
        list: List containing all three models loaded with their best weights and set to evaluation mode.
    """
    # Determine available device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title = "NEURAL NETWORKS"
    width = 60
    print()
    print("|" + "=" * (width - 2) + "|")
    print(f"|{title.center(width - 2)}|")
    print("|" + "=" * (width - 2) + "|")
    # Warn USER if running on CPU
    if device.type == "cpu":
        print("WARNING! No cuda GPU detected.")
        print("Running on CPU will be extremely slow (hours instead of minutes).")
        # Request USER confirmation to proceed
        while True:
            response = input("\nDo you want to proceed anyway? (y/n): ").strip().lower()
            if response in ["y", "yes", "Y", "Yes"]:
                print("Proceeding with CPU...")
                break
            elif response in ["n", "no", "N", "No"]:
                print("Execution cancelled.")
                exit(0)
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    print(f"Loading all the models from models/ on {device}...")
    # Load UNET model
    print("    -> Loading UNET... ", end="")
    unet.load_state_dict(torch.load("models/UNet.pth", map_location=device))
    unet = unet.to(device)
    # Set to evaluation mode
    unet.eval()
    print("Done.")
    # Load SWINUNETR model
    print("    -> Loading SWINUNETR... ", end="")
    swinunetr.load_state_dict(torch.load("models/SwinUNETR.pth", map_location=device))
    swinunetr = swinunetr.to(device)
    # Set to evaluation mode
    swinunetr.eval()
    print("Done.")
    # Load SEGRESNETDS model
    print("     -> Loading SEGRESNETDS... ", end="")
    segresnetds.load_state_dict(torch.load("models/SegResNetDS.pth", map_location=device))
    segresnetds = segresnetds.to(device)
    # Set to evaluation mode
    segresnetds.eval()
    print("Done.")
    return [unet, swinunetr, segresnetds]

def models_weights(
    k: int = 1, 
    dsc: list = [0.8424, 0.8233, 0.8550]
    ):
    """Calculate normalized weights for ensemble models based on their validation performance.
     The default DSC values represent the validation performance of:
    - UNet: 0.8424
    - SwinUNETR: 0.8233
    - SegResNetDS: 0.8550
    Higher k values increase the weight difference between models, favoring better performers.

    Args:
        k (int, optional): Exponent for weight calculation. 
                           Default is 1 (linear weighting).
        dsc (list, optional): List of Dice Similarity Coefficients for each model. 
                              Default values from validation results.

    Returns:
        list: Normalized weights that sum to 1.0, one for each model in the ensemble.
    """
    # Rise each DSC score to the power k
    dice_squared = [d**k for d in dsc]
    # Calculate total sum for normalization
    total_sq = sum(dice_squared)
    # Normalize weights
    weights = [d/total_sq for d in dice_squared]
    return weights

def preprocess_nnunet(
    input_path: str,
    target_size: tuple = (160,192,160), 
    nnunet_directory: str = "nnUNet_work_dir",
    ):
    """Preprocess FLAIR image for nnUNet inference.
    This function applies spatial transformations to prepare the FLAIR image for nnUNet
    prediction. The preprocessing includes foreground cropping, padding for divisibility,
    and center cropping to a fixed size. The processed image is saved with the nnUNet
    naming convention (_0000.nii suffix).
    The function also sets up the required nnUNet environment variables if not already configured.
    
    Args:
        input_path (str): Path to the preprocessed FLAIR image.
        target_size (tuple, optional): Target spatial dimensions (H, W, D). 
                                       Default (160, 192, 160).
        nnunet_directory (str, optional): Working directory for nnUNet files. 
                                          Default "nnUNet_work_dir".
    """
    # Create output directory
    os.makedirs(nnunet_directory, exist_ok=True)
    title = "NNUNET SETTING"
    width = 60
    print()
    print("|" + "=" * (width - 2) + "|")
    print(f"|{title.center(width - 2)}|")
    print("|" + "=" * (width - 2) + "|")
    print("Preprocessing FLAIR image for nnUNet... ", end="")
    # Define preprocessing pipeline
    nnunet_transforms = Compose([
        LoadImage(image_only=True),             # Load image as MetaTensor with metadata
        EnsureChannelFirst(),                   # Ensure channel dimension is first
        CropForeground(margin=10),              # Crop to foreground with margin
        DivisiblePad(k=16),                     # Pad to make dimensions divisible by 16
        CenterSpatialCrop(roi_size=target_size),  # Crop to fixed target size
    ])
    # Apply transformations
    flair_test  = nnunet_transforms(input_path)
    # Convert to numpy and extract metadata
    flair_data = flair_test.numpy().squeeze()
    flair_affine = flair_test.meta['affine'].numpy()
    print("Done.")
    # Generate output filename (nnUNet convention: _0000.nii)
    original_filename = os.path.basename(input_path)
    output_filename = original_filename.replace(".nii.gz", "_0000.nii")
    output_path = os.path.join(nnunet_directory, output_filename)
    # Save as Nifti file
    flair_nii = nib.Nifti1Image(flair_data.astype(np.float32), affine=flair_affine)
    nib.save(flair_nii, output_path)
    print("Preprocessed image saved in nnUNet_work_dir/*_0000.nii")
    print()
    print("Setting environment variables for nnUNet...")
    # Set nnUNet environment variables if not alredy configured
    if os.environ.get("nnUNet_raw"):
        print("    -> nnUNet_raw already exists.")
    else:
        print("    -> Setting nnUNet_raw.")
        os.environ["nnUNet_raw"] = "nnUNet/nnUNet_raw"
    if os.environ.get("nnUNet_preprocessed"):
        print("    -> nnUNet_preprocessed already exists.")
    else:
        print("    -> Setting nnUNet_preprocessed.")
        os.environ["nnUNet_preprocessed"] = "nnUNet/nnUNet_preprocessed"
    if os.environ.get("nnUNet_results"):
        print("    -> nnUNet_results already exists.")
    else:
        print("    -> Setting nnUNet_results.")
        os.environ["nnUNet_results"] = "nnUNet/nnUNet_results"

def inference_nnUNet(
    nnunet_input: str = "nnUNet_work_dir",
    nnunet_output: str ="work_dir"
):
    """Run nnUNet inference for MS segmentation.
    
    This function executes the nnUNet prediction pipeline using a pre-trained 3D full-resolution
    model (Dataset 300, fold 0). The model generates both segmentation masks and probability maps
    for lesion detection. The model is available in:
    nnUNet/
    └── nnUNet_results/
        └── Dataset300_FLAIR/
            └── nnUNetTrainer__nnUNetPlans__3d_fullres/
                └── fold_0/
                    └── checkpoint_best.pth
    Args:
        nnunet_input (str, optional): Directory containing preprocessed images (_0000.nii files).
                                     Default "nnUNet_work_dir".
        nnunet_output (str, optional): Directory for saving predictions and probabilities.
                                      Default "work_dir".
    """
    # Determine device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    title = "INFERENCE"
    width = 60
    print()
    print("|" + "=" * (width - 2) + "|")
    print(f"|{title.center(width - 2)}|")
    print("|" + "=" * (width - 2) + "|")  
    print("Running nnUNet inference... ", end="")
    # Running nUNet prediction
    subprocess.run([
        "nnUNetv2_predict",
        "-i", nnunet_input,
        "-o", nnunet_output,
        "-d", "300",
        "-c", "3d_fullres",
        "-f", "0",
        "-chk", "checkpoint_best.pth",
        "--save_probabilities",
        "-device", device
    ], check=True, capture_output=True)
    print("Done.")

def inference(
    models: list,
    input_filename: str = "flair_preprocessed.nii.gz",
    work_directory: str = "work_dir", 
    target_size: tuple = (160,192,160)
):
    """Run ensemble inference combining MONAI models and nnUNet for MS segmentation.
    This function performs a complete inference pipeline that combines predictions from:
    - Three MONAI models (UNet, SwinUNETR, SegResNetDS) with weighted averaging
    - One nnUNet model (3D full-resolution)
    The final prediction is a 50/50 ensemble of MONAI and nnUNet probability maps.
    Both binary segmentation masks and probability maps are resampled back to the 
    original image space and saved.
    
    Args:
        input_filename (str, optional): Filename of preprocessed FLAIR image.
                                        Default "flair_preprocessed.nii.gz".
        work_directory (str, optional): Working directory containing the input file and for outputs. 
                                        Default "work_dir".
        target_size (tuple, optional): Target spatial dimensions for center crop (H, W, D). 
                                       Default (160, 192, 160).
        models (list): List of three loaded MONAI models in evaluation mode [UNet, SwinUNETR, SegResNetDS].
    
    Returns:
        list: filename of the preprocessed image masks 
              (e.g. "flair_preprocessed_binary.nii.gz", "flair_preprocessed_probs.nii.gz)
    """
    # Setup
    os.makedirs(work_directory, exist_ok=True)
    use_amp = torch.cuda.is_available()
    device = torch.device("cuda" if use_amp else "cpu")
    input_path = os.path.join(work_directory, input_filename)
    weights = models_weights()
    # Preprocess for nnUNet
    preprocess_nnunet(input_path, target_size)
    # Dataset dictionary
    datad = [
        {"flair":input_path}
    ]
    # MONAI preprocessing pipeline
    transforms = Compose([
        LoadImaged(keys="flair", image_only=False),
        EnsureChannelFirstd(keys="flair"),
        CropForegroundd(keys="flair", source_key="flair", margin=10),
        NormalizeIntensityd(keys="flair", nonzero=True, channel_wise=True),
        DivisiblePadd(keys="flair", k=16),
        CenterSpatialCropd(keys="flair", roi_size=target_size),
        ToTensord(keys="flair")
    ])
    # Apply transformations
    datasetd = Dataset(
        data=datad,
        transform=transforms
    )
    # Extract transformed data with metadata and add batch dimension
    datad = datasetd[0]
    flair = datad["flair"].unsqueeze(0).to(device)
    # Setup resampler for converting back to original space
    resampler = SpatialResample(mode='nearest', padding_mode='zeros')
    resampler_prob = SpatialResample(mode='trilinear', padding_mode='zeros')
    # Run nnUNet inference
    inference_nnUNet()
    print()
    print("Running MONAI inference...")
    with torch.no_grad():
        # MONAI ensemble inference with optional mixed precision
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                monai_preds = []
                for i, model in enumerate(models):
                    print(f"    -> MONAI ({i+1}/{len(models)})... ", end="")
                    pred = model(flair)
                    pred_prob = torch.softmax(pred, dim=1)
                    monai_preds.append(pred_prob)
                    print("Done.")
        else:
            monai_preds = []
            for i, model in enumerate(models):
                print(f"    -> MONAI ({i+1}/{len(models)})... ", end="")
                pred = model(flair)
                pred_prob = torch.softmax(pred, dim=1)
                monai_preds.append(pred_prob)
                print("Done.")
        # Weighted ensemble of MONAI predictions
        weights_tensor = torch.tensor(weights, device=device)
        monai_ensemble = sum(w * p for w, p in zip(weights_tensor, monai_preds))
        # Load nnUNet probability maps
        original_filename = os.path.basename(input_path)
        nnunet_prob_filename = original_filename.replace(".nii.gz", ".npz")
        nnunet_prob_path = os.path.join(work_directory, nnunet_prob_filename)
        nnunet_data = np.load(nnunet_prob_path)
        nnunet_probs = nnunet_data['probabilities']
        # Fix axis order
        nnunet_probs_fixed = np.transpose(nnunet_probs, (0, 3, 2, 1))
        nnunet_probs_tensor = torch.from_numpy(nnunet_probs_fixed).unsqueeze(0).float().to(device)
        # Combine MONAI and nnUNet predictions (50/50 weighted average)
        print()
        print("Calculating the ensemble result (50% MONAI + 50% nnUNet)...")
        ensemble_prob = 0.5 * monai_ensemble + 0.5 * nnunet_probs_tensor
        # Generate binary segmentation (argmax)
        pred_label = torch.argmax(ensemble_prob, dim=1, keepdim=True)
        pred_cpu = pred_label.cpu().float()
        # Extract lesion probability channel (channel 1)
        ensemble_prob_cpu = ensemble_prob[:, 1:2].cpu().float()
        # Create MetaTensor with transformed space metadata
        pred_metatensor = MetaTensor(
            pred_cpu[0],
            affine=datad["flair"].meta["affine"],
            meta=datad["flair"].meta
        )
        # Create MetaTensor for probability map
        prob_metatensor = MetaTensor(
            ensemble_prob_cpu[0],
            affine=datad["flair"].meta["affine"],
            meta=datad["flair"].meta
        )
        # Load original image for spatial metadata
        orig_nii = nib.load(input_path)
        # Resample binary mask to original space using nearest neighbor
        pred_resampled = resampler(
            img=pred_metatensor,
            dst_affine=torch.from_numpy(orig_nii.affine),
            spatial_size=orig_nii.shape
        )
        # Resample probability map to original space using trilinear interpolation
        prob_resampled = resampler_prob(
            img=prob_metatensor,
            dst_affine=torch.from_numpy(orig_nii.affine),
            spatial_size=orig_nii.shape
        )
        # Save binary segmentation mask
        pred_original = pred_resampled.numpy()[0].astype(np.uint8)
        # Save probability map
        prob_original = prob_resampled.numpy()[0]
    print("Saving results...")
    nifti_img = nib.Nifti1Image(pred_original, orig_nii.affine, orig_nii.header)
    base_name = original_filename.replace(".nii.gz", "")
    output_path = os.path.join(work_directory, f"{base_name}_binary.nii.gz")
    nib.save(nifti_img, output_path)
    print("    -> Binary output uploaded and available in\n       work_dir/*_binary.nii.gz!")            
    nifti_prob_img = nib.Nifti1Image(prob_original, orig_nii.affine, orig_nii.header)
    prob_output_path = os.path.join(work_directory, f"{base_name}_probs.nii.gz")
    nib.save(nifti_prob_img, prob_output_path)
    print("    -> Probability output uploaded and available in\n       work_dir/*_probs.nii.gz!")
    # Cleaning (change this if you want to mantain some of the files)
    name = input_filename.replace(".nii.gz", "")
    if os.path.exists("nnUNet_work_dir"):
        shutil.rmtree("nnUNet_work_dir")
    if os.path.exists(f"{work_directory}/{name}.nii"):
        os.remove(f"{work_directory}/{name}.nii")
    if os.path.exists(f"{work_directory}/{name}.npz"):
        os.remove(f"{work_directory}/{name}.npz")
    if os.path.exists(f"{work_directory}/{name}.pkl"):
        os.remove(f"{work_directory}/{name}.pkl")
    if os.path.exists(f"{work_directory}/dataset.json"):
        os.remove(f"{work_directory}/dataset.json")
    if os.path.exists(f"{work_directory}/plans.json"):
        os.remove(f"{work_directory}/plans.json")
    if os.path.exists(f"{work_directory}/predict_from_raw_data_args.json"):
        os.remove(f"{work_directory}/predict_from_raw_data_args.json")
    return [os.path.basename(output_path), os.path.basename(prob_output_path)]