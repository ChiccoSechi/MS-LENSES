import os
import shutil
import logging
import subprocess

import torch
import numpy as np
import nibabel as nib

from monai.data import (
    Dataset,
    MetaTensor 
)
from monai.transforms import (
    Compose, 
    ToTensord,
    SpatialResample,
    NormalizeIntensityd,
    LoadImage, LoadImaged, 
    DivisiblePad, DivisiblePadd,
    CropForeground, CropForegroundd, 
    CenterSpatialCrop, CenterSpatialCropd,
    EnsureChannelFirst, EnsureChannelFirstd
)
from monai.networks.nets import (
    UNet, 
    SwinUNETR, 
    SegResNetDS
)

from utils import _models_weights
from utils import (
    PATH_NNUNET_DIR,
    PATH_NNUNET_RAW,
    PATH_NNUNET_PREPROCESSED,
    PATH_NNUNET_RESULTS,
    DIMENSION,
    MARGIN,
    K_VALUE,
    PATH_UNET,          # models/UNet.pth     
    PATH_SWINUNETR,     # models/SwinUNETR.pth  
    PATH_SEGRESNETDS,   # models/SegResNetDS.pth
    WORK_DIR,
    PATH_FLAIR_FINAL
)

logger = logging.getLogger(__name__)

class nnUNet:
    """
    Handles preprocessing and inference for the nnUNet model.
    Manages environment setup, input formatting, and subprocess-based prediction.
    """
    
    def __init__(self, 
                 device: torch.device,
                 work_dir: str = WORK_DIR,
                 input_path: str = PATH_FLAIR_FINAL):
        """
        Args:
            device (torch.device): device to run inference on (CPU or CUDA).
            work_dir (str): directory where intermediate and final files will be saved.
            input_path (str): path to the preprocessed FLAIR image.
        """
        
        self.input_path = input_path
        self.work_dir = work_dir
        self.device = device
        self.target_size = DIMENSION
        self.nnunet_work_dir = PATH_NNUNET_DIR
        
        os.makedirs(self.nnunet_work_dir, exist_ok=True)
    
    
    def preprocessing(self):
        """
        Prepares the input image for nnUNet inference by applying
        the required spatial transforms and saving it in NIfTI format.
        """
        
        # Validate nnUNet environment variables before running
        self._nnunet_environment_check()
        
        #  Define the preprocessing transform pipeline for nnUNet
        nnunet_transforms = Compose([
            LoadImage(image_only=True),                     # Load image as MetaTensor with metadata
            EnsureChannelFirst(),                           # Ensure channel dimension is first
            CropForeground(margin=MARGIN),                  # Crop to foreground with margin
            DivisiblePad(k=K_VALUE),                        # Pad to make dimensions divisible by 16
            CenterSpatialCrop(roi_size=self.target_size),   # Crop to fixed target size
        ])
        
        full_input_path = os.path.join(self.work_dir, os.path.basename(self.input_path))
        # Apply transforms and extract numpy array and affine for saving
        flair_nnunet = nnunet_transforms(full_input_path)
        flair_data = flair_nnunet.numpy().squeeze()
        flair_affine = flair_nnunet.meta['affine'].numpy()
        self._save_as_nifti_file(flair_data=flair_data, flair_affine=flair_affine)
        
    def _save_as_nifti_file(self, 
                            flair_data: np.ndarray, 
                            flair_affine:np.ndarray):
        """
        Saves the preprocessed FLAIR image as a NIfTI file following
        the nnUNet input naming convention (*_0000.nii).

        Args:
            flair_data (np.ndarray): preprocessed FLAIR volume.
            flair_affine (np.ndarray): affine matrix for voxel-to-world mapping.
        """
        
        # Build output path following nnUNet convention: <name>_0000.nii
        original_filename = os.path.basename(self.input_path)
        output_filename = original_filename.replace(".nii.gz", "_0000.nii")
        output_path = os.path.join(self.nnunet_work_dir, output_filename)
        
        flair_nii = nib.Nifti1Image(flair_data.astype(np.float32), affine=flair_affine)
        nib.save(flair_nii, output_path)
        
    def _nnunet_environment_check(self):
        """
        Ensures required nnUNet environment variables are set.
        Falls back to hardcoded defaults and logs a warning for each missing variable.
        """
        
        defaults = {
            "nnUNet_raw": PATH_NNUNET_RAW,
            "nnUNet_preprocessed": PATH_NNUNET_PREPROCESSED,
            "nnUNet_results": PATH_NNUNET_RESULTS,
        }
        for var, default in defaults.items():
            if not os.environ.get(var):
                logger.warning(
                    f"Environment variable {var} not set, using default: {default}"
                )
                os.environ[var] = default
                
    def inference(self):
        """
        Runs nnUNet inference via subprocess.

        Raises:
            RuntimeError: if nnUNet exits with a non-zero return code.
        """
        
        try:
            subprocess.run([
                "nnUNetv2_predict",
                "-i", self.nnunet_work_dir,
                "-o", self.work_dir,
                "-d", "300",
                "-c", "3d_fullres",
                "-f", "0",
                "-chk", "checkpoint_best.pth",
                "--save_probabilities",
                "-device", self.device.type
            ], check=True, capture_output=True)
            
            logger.info(
                "nnUNet inference completed."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"nnUNet prediction failed (exit code: {e.returncode})\n{e.stderr.decode()}"
            )

def monai_inference(device: torch.device,
                    work_dir: str = WORK_DIR, 
                    input_filename: str = PATH_FLAIR_FINAL):
    """
    Runs inference with the three MONAI models (UNet, SwinUNETR, SegResNetDS),
    combines their predictions with nnUNet via weighted ensemble, and saves
    the resulting binary mask and probability map.

    Args:
        device (torch.device): device to run inference on (CPU or CUDA).
        work_dir (str): directory containing the preprocessed FLAIR image and
                        where output masks will be saved.
        input_filename (str): filename of the preprocessed FLAIR image.
                              Defaults to PATH_FLAIR_FINAL.
    """
    
    configs = [
            ("UNet", UNet(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=2,
                channels=(64, 128, 256, 512), 
                strides=(2, 2, 2),
                num_res_units=4, 
                dropout=0.4
            ), PATH_UNET),
            ("SwinUNETR", SwinUNETR(
                in_channels=1, 
                out_channels=2, 
                spatial_dims=3,
                feature_size=24, 
                use_checkpoint=True,
                depths=(2, 2, 2, 2), 
                num_heads=(3, 6, 12, 24),
                window_size=5, 
                mlp_ratio=4.0,
                drop_rate=0.2, 
                attn_drop_rate=0.2, 
                dropout_path_rate=0.2
            ), PATH_SWINUNETR),
            ("SegResNetDS", SegResNetDS(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=2,
                init_filters=32, 
                blocks_down=(2, 2, 4, 4),
                blocks_up=(2, 2, 2), 
                act="PRELU", 
                norm="INSTANCE",
                dsdepth=3, 
                upsample_mode='deconv'
            ), PATH_SEGRESNETDS),
        ]
    
    input_path = os.path.join(work_dir, input_filename)
    
    # Compute DSC-based weights for the MONAI ensemble
    weights = _models_weights()
    
    # MONAI preprocessing pipeline
    datadict = [
        {"flair":input_path}
    ]
    transforms = Compose([
        LoadImaged(keys="flair", image_only=False),                         # Load image as MetaTensor with metadata
        EnsureChannelFirstd(keys="flair"),                                  # Ensure channel dimension is first
        CropForegroundd(keys="flair", source_key="flair", margin=MARGIN),   # Crop to foreground with margin
        NormalizeIntensityd(keys="flair", nonzero=True, channel_wise=True), # Intensity normalization
        DivisiblePadd(keys="flair", k=K_VALUE),                             # Pad to make dimensions divisible by 16
        CenterSpatialCropd(keys="flair", roi_size=DIMENSION),               # Crop to fixed target size
        ToTensord(keys="flair")                                             # Ensure output is a torch.Tensor
    ])
    
    # Apply transforms and extract the sample with batch dimension
    datasetdict = Dataset(
        data=datadict,
        transform=transforms
    )
    datadict = datasetdict[0]
    flair = datadict["flair"].unsqueeze(0).to(device)
        
    # Run inference for each MONAI model and collect softmax predictions          
    monai_preds = []
    for name, model, path in configs:
        model.load_state_dict(torch.load(path, map_location=device))    # Load model to GPU
        model.to(device).eval() # Evaluation mode
        with torch.no_grad():   # No gradient update
            with torch.amp.autocast(device_type=device.type):   # Mixed precision if available
                pred = torch.softmax(model(flair), dim=1)
        monai_preds.append(pred)
        logger.info(
            f"{name} inference completed."
        )
        # Free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()
    
    # Weighted ensemble of the three MONAI predictions
    weights_tensor = torch.tensor(weights, device=device)
    monai_ensemble = sum(w * p for w, p in zip(weights_tensor, monai_preds))
    
    # Load and align nnUNet probability map
    original_filename = os.path.basename(input_path)
    nnunet_probs_tensor = _load_nnunet_prediction(work_dir=work_dir, 
                                                  device=device, 
                                                  original_filename=original_filename)
    
    # Average MONAI ensemble and nnUNet predictions (equal weight)
    ensemble_prob = 0.5 * monai_ensemble + 0.5 * nnunet_probs_tensor
   
    # Resample predictions back to original image space
    orig_nii = nib.load(input_path)
    binary_mask_orig, probability_mask_orig = _resampling(ensemble_prob=ensemble_prob, 
                                                          datadict=datadict, 
                                                          orig_nii=orig_nii)
    
    # Save final binary mask and probability map
    binary_path, prob_path = _save_masks(binary_mask=binary_mask_orig, 
                                         probability_mask=probability_mask_orig,
                                         orig_nii=orig_nii,
                                         original_filename=original_filename,
                                         work_dir=work_dir)
    logger.info(
        f"Results available in:\n"
        f"                           - {binary_path}\n"
        f"                           - {prob_path}"
    )
    
    _cleanup(work_dir=work_dir, input_filename=input_filename)

def _load_nnunet_prediction(work_dir: str, 
                            device: torch.device,
                            original_filename: str) -> torch.Tensor:
    """
    Loads the nnUNet probability map from a .npz file and converts it
    to a batched torch.Tensor with corrected axis order.

    Args:
        work_dir (str): directory containing the nnUNet output .npz file.
        device (torch.device): target device for the output tensor.
        original_filename (str): original FLAIR filename (used to derive the .npz path).

    Returns:
        torch.Tensor: probability map tensor with shape [1, 2, H, W, D].
    """
    
    # Derive .npz filename from the original FLAIR filename
    nnunet_prob_filename = original_filename.replace(".nii.gz", ".npz")
    nnunet_prob_path = os.path.join(work_dir, nnunet_prob_filename)
    nnunet_data = np.load(nnunet_prob_path)
    nnunet_probs = nnunet_data['probabilities']
    # Reorder axes from nnUNet convention (C, D, W, H) to (C, H, W, D)
    nnunet_probs_fixed = np.transpose(nnunet_probs, (0, 3, 2, 1))
    nnunet_probs_tensor = torch.from_numpy(nnunet_probs_fixed).unsqueeze(0).float().to(device)
    return nnunet_probs_tensor


def _resampling(ensemble_prob: torch.Tensor,
                datadict: dict,
                orig_nii: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Resamples ensemble predictions back to the original image space.

    Args:
        ensemble_prob (torch.Tensor): ensemble probability map with shape [1, 2, H, W, D].
        datadict (dict): transformed MONAI dataset dictionary with spatial metadata.
        orig_nii (nib.Nifti1Image): original NIfTI image used as the resampling target.

    Returns:
        tuple[np.ndarray, np.ndarray]: binary mask and probability map in original space.
    """
    
    # Extract lesion probability map and binary segmentation via argmax
    probability_mask_np = ensemble_prob[:, 1:2].cpu().float()
    binary_mask = torch.argmax(ensemble_prob, dim=1, keepdim=True)
    binary_mask_np = binary_mask.cpu().float()
    
    # Wrap arrays in MetaTensor to carry spatial metadata through resampling
    binary_mask_metatensor = MetaTensor(
        binary_mask_np[0],
        affine=datadict["flair"].meta["affine"],
        meta=datadict["flair"].meta
    )
    
    probability_mask_metatensor = MetaTensor(
        probability_mask_np[0],
        affine=datadict["flair"].meta["affine"],
        meta=datadict["flair"].meta
    )
    
    # Nearest-neighbour for binary mask to preserve discrete labels
    resampler_binary = SpatialResample(mode='nearest', padding_mode='zeros')
    # Trilinear interpolation for probability map to preserve continuous values
    resampler_probability = SpatialResample(mode='trilinear', padding_mode='zeros')
    
    # Resample both masks to original image space
    binary_mask_resampled = resampler_binary(
        img=binary_mask_metatensor,
        dst_affine=torch.from_numpy(orig_nii.affine),
        spatial_size=orig_nii.shape
    )
    probability_mask_resampled = resampler_probability(
        img=probability_mask_metatensor,
        dst_affine=torch.from_numpy(orig_nii.affine),
        spatial_size=orig_nii.shape
    )
    
    binary_mask_orig = binary_mask_resampled.numpy()[0].astype(np.uint8)
    probability_mask_orig = probability_mask_resampled.numpy()[0]
    return binary_mask_orig, probability_mask_orig
    
def _save_masks(binary_mask: np.ndarray,
                probability_mask: np.ndarray,
                orig_nii: nib.Nifti1Image,
                original_filename: str,
                work_dir: str) -> tuple[str, str]:
    """
    Saves binary and probability masks as NIfTI files.

    Args:
        binary_mask (np.ndarray): binary segmentation mask in original space.
        probability_mask (np.ndarray): probability map in original space.
        orig_nii (nib.Nifti1Image): original NIfTI image for affine and header.
        original_filename (str): original FLAIR filename used to derive output names.
        work_dir (str): output directory.

    Returns:
        tuple[str, str]: paths to the saved binary mask and probability map.
    """
    
    base_name = original_filename.replace(".nii.gz", "")

    binary_path = os.path.join(work_dir, f"{base_name}_binary.nii.gz")
    nib.save(nib.Nifti1Image(binary_mask, orig_nii.affine, orig_nii.header), binary_path)

    prob_path = os.path.join(work_dir, f"{base_name}_probability.nii.gz")
    nib.save(nib.Nifti1Image(probability_mask, orig_nii.affine, orig_nii.header), prob_path)

    return binary_path, prob_path
    
def _cleanup(work_dir: str, input_filename: str) -> None:
    """
    Removes temporary files generated during nnUNet inference.

    Args:
        work_dir (str): working directory containing nnUNet output files.
        input_filename (str): original FLAIR filename used to derive temp file names.
    """
    
    name = input_filename.replace(".nii.gz", "")

    # Remove nnUNet input directory
    if os.path.exists(PATH_NNUNET_DIR):
        shutil.rmtree(PATH_NNUNET_DIR)

    # Remove nnUNet output files from work_dir
    temp_nnunet_files = [
        f"{name}.nii",
        f"{name}.npz",
        f"{name}.pkl",
        "dataset.json",
        "plans.json",
        "predict_from_raw_data_args.json"
    ]
    for filename in temp_nnunet_files:
        path = os.path.join(work_dir, filename)
        if os.path.exists(path):
            os.remove(path)