import os
import shutil
import logging
import subprocess

import ants
import torch

from utils import (
    WORK_DIR,
    PATH_MNI152_TEMPLATE,
    PATH_MNI152_BRAIN_MASK,
    PATH_MNI152_BRAIN_EXTRACTED,
    PATH_FLAIR_N4,
    PATH_FALIR_HD_BET,
    PATH_FLAIR_FINAL
)

logger = logging.getLogger(__name__)



class PreprocessingPipeline:
    """
    Manages the full MRI preprocessing pipeline:
    - N4 bias field correction
    - Brain extraction via HD-BET
    - MNI152 registration
    """
    
    def __init__(self, 
                 input_file: str, 
                 device: torch.device,
                 work_dir: str = WORK_DIR):
        """
        Args:
            input_file (str): path to the input FLAIR image.
            device (torch.device): device to run hd-bet on (CPU or CUDA).
            work_dir (str): directory where intermediate and final files will be saved.
        """
        
        self.input_path = input_file
        self.work_dir = work_dir
        self.device = device
        self.flair = None
        
        # Create working directory
        os.makedirs(self.work_dir, exist_ok=True)
    
        # Intermediate file paths
        self.path_flair_n4    = os.path.join(work_dir, PATH_FLAIR_N4)
        self.path_flair_hdbet = os.path.join(work_dir, PATH_FALIR_HD_BET)
        self.path_flair_final = os.path.join(work_dir, PATH_FLAIR_FINAL)
        
    def mni152(self):
        """
        Extracts the brain from the MNI152 standard template.
        """
        
        if os.path.exists(PATH_MNI152_BRAIN_EXTRACTED):
            logger.info(
                "MNI152 brain extracted template already exists."
            )
            return
        
        # Load the MNI152 template
        mni152_template = ants.image_read(PATH_MNI152_TEMPLATE)
        # Load the MNI152 binary brain mask
        mni152_brain_mask = ants.image_read(PATH_MNI152_BRAIN_MASK)  
        # Apply the mask to extract the brain
        mni152_brain_only = ants.mask_image(mni152_template, mni152_brain_mask)
        # Save the resulting image
        ants.image_write(mni152_brain_only, PATH_MNI152_BRAIN_EXTRACTED)
        
    def n4(self):
        """
        Applies N4 bias field correction to the input image.
        """
        
        # Load the input image
        self.flair = ants.image_read(self.input_path)
        # Apply N4 correction
        self.flair = ants.n4_bias_field_correction(self.flair)
        # Save the corrected image
        ants.image_write(self.flair, self.path_flair_n4)
    
    def brain_extraction(self):
        """
        Performs brain extraction using HD-BET.
        Falls back to CPU if GPU VRAM is insufficient (< 16 GB).
        """
           
        if self.device.type == "cuda":
            # Compute total VRAM in GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total < 16.0:
                logger.warning(
                    f"GPU has only {total:.2f} GB VRAM. HD-BET needs >16 GB. Switching to CPU."
                )
                # Fall back to CPU
                self.device = torch.device("cpu")
        else:
            logger.info(
                f"Device: {self.device.type}."
            )
        # Run HD-BET brain extraction
        self._run_hdbet()
        self.flair = ants.image_read(self.path_flair_hdbet)
        
    def syn_registration(self):
        """
        Performs nonlinear SyN registration onto the MNI152 template.
        """
        
        # Special case: image loaded directly from disk (pipeline resumed mid-way)
        if self.flair is None:
            self.flair = ants.image_read(self.input_path)
        
        # Load the brain-extracted MNI152 template
        mni152_brain = ants.image_read(PATH_MNI152_BRAIN_EXTRACTED)
        # Run SyN registration
        registration = ants.registration(
            fixed = mni152_brain,
            moving = self.flair,
            type_of_transform="SyN",
            aff_metric="mattes",
            syn_metric="mattes"
        )
        # Store warped output (FLAIR) and save inverse transforms
        self.flair = registration["warpedmovout"]
        self._save_transforms(registration["invtransforms"])
        ants.image_write(self.flair, self.path_flair_final)
        logger.info(
            f"Preprocessed file saved in {self.path_flair_final}."
        )
        
        self._cleanup()
    
    def _cleanup(self):
        """
        Removes intermediate temporary files.
        """
        
        for path in [self.path_flair_n4, self.path_flair_hdbet]:
            if os.path.exists(path):
                os.remove(path)
        
    def _run_hdbet(self):
        """
        Runs HD-BET via subprocess.
        
        Raises:
            RuntimeError: if HD-BET exits with a non-zero return code.
            EnvironmentError: if HD-BET is not installed.
        """
        
        cmd = [
            "hd-bet",
            "-i", self.path_flair_n4,
            "-o", self.path_flair_hdbet,
            "-device", self.device.type,
            "--disable_tta"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"HD-BET failed (device: {self.device.type}, exit code: {e.returncode})"
            )
        except FileNotFoundError:
            raise EnvironmentError(
                "HD-BET not found. Install with: pip install hd-bet"
            )

    def _save_transforms(self, 
                         inv_transforms: list):
        """
        Saves inverse registration transforms to the working directory.

        Args:
            inv_transforms (list): list of inverse transform file paths.
        """
        for i, transform in enumerate(inv_transforms):
            suffix = ".nii.gz" if transform.endswith(".nii.gz") else ".mat"
            output_path = os.path.join(self.work_dir, f"transform_{i}{suffix}")
            shutil.copy(transform, output_path)
        logger.info(
            f"Saved inverse transforms in {self.work_dir}."
        )