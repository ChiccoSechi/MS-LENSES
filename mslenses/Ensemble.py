from PreprocessingPipelineFunctions import (
    flair_preprocessing,
    validate_input,
    check_models
)
from EnsembleInferenceFunctions import (
    load_models,
    inference
)
from PostprocessingFunctions import (
    from_prediction_to_orig_space,
    adaptive_hysteresis_threshold
)
from monai.networks.nets import (
    UNet, 
    SwinUNETR, 
    SegResNetDS
)
import ants
import torch
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# Parse command line arguments
parser = argparse.ArgumentParser()
# Add input argument
parser.add_argument("-i", "--input", type=validate_input, required=True, 
                    help="Input file: a FLAIR image with *.nii.gz extension.")
# Add preprocessed argument (skip N4 and brain extraction if already done)
parser.add_argument("--preprocessed", action="store_true",
                    help="The input is already preprocessed and doesn't need to be preprocessed (true/false)")
# ARGUMENTS FOR HYSTERESIS THRESHOLD
parser.add_argument("-lt", "--low_threshold", type=float, default=0.3, 
                    help="Low threshold for adaptive hysteresis threshold (default: 0.3)")
parser.add_argument("-ht", "--high_threshold", type=float, default=0.6, 
                    help="High threshold for adaptive hysteresis threshold (default: 0.6)")
parser.add_argument("-s", "--sigma", type=float, default=0.1, 
                    help="Sigma value for adaptive hysteresis threshold (default: 0.1)")
parser.add_argument("-c", "--connectivity", type=int, default=6, choices=[6, 18, 26], 
                    help="Connectivity for adaptive hysteresis threshold [6, 18, 26] (default: 6)")
# Parse command line
args = parser.parse_args()
# Get arguments
image = args.input
preprocessed = args.preprocessed
low_threshold = args.low_threshold
high_threshold = args.high_threshold
sigma = args.sigma
connectivity = args.connectivity
# Validate
if not (0 <= args.low_threshold <= 1 and 0 <= args.high_threshold <= 1):
    parser.error("Thresholds must be between 0 and 1")
if args.low_threshold > args.high_threshold:
    parser.error(f"Low threshold ({args.low_threshold}) must be ≤ high threshold ({args.high_threshold})")
if args.sigma <= 0:
    parser.error("Sigma must be positive")
check_models()
# Load MNI152 brain-extracted template for registration
mni152_brain = ants.image_read("MNI152/mni152_brain_extracted.nii.gz")
if preprocessed:
    # Registration-only pipeline (skip N4 and brain extraction)
    preprocessed_filename = flair_preprocessing(image, mni152_brain, work_directory="work_dir", preprocessed=preprocessed)
else:
    # Check if GPU is available for HD-BET
    if torch.cuda.is_available():
        print()
        print(f"GPU NVIDIA (CUDA) available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
    # Full preprocessing pipeline (N4 + brain extraction + registration)
    preprocessed_filename = flair_preprocessing(image, mni152_brain, work_directory="work_dir", device = device, preprocessed=preprocessed)
# Initialize UNET architecture with residual units
unet = UNet(
    spatial_dims=3, 
    in_channels=1, 
    out_channels=2,
    channels=(64, 128, 256, 512), 
    strides=(2, 2, 2),
    num_res_units=4, 
    dropout=0.4
)
# Initialize SWINUNETR (Swin Transformer-based) architecture
swinunetr = SwinUNETR(
    in_channels=1,
    out_channels=2,
    spatial_dims=3,
    feature_size=24,
    use_checkpoint=True,
    depths=(2,2,2,2),
    num_heads=(3, 6, 12, 24),
    window_size=5,
    mlp_ratio=4.0,
    drop_rate=0.2, 
    attn_drop_rate=0.2, 
    dropout_path_rate=0.2
)
# Initialize SEGRESNETDS architecture
segresnetds = SegResNetDS(
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
)
# Load pre-trained weights for all models
models = load_models(unet, swinunetr, segresnetds)
# Run ensemble inference (MONAI + nnUNet)
filename_binary, filename_probs = inference(models, input_filename=preprocessed_filename, work_directory="work_dir")
# Postprocessing
probs_postprocessed = from_prediction_to_orig_space(image, filename_binary, filename_probs, work_directory="work_dir")
# Adaptive hysteresis threshold
adaptive_hysteresis_threshold(probs_postprocessed, image, work_directory="work_dir",
                              low_threshold=low_threshold,
                              high_threshold=high_threshold,
                              sigma=sigma, connectivity=connectivity)

title = "RESULTS"
width = 60
print()
print("|" + "=" * (width - 2) + "|")
print(f"|{title.center(width - 2)}|")
print("|" + "=" * (width - 2) + "|")  
print("You can find the inference and preprocessing results in work_dir/:")
print("    -> *_preprocessed.nii.gz:",
      "           └── Flair image in MNI152 space.")
print()
print("    -> *_preprocessed_binary.nii.gz:", 
      "           └── Binary mask in MNI152 space.")
print()
print("    -> *_preprocessed_probs.nii.gz:", 
      "           └── Probability mask in MNI152 space.")
print()
print("    -> *_orig_binary.nii.gz:", 
      "           └── Binary mask in original space.")
print()
print("    -> *_orig_probs.nii.gz:", 
      "           └── Probability mask in original space.")
print()
print("    -> *_probs_hysteresis.nii.gz:",
      "           └── Binarized mask using hysteresis threshold.")
print()
print("The most accurate results are obtained with" 
      "\n_probs_hysteresis.nii.gz, however, we have chosen to make" 
      "\nall results available, including intermediate ones, to make" 
      "\nthe approach analyzable and reusable."
      "\n\nIf you found this project useful, please leave a star"
      "\nand, if you wish, cite it.")
