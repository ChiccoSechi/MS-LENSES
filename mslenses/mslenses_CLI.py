from utils import input_parser, check_if_models_exists

from preprocessing import PreprocessingPipeline
import torch
from inference import nnUNet, monai_inference
from postprocessing import PostprocessingPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

print(
    r"╔═════════════════════════════════════════════════════════╗" "\n"
    r"║        __  __ ___     _    ___ _  _ ___ ___ ___         ║" "\n"
    r"║       |  \/  / __|___| |  | __| \| / __| __/ __|        ║" "\n"
    r"║       | |\/| \__ \___| |__| _|| .` \__ \ _|\__ \        ║" "\n"
    r"║       |_|  |_|___/   |____|___|_|\_|___/___|___/        ║" "\n"
    r"╠═════════════════════════════════════════════════════════╣" "\n"
    r"║ Multiple Sclerosis: Lesion Ensemble Segmentation System ║" "\n"
    r"╚═════════════════════════════════════════════════════════╝"
)

args = input_parser()

check_if_models_exists()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available: a GPU is required.")
device = torch.device("cuda")

preprocessing = PreprocessingPipeline(input_file=args.input,
                                      device=device,
                                      work_dir=args.output)

preprocessing.mni152()

if not args.full_preprocessed:
    if not args.preprocessed:
        logger.info("Preprocessing: N4 bias field correction, brain extraction, SyN MNI152 registration.")
        preprocessing.n4()
        preprocessing.brain_extraction()
    else:
        logger.info("Preprocessing: SyN MNI152 registration only (N4 and brain extraction skipped).")
    preprocessing.syn_registration()
else:
    logger.info("Preprocessing: skipped.")
    
if args.only_preprocessing:
    logger.info("Preprocessing only. Skipping inference and postprocessing.")
else:
    logger.info("Inference: nnUNet preprocessing and prediction.")
    nnunet = nnUNet(device=device,
                    work_dir=args.output)
    nnunet.preprocessing()
    nnunet.inference()

    logger.info("Inference: MONAI ensemble (UNet, SwinUNETR, SegResNetDS).")
    monai_inference(device=device,
                    work_dir=args.output)

    logger.info("Postprocessing: back-transformation to original space and adaptive hysteresis thresholding.")
    postprocessing = PostprocessingPipeline(original_flair=args.input,
                                            work_dir=args.output)

    postprocessing.to_original_space()
    postprocessing.adaptive_hysteresis_threshold(low_threshold=args.low_threshold,
                                                 high_threshold=args.high_threshold,
                                                 sigma=args.sigma,
                                                 connectivity=args.connectivity)
    postprocessing.cleanup()

