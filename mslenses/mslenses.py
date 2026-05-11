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
    "╔═════════════════════════════════════════════════════════╗\n"
    "║        __  __ ___     _    ___ _  _ ___ ___ ___         ║\n"
    "║       |  \/  / __|___| |  | __| \| / __| __/ __|        ║\n"
    "║       | |\/| \__ \___| |__| _|| .` \__ \ _|\__ \\        ║\n"
    "║       |_|  |_|___/   |____|___|_|\_|___/___|___/        ║\n"
    "╠═════════════════════════════════════════════════════════╣\n"
    "║ Multiple Sclerosis: Lesion Ensemble Segmentation System ║\n"
    "╚═════════════════════════════════════════════════════════╝\n"
)

args = input_parser()

check_if_models_exists()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available: a GPU is required.")
device = torch.device("cuda")

preprocessing = PreprocessingPipeline(input_file=args.input,
                                      device=device)

preprocessing.mni152()

if not args.full_preprocessed:
    if not args.preprocessed:
        preprocessing.n4()
        preprocessing.brain_extraction()
    preprocessing.syn_registration()
    
if args.only_preprocessing:
    logger.info("Preprocessing only. Skipping inference and postprocessing.")
else:
    print("NNUNET")
    nnunet = nnUNet(device=device)

    nnunet.preprocessing()
    nnunet.inference()
    print("MONAI")
    monai_inference(device=device)

    print("POSTPROCESSING")
    postprocessing = PostprocessingPipeline(original_flair=args.input)

    postprocessing.to_original_space()
    postprocessing.adaptive_hysteresis_threshold(low_threshold=args.low_threshold,
                                                 high_threshold=args.high_threshold,
                                                 sigma=args.sigma,
                                                 connectivity=args.connectivity)
    postprocessing.cleanup()

