import os
import sys
import time
import torch
import gradio as gr

# CWD is set to mslenses/ so that relative paths in utils.py works
_MSLENSES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mslenses"))
os.chdir(_MSLENSES_DIR)
if _MSLENSES_DIR not in sys.path:
    sys.path.insert(0, _MSLENSES_DIR)

# mslenses pipeline modules (resolved via sys.path above)
from preprocessing import PreprocessingPipeline
from inference import nnUNet, monai_inference
from postprocessing import PostprocessingPipeline
from utils import WORK_DIR, PATH_FLAIR_FINAL

# Gradio support utilities from the same folder
from ui_utils import (
    OUTPUT_LABELS,
    OUTPUT_FILENAMES,
    MODE_PREPROC_ONLY,
    N_OUTPUTS,
    RADIO_VERTICAL_CSS,
    render_nifti_montage,
    compute_center,
    distribute_results,
    on_preprocessing_mode_change,
)

# Pipeline Generator
def run_pipeline(
    flair_path: str,
    preprocessing_methods: str,
    low_threshold: float,
    high_threshold: float,
    sigma: float,
    connectivity: int,
):
    """
    Generator that drives the full MS-LENSES pipeline step by step.
    Yields one tuple per pipeline step to update the Gradio UI progressively.
    The final yield carries the results state used by distribute_results().

    Args:
        flair_path (str): path to the input FLAIR .nii.gz file.
        preprocessing_methods (str): selected preprocessing mode string.
        low_threshold (float): hysteresis low threshold (0-1).
        high_threshold (float): hysteresis high threshold (0-1).
        sigma (float): Gaussian sigma for the adaptive hysteresis kernel.
        connectivity (int): neighbourhood connectivity (6 / 18 / 26).

    Yields:
        tuple: (log_text, state, *widget_states) where log_text is the
            accumulated log string, state is the results payload or None,
            and widget_states are 7 gr.update() objects for the input
            widgets and start button.
    """
    
    log_lines = []
    is_preprocessing_only = preprocessing_methods == MODE_PREPROC_ONLY
    hysteresis_interactive = not is_preprocessing_only

    def _get_log_text() -> str:
        """
        Return the accumulated log as a single newline-joined string.
        """
        
        return "\n".join(log_lines)

    def _locked_widget_states() -> tuple:
        """
        
        Disable all input widgets and the start button during execution.
        """
        return tuple(gr.update(interactive=False) for _ in range(7))

    def _unlocked_widget_states() -> tuple:
        """
        Re-enable widgets after the run.
        """
        
        return (
            gr.update(interactive=True), gr.update(interactive=True),
            *(gr.update(interactive=hysteresis_interactive) for _ in range(4)),
            gr.update(interactive=True),
        )

    def _build_yield(state=None, locked=True) -> tuple:
        """
        Build the tuple yielded to Gradio on each pipeline step.

        Args:
            state:  Results payload for .then(); None on intermediate/error steps.
            locked: True keeps widgets disabled; False re-enables them.
        """
        widget_states = _locked_widget_states() if locked else _unlocked_widget_states()
        return (_get_log_text(), state, *widget_states)

    # Lock all inputs before any heavy computation.
    yield _build_yield()

    # INPUT VALIDATION 
    if flair_path is None:
        log_lines.append("Please load a .nii.gz file.")
        yield _build_yield(locked=False)
        return

    if preprocessing_methods is None:
        log_lines.append("Please select a preprocessing method.")
        yield _build_yield(locked=False)
        return

    if not is_preprocessing_only and not torch.cuda.is_available():
        log_lines.append("CUDA not available: a NVIDIA GPU is mandatory for inference.")
        yield _build_yield(locked=False)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    try:
        # PREPROCESSING
        t_preprocessing = time.time()

        preprocessing_pipeline = PreprocessingPipeline(
            input_file=flair_path, device=device, work_dir=WORK_DIR
        )
        preprocessing_pipeline.mni152()
        log_lines.append("- MNI152 template correctly loaded.")
        yield _build_yield()

        log_lines.append("PREPROCESSING:")
        yield _build_yield()

        if preprocessing_methods.startswith("N4 + HD-BET") or is_preprocessing_only:
            preprocessing_pipeline.n4()
            log_lines.append("- N4 bias field correction: DONE.")
            yield _build_yield()
            preprocessing_pipeline.brain_extraction()
            log_lines.append("- Brain extraction (via HD-BET): DONE.")
            yield _build_yield()
            preprocessing_pipeline.syn_registration()
            log_lines.append("- SyN MNI152 registration: DONE.")
            yield _build_yield()
        elif preprocessing_methods.startswith("MNI152"):
            preprocessing_pipeline.syn_registration()
            log_lines.append("- SyN MNI152 registration: DONE.")
            yield _build_yield()
        else:
            log_lines.append("- SKIPPED.")
            yield _build_yield()

        preprocessing_time = time.time() - t_preprocessing
        log_lines.append(f"> Preprocessing time: {preprocessing_time:.1f}s")
        yield _build_yield()

        # Early exit for preprocessing-only mode.
        if is_preprocessing_only:
            preprocessed_flair_path = os.path.join(WORK_DIR, PATH_FLAIR_FINAL)
            preprocessed_flair_path = preprocessed_flair_path if os.path.exists(preprocessed_flair_path) else None
            output_file_paths = [preprocessed_flair_path] + [None] * (N_OUTPUTS - 1)
            output_montages   = [render_nifti_montage(preprocessed_flair_path)] + [None] * (N_OUTPUTS - 1)
            total_time = time.time() - start_time
            log_lines.append(f"Total: {total_time:.1f}s")
            yield _build_yield(state=(output_file_paths, output_montages), locked=False)
            return

        # INFERENCE
        t_inference = time.time()

        log_lines.append("INFERENCE:")
        yield _build_yield()
        nnunet_pipeline = nnUNet(device=device, work_dir=WORK_DIR)
        nnunet_pipeline.preprocessing()
        log_lines.append("- nnUNet preprocessing: DONE.")
        yield _build_yield()
        nnunet_pipeline.inference()
        log_lines.append("- nnUNet inference: DONE.")
        yield _build_yield()

        monai_inference(device=device, work_dir=WORK_DIR)
        log_lines.append("- MONAI ensemble inference (UNet + SwinUNetr + SegResNetDS): DONE.")
        yield _build_yield()

        inference_time = time.time() - t_inference
        log_lines.append(f"> Inference time: {inference_time:.1f}s")
        yield _build_yield()

        # POSTPROCESSING
        t_postprocessing = time.time()

        log_lines.append("POSTPROCESSING:")
        yield _build_yield()
        postprocessing_pipeline = PostprocessingPipeline(original_flair=flair_path, work_dir=WORK_DIR)
        postprocessing_pipeline.to_original_space()
        log_lines.append("- Back-transformation: DONE.")
        yield _build_yield()
        postprocessing_pipeline.adaptive_hysteresis_threshold(
            low_threshold=float(low_threshold),
            high_threshold=float(high_threshold),
            sigma=float(sigma),
            connectivity=int(connectivity),
        )
        postprocessing_pipeline.cleanup()
        log_lines.append("- Hysteresis thresholding: DONE.")
        yield _build_yield()

        postprocessing_time = time.time() - t_postprocessing
        log_lines.append(f"> Postprocessing time: {postprocessing_time:.1f}s")
        yield _build_yield()

    except Exception as error:
        elapsed_seconds = time.time() - start_time
        log_lines.append(f"[ERROR]: {error}")
        log_lines.append(f"> Stopped at {elapsed_seconds:.1f}s")
        yield _build_yield(locked=False)
        return

    # index 3 (FLAIR Original Space) is filled with the input flair_path because OUTPUT_FILENAMES has None there.
    output_file_paths = [
    flair_path if fn is None else os.path.join(WORK_DIR, fn)
    for fn in OUTPUT_FILENAMES
]
    # Shared centres: each space uses the CoM of its own binary mask so that
    # FLAIR, binary, probability, and hysteresis all show the same slice.
    # Falls back to the FLAIR CoM (MNI or original) when no lesions are found.
    mni_center = compute_center(output_file_paths[1])
    if mni_center is None:
        mni_center = compute_center(output_file_paths[0])

    orig_center = compute_center(output_file_paths[4])
    if orig_center is None:
        orig_center = compute_center(flair_path)

    # min-max normalisation
    _FLAIR_INDICES = {0, 3} 
    output_montages = [
        render_nifti_montage(
            fp,
            center=mni_center if i < 3 else orig_center,
            vrange=None if i in _FLAIR_INDICES else (0, 1),
        )
        for i, fp in enumerate(output_file_paths)
    ]

    total_time = time.time() - start_time
    log_lines.append(f"Completed in {total_time:.1f}s")
    yield _build_yield(state=(output_file_paths, output_montages), locked=False)

# UI
with gr.Blocks(title="MS-LENSES") as demo:
    gr.Markdown(
        "# MS-LENSES\n"
        "**MS Lesion Ensemble Segmentation System** - "
        "Automatic Lesion Segmentation from FLAIR MRI."
    )

    with gr.Row():
        with gr.Column(scale=1):
            flair_file_input = gr.File(
                label="Load FLAIR image (.nii.gz):",
                file_count="single",
                file_types=[".gz"],
                type="filepath",
                interactive=True,
            )
            with gr.Group():
                gr.Markdown("**Preprocessing**")
                preprocessing_methods = gr.Radio(
                    choices=[
                        "N4 + HD-BET + MNI152 registration",
                        "MNI152 registration only",
                        "Skipped (no Preprocessing)",
                        MODE_PREPROC_ONLY,
                    ],
                    value=None,
                    label="Mode:",
                    elem_classes=["radio-vertical"],
                )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**Postprocessing: Hysteresis thresholding parameters**")
                low_threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Low threshold")
                high_threshold = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="High threshold")
                sigma = gr.Number(value=0.1, minimum=0.001, label="Sigma")
                connectivity_radio = gr.Radio(choices=[6, 18, 26], value=6, label="Connectivity")

    start_button = gr.Button("Start Pipeline", variant="primary", size="lg")
    log_textbox  = gr.Textbox(label="Log", lines=5, interactive=False)

    # DOWNLOAD SECTION
    gr.Markdown("### Download output files")
    download_components = []
    with gr.Row():
        download_components.append(gr.File(label=OUTPUT_LABELS[0], interactive=False))
        for label in OUTPUT_LABELS[1:3]:
            download_components.append(gr.File(label=label, interactive=False))
    with gr.Row():
        for label in OUTPUT_LABELS[4:]:
            download_components.append(gr.File(label=label, interactive=False))

    # VISUALIZATION SECTION
    gr.Markdown("### 3-AXIS visualization")
    image_components = []
    with gr.Row():
        image_components.append(gr.Image(label=OUTPUT_LABELS[0], show_label=True,
                                         buttons=["download", "fullscreen"]))
        for label in OUTPUT_LABELS[1:3]:
            image_components.append(gr.Image(label=label, show_label=True,
                                             buttons=["download", "fullscreen"]))
    with gr.Row():
        for label in OUTPUT_LABELS[3:]:
            image_components.append(gr.Image(label=label, show_label=True,
                                             buttons=["download", "fullscreen"]))

    pipeline_results_state = gr.State(None)

    pipeline_input_widgets = [
        flair_file_input, preprocessing_methods,
        low_threshold, high_threshold, sigma, connectivity_radio,
    ]
    
    pipeline_output_widgets = [log_textbox, pipeline_results_state, *pipeline_input_widgets, start_button]

    preprocessing_methods.change(
        fn=on_preprocessing_mode_change,
        inputs=preprocessing_methods,
        outputs=[
            low_threshold, high_threshold, sigma, connectivity_radio,
            download_components[1], download_components[2], download_components[3],
            download_components[4], download_components[5],
            image_components[1], image_components[2], image_components[3],
            image_components[4], image_components[5], image_components[6],
        ],
    )

    start_button.click(
        fn=run_pipeline,
        inputs=pipeline_input_widgets,
        outputs=pipeline_output_widgets,
    ).then(
        fn=distribute_results,
        inputs=[pipeline_results_state],
        outputs=[*download_components, *image_components],
    )

demo.launch(css=RADIO_VERTICAL_CSS)
