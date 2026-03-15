"""
SD3.5 Scheduled Animation Pipeline
====================================
Pure Python equivalent of the ComfyUI SD3.5 + ControlNet + IPAdapter workflow.

THREE PIPELINE MODES  set pipeline_mode in PipelineConfig:

  "controlnet_sd35"   SD3.5 Large + SD3.5-specific ControlNet  � ideal, needs
                      InstantX/SD3.5-Large-Controlnet-Canny      correct weights

  "controlnet_sd3"    SD3 Medium + SD3 Medium ControlNet        � works today,
                      stabilityai/stable-diffusion-3-medium      lower quality
                      InstantX/SD3-Controlnet-Canny

  "img2img"           SD3.5 Large img2img, no ControlNet        � fallback,
                      Canny frame used as init image              most compatible

The tensor mismatch error (size 2432 vs 1536) means you tried to pair the
SD3 Medium ControlNet with SD3.5 Large. Use mode "controlnet_sd35" with the
matching weights, or fall back to "img2img".

Requirements:
    pip install diffusers transformers accelerate torch torchvision \
                opencv-python pillow numpy tqdm
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3Pipeline,
    SD3ControlNetModel,
)
from diffusers.utils import load_image
from transformers import SiglipVisionModel, SiglipImageProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    #  Pipeline mode 
    # "controlnet_sd35" | "controlnet_sd3" | "img2img"
    pipeline_mode: str       = "img2img"

    #  Model IDs 
    # controlnet_sd35: sd3_model="stabilityai/stable-diffusion-3.5-large"
    #                  controlnet_model="InstantX/SD3.5-Large-Controlnet-Canny"
    # controlnet_sd3:  sd3_model="stabilityai/stable-diffusion-3-medium-diffusers"
    #                  controlnet_model="InstantX/SD3-Controlnet-Canny"
    # img2img:         sd3_model="stabilityai/stable-diffusion-3.5-large"
    #                  controlnet_model="" (ignored)
    sd3_model_id: str           = "stabilityai/stable-diffusion-3.5-large"
    controlnet_model_id: str    = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
    ipadapter_model_id: str     = "InstantX/SD3.5-Large-IP-Adapter"   # set to "InstantX/SD3-IPAdapter" to enable
    image_encoder_id: str       = "google/siglip-so400m-patch14-384"

    #  I/O paths 
    input_video_path: str       = "input_video.mp4"
    input_video_path_global: str = "input_videos"  # for batch processing multiple videos
    output_frames_dir: str      = "output/frames"
    output_video_path: str      = "output/animation_output.mp4"
    style_image_path: str       = "style_image.png"                           # leave "" to disable

    #  Scheduled prompts (4 keyframes) 
    prompt_start: str           = "a misty forest at dawn, golden light through trees, cinematic, photorealistic, 8k"
    prompt_start_mid: str       = "a sunlit forest at midday, vibrant greens, dappled light, cinematic, 8k"
    prompt_end_mid: str         = "a forest at golden hour, warm amber tones, long shadows, cinematic, 8k"
    prompt_end: str             = "a moonlit forest at night, silver light, deep shadows, cinematic, 8k"

    negative_prompt: str        = "blurry, low quality, distorted, watermark, text, ugly, deformed, noise"

    #  Sampling 
    steps: int                  = 20
    cfg: float                  = 4.5
    controlnet_strength: float  = 1.0
    img2img_strength: float     = 0.80  # img2img mode only: 0=keep init, 1=ignore it
    ipadapter_weight: float     = 0.5
    seed: int                   = 42
    width: int                  = 1024
    height: int                 = 576

    #  Video 
    output_fps: int             = 24
    canny_low_threshold: int    = 100
    canny_high_threshold: int   = 200
    dissolve_frames: float      = 6 # how many frames are used to dissolve, for 24fps 6 is 0.25s, 4.8 is 0.2s, 12 is 0.5s, etc.
    hold_frames: int            = 4 # how many frames to hold the keyframe prompts, for 24fps 4 is ~0.17s, 6 is 0.25s, 12 is 0.5s, etc.

    #  Hardware 
    device: str                 = "cuda"   # "cpu" or "mps" for Apple Silicon
    dtype: torch.dtype          = torch.bfloat16


# ---------------------------------------------------------------------------
# 1. Video ControlNet frames  (Canny edge extraction)
# ---------------------------------------------------------------------------

def extract_canny_frames(
    video_path: str,
    output_dir: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
    target_width: int = 1024,
    target_height: int = 576,
) -> list[Path]:
    """
    Extract every frame from a video, apply Canny edge detection,
    and save as PNG. Returns sorted list of output frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_paths: list[Path] = []

    print(f"Extracting {total} frames from {video_path}...")
    for idx in tqdm(range(total), unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to target resolution
        frame = cv2.resize(frame, (target_width, target_height),
                           interpolation=cv2.INTER_AREA)

        # Convert to greyscale and apply Canny
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, low_threshold, high_threshold)

        # Convert single-channel edges � RGB PIL image (ControlNet expects RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(edges_rgb)

        out_path = output_dir / f"canny_{idx:05d}.png"
        pil_image.save(out_path)
        saved_paths.append(out_path)

    cap.release()
    print(f"Saved {len(saved_paths)} Canny frames to {output_dir}")
    return sorted(saved_paths)


# ---------------------------------------------------------------------------
# 2. Prompt scheduling  (linear interpolation across 4 keyframes)
# ---------------------------------------------------------------------------

def interpolate_prompts(
    prompt_start: str,
    prompt_start_mid: str,
    prompt_end_mid: str,
    prompt_end: str,
    total_frames: int,
) -> list[str]:
    """
    Returns one prompt string per frame, linearly interpolated between the
    4 keyframe prompts.

    Because language models cannot interpolate text the way you blend pixels,
    the strategy here is:
      - Frames 0 - 33%:  start - start_mid   (weighted blend description)
      - Frames 33% - 66%:  start_mid - end_mid
      - Frames 66% - 100%: end_mid - end

    In practice, for SD3.5 we send the two bracketing prompts to the dual
    positive conditioning and weight them. This function returns tuples but
    is simplified here to a single weighted string for readability.
    For true dual-prompt weighting, see `build_conditioned_prompt()` below.
    """
    keyframes = [
        (0.00,  prompt_start),
        (0.33,  prompt_start_mid),
        (0.66,  prompt_end_mid),
        (1.00,  prompt_end),
    ]

    scheduled: list[str] = []
    for i in range(total_frames):
        progress = i / max(total_frames - 1, 1)

        # Find the two surrounding keyframes
        for k in range(len(keyframes) - 1):
            t0, p0 = keyframes[k]
            t1, p1 = keyframes[k + 1]
            if t0 <= progress <= t1:
                # Normalised position between these two keyframes
                alpha = (progress - t0) / (t1 - t0) if t1 > t0 else 0.0
                # Below 50% of segment � use earlier prompt; above � later
                # For a smooth transition, concatenate with weights embedded
                # as a prompt blend (SD3.5 responds well to this)
                if alpha < 0.5:
                    scheduled.append(p0)
                else:
                    scheduled.append(p1)
                break
        else:
            scheduled.append(prompt_end)

    return scheduled


def get_prompt_weight(
    prompt_a: str,
    prompt_b: str,
    alpha: float,
) -> tuple[str, float, str, float]:
    """
    Returns (prompt_a, weight_a, prompt_b, weight_b) for use with
    SD3.5's dual-conditioning support via diffusers.
    """
    return prompt_a, 1.0 - alpha, prompt_b, alpha


# ---------------------------------------------------------------------------
# 3. Load models
# ---------------------------------------------------------------------------

from diffusers.image_processor import VaeImageProcessor


class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)
    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image
    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        return super().postprocess(image, **kwargs, do_denormalize=do_denormalize)

PipelineType = Union[
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Img2ImgPipeline,
]

def load_pipeline(config: PipelineConfig) -> PipelineType:
    """
    Load the appropriate pipeline based on config.pipeline_mode:

      controlnet_sd35   SD3.5 Large + matching SD3.5 ControlNet
      controlnet_sd3    SD3 Medium + SD3 Medium ControlNet
      img2img           SD3.5 Large img2img (no ControlNet, Canny as init image)

    The error "size 2432 vs 1536" means the ControlNet hidden dim does not
    match the base model. SD3 Medium uses 1536, SD3.5 Large uses 2432.
    They are NOT interchangeable. Use img2img mode if you only have the
    SD3 Medium ControlNet weights but want SD3.5 Large.
    """

    mode = config.pipeline_mode

    if mode in ("controlnet_sd35", "controlnet_sd3"):
        if not config.controlnet_model_id:
            raise ValueError(
                f"pipeline_mode='{mode}' requires controlnet_model_id. "
                "Use 'InstantX/SD3.5-Large-Controlnet-Canny' for sd35 "
                "or 'InstantX/SD3-Controlnet-Canny' for sd3."
            )
        
        use_style = bool(config.style_image_path and config.ipadapter_model_id)
        if use_style:
            print(f"Loading SigLip image encoder: {config.ipadapter_model_id}")
            feature_extractor = SiglipImageProcessor.from_pretrained(
                config.image_encoder_id,
            )
            image_encoder = SiglipVisionModel.from_pretrained(
                config.image_encoder_id,
                torch_dtype=config.dtype,
            )
            #  Keep this on the cpu
            image_encoder = image_encoder.to("cpu")
        else:
            feature_extractor = None
            image_encoder = None

        print(f"Loading ControlNet: {config.controlnet_model_id}")
        controlnet = SD3ControlNetModel.from_pretrained(
            config.controlnet_model_id,
            torch_dtype=config.dtype,
        )
        print(f"Loading base model: {config.sd3_model_id}")
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            config.sd3_model_id,
            controlnet=controlnet,
            torch_dtype=config.dtype,
            feature_extractor=feature_extractor,
            image_encoder= image_encoder,
            use_safetensors=True,
        )
        # print(f"Loading IPAdapter model: {config.ipadapter_model_id}")
        pipe.image_processor = SD3CannyImageProcessor()
    elif mode == "img2img":
        print(f"Loading SD3.5 img2img pipeline: {config.sd3_model_id}")
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            config.sd3_model_id,
            torch_dtype=config.dtype,
            use_safetensors=True,
        )
        use_style = False
    else:
        raise ValueError(
            f"Unknown pipeline_mode '{mode}'. "
            "Choose: 'controlnet_sd35', 'controlnet_sd3', 'img2img'"
        )

    # Memory optimisation  remove if you have e24 GB VRAM
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()  # slower, lowest VRAM (~4 GB)
    pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()

    pipe.text_encoder.to("cpu")    # CLIP-L   ~0.5GB
    pipe.text_encoder_2.to("cpu")  # CLIP-G   ~1.5GB
    pipe.text_encoder_3.to("cpu")  # T5-XXL   ~9.5GB
    torch.cuda.empty_cache()

    # Optional IPAdapter (ControlNet modes only not supported in img2img here)
    # if config.ipadapter_model_id and config.style_image_path and mode != "img2img":
    #     print("Loading IPAdapter...")
    #     pipe.load_ip_adapter(
    #         config.ipadapter_model_id,
    #         subfolder="",
    #         weight_name="ip-adapter.bin",
    #     )
    #     pipe.set_ip_adapter_scale(config.ipadapter_weight)

    if use_style:
        print(f"Loading IPAdapter weights: {config.ipadapter_model_id}")
        pipe.load_ip_adapter(
            config.ipadapter_model_id,
            weight_name="ip-adapter.bin",
        )
        pipe.set_ip_adapter_scale(config.ipadapter_weight)

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"Pipeline device: {pipe.device}")
    print(f"Transformer device: {pipe.transformer.device}")
    print(f"VAE device: {pipe.vae.device}")
    print(f"VRAM allocated: {allocated:.1f} GB")
    print(f"VRAM reserved:  {reserved:.1f} GB")

    return pipe


# ---------------------------------------------------------------------------
# 4. Generate a single frame
# ---------------------------------------------------------------------------

def generate_frame(
    pipe: PipelineType,
    config: PipelineConfig,
    controlnet_image: Image.Image,
    prompt: str,
    frame_index: int,
    style_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Run one inference pass for a single frame.
    Handles all three pipeline modes transparently.
    """
    generator   = torch.Generator(device=config.device).manual_seed(config.seed)
    mode        = config.pipeline_mode

    # Move text encoders to GPU just for encoding, then back to CPU
    pipe.text_encoder.to(config.device)
    pipe.text_encoder_2.to(config.device)
    pipe.text_encoder_3.to(config.device)

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=config.negative_prompt,
            device=config.device,
        )

    # Offload text encoders back to CPU before denoising
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()

    ip_adapter_image_embeds = None
    if style_image is not None and config.style_image_path and config.ipadapter_model_id:
        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            pipe.image_encoder.to(config.device)

        with torch.no_grad():
            ip_adapter_image_embeds = pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=style_image,
                ip_adapter_image_embeds=None,
                device=config.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            pipe.image_encoder.to("cpu")
        torch.cuda.empty_cache()  

    # Denoising 
    if mode in ("controlnet_sd35", "controlnet_sd3"):
        call_kwargs = dict(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            control_image=controlnet_image,
            controlnet_conditioning_scale=config.controlnet_strength,
            num_inference_steps=config.steps,
            guidance_scale=config.cfg,
            width=config.width,
            height=config.height,
            generator=generator,
        )
        if ip_adapter_image_embeds is not None:
            call_kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds

    elif mode == "img2img":
        call_kwargs = dict(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=controlnet_image,
            strength=config.img2img_strength,
            num_inference_steps=config.steps,
            guidance_scale=config.cfg,
            width=config.width,
            height=config.height,
            generator=generator,
        )

    result = pipe(**call_kwargs)
    return result.images[0]


# ---------------------------------------------------------------------------
# 5. Frames � video
# ---------------------------------------------------------------------------

def compile_video(
    config: PipelineConfig,
    frames_dir: str,
    output_path: str,
    fps: int = 24,
    dissolve_frames: int = 6,
    hold_frames: int = 4,
) -> None:
    """
    Compile all PNG frames in frames_dir (sorted by name) into an MP4.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc

    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("generated_*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"No generated_*.png frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    fourcc = VideoWriter_fourcc(*"mp4v")
    writer = VideoWriter(output_path, fourcc, fps, (w, h))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dissolve_frames = int(config.dissolve_frames)
    hold_frames = int(config.hold_frames)

    total_output_frames = 1 + (len(frame_paths) - 1) * (1 + dissolve_frames + hold_frames)

    print(
        f"Compiling {len(frame_paths)} source frames → {total_output_frames} output "
        f"frames ({dissolve_frames} dissolve frames per transition) into {output_path}"
    )

    with tqdm(total=total_output_frames, unit="frame") as pbar:
        prev_frame = cv2.imread(str(frame_paths[0]))

        for path in frame_paths[1:]:
            curr_frame = cv2.imread(str(path))

            for _ in range(hold_frames):
                writer.write(prev_frame)
                pbar.update(1)
            
            for i in range(1, dissolve_frames + 1):
                alpha = i / dissolve_frames
                alpha = alpha * alpha * (3 - 2 * alpha)
                blended = cv2.addWeighted(prev_frame, 1.0 - alpha, curr_frame, alpha, 0)
                writer.write(blended)
                pbar.update(1)

            prev_frame = curr_frame
        
        for _ in range(hold_frames):
            writer.write(prev_frame)
            pbar.update(1)

    writer.release()
    print(f"Video saved: {output_path}")

def compile_video_dissolve(
    frames_dir: str,
    output_path: str,
    fps: int = 24,
    dissolve_frames: float = 6.0,
    hold_frames: int = 4,
) -> None:
    """
    Compile all PNG frames in frames_dir (sorted by name) into an MP4.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc

    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("generated_*.png"))

    if not frame_paths:
        raise FileNotFoundError(f"No generated_*.png frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    output_path = Path(output_path)
    if output_path.suffix != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = VideoWriter_fourcc(*"mp4v")
    writer = VideoWriter(output_path, fourcc, fps, (w, h))

    dissolve_frames = int(dissolve_frames)
    hold_frames = int(hold_frames)

    total_output_frames = 1 + (len(frame_paths) - 1) * (1 + dissolve_frames + hold_frames)

    print(
        f"Compiling {len(frame_paths)} source frames → {total_output_frames} output "
        f"frames ({dissolve_frames} dissolve frames per transition) into {output_path}"
    )

    with tqdm(total=total_output_frames, unit="frame") as pbar:
        prev_frame = cv2.imread(str(frame_paths[0]))

        for path in frame_paths[1:]:
            curr_frame = cv2.imread(str(path))

            for _ in range(hold_frames):
                writer.write(prev_frame)
                pbar.update(1)
            
            for i in range(1, dissolve_frames + 1):
                alpha = i / dissolve_frames
                alpha = alpha * alpha * (3 - 2 * alpha)
                blended = cv2.addWeighted(prev_frame, 1.0 - alpha, curr_frame, alpha, 0)
                writer.write(blended)
                pbar.update(1)

            prev_frame = curr_frame
        
        for _ in range(hold_frames):
            writer.write(prev_frame)
            pbar.update(1)

    writer.release()
    print(f"Video saved: {output_path}")


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> None:

    #  Step 1: Extract Canny frames from input video 
    if not (Path(config.output_frames_dir) / "canny").exists():
        canny_dir = Path(config.output_frames_dir) / "canny"
        canny_frames = extract_canny_frames(
            video_path=config.input_video_path,
            output_dir=str(canny_dir),
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
            target_width=config.width,
            target_height=config.height,
        )
    else:
        canny_frames = sorted((Path(config.output_frames_dir) / "canny").glob("canny_*.png"))
        print(f"Found {len(canny_frames)} existing Canny frames in {config.output_frames_dir}/canny/")
    total_frames = len(canny_frames)

    #  Step 2: Build scheduled prompts 
    prompts = interpolate_prompts(
        config.prompt_start,
        config.prompt_start_mid,
        config.prompt_end_mid,
        config.prompt_end,
        total_frames,
    )

    #  Step 3: Load models 
    pipe = load_pipeline(config)

    #  Step 4: Load style image if provided 
    style_image: Optional[Image.Image] = None
    if config.style_image_path and Path(config.style_image_path).exists():
        style_image = load_image(config.style_image_path).resize(
            (config.width, config.height)
        )
        print(f"Style image loaded: {config.style_image_path}")

    #  Step 5: Generate frames 
    gen_dir = Path(config.output_frames_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {total_frames} frames...")
    for i, (canny_path, prompt) in enumerate(
        tqdm(zip(canny_frames, prompts), total=total_frames, unit="frame")
    ):
        controlnet_image = Image.open(canny_path).convert("RGB")

        out_path = gen_dir / f"generated_{i:05d}.png"

        if out_path.exists():
            continue  # resume from where you left off

        output_image = generate_frame(
            pipe=pipe,
            config=config,
            controlnet_image=controlnet_image,
            prompt=prompt,
            frame_index=i,
            style_image=style_image,
        )

        output_image.save(out_path)

    #  Step 6: Compile output video 
    compile_video_dissolve(
        frames_dir=config.output_frames_dir,
        output_path=config.output_video_path,
        fps=config.output_fps,
        dissolve_frames=config.dissolve_frames,
        hold_frames=config.hold_frames,
    )

    print("\n Pipeline complete.")
    print(f"Frames: {config.output_frames_dir}/generated_*.png")
    print(f"Video:  {config.output_video_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # =========================================================================
    # CHOOSE YOUR MODE  uncomment ONE block below
    # =========================================================================

    #  MODE A: SD3.5 Large + SD3.5-specific ControlNet (best quality) 
    # Requires: InstantX/SD3.5-Large-Controlnet-Canny weights
    # config = PipelineConfig(
    #     pipeline_mode        = "controlnet_sd35",
    #     sd3_model_id         = "stabilityai/stable-diffusion-3.5-large",
    #     controlnet_model_id  = "InstantX/SD3.5-Large-Controlnet-Canny",
    #     ...
    # )

    #  MODE B: SD3 Medium + SD3 Medium ControlNet 
    # Use this if you only have the original InstantX/SD3-Controlnet-Canny.
    # Will NOT work with SD3.5 Large (causes the 2432 vs 1536 tensor error).
    # config = PipelineConfig(
    #     pipeline_mode        = "controlnet_sd3",
    #     sd3_model_id         = "stabilityai/stable-diffusion-3-medium-diffusers",
    #     controlnet_model_id  = "InstantX/SD3-Controlnet-Canny",
    #     ...
    # )

    #  MODE C: SD3.5 Large img2img  no ControlNet needed (active default) 
    # Canny edges used as init image. Adjust img2img_strength:
    #   0.6 = strong structural preservation, less prompt influence
    #   0.85 = more creative freedom, looser structure
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = PipelineConfig(
        pipeline_mode           = "controlnet_sd35",
        sd3_model_id            = "stabilityai/stable-diffusion-3.5-large",
        controlnet_model_id     = "stabilityai/stable-diffusion-3.5-large-controlnet-canny",

        #  I/O 
        input_video_path        = "input_video.mp4",
        input_video_path_global = "input_videos",
        output_frames_dir       = "output/frames",
        output_video_path       = "output/animation_output.mp4",
        style_image_path        = "style_image.png",   # e.g. "style.jpg" or leave "" to skip

        #  Your 4 keyframe prompts 
        prompt_start            = "Pre-Raphaelite surrealist tableau, deep ancient forest clearing at golden hour, cinematic, 8k",
        prompt_start_mid        = "Pre-Raphaelite surrealist tableau, ancient forest clearing sunsetting, vibrant greens, cinematic, 8k",
        prompt_end_mid          = "Pre-Raphaelite surrealist tableau, ancient forest in the early evening, glowing yellow flowers, cinematic, 8k",
        prompt_end              = "Pre-Raphaelite surrealist tableau, ancient forest, blooming orange flowers, silver light, cinematic, 8k",

        #  Sampling 
        steps                 = 30,
        cfg                   = 3.5,
        controlnet_strength   = 0.80,
        seed                  = 42,
        width                 = 768,
        height                = 432,
        output_fps            = 24,
    )


    try:
        video_files = sorted(
            Path(config.input_video_path_global).glob("input_video_*.mp4")
        )

        if not video_files:
            raise FileNotFoundError(
                f"No input_video_*.mp4 files found in {config.input_video_path_global}"
            )

        print(f"Found {len(video_files)} videos to process.")

        for i, video_path in enumerate(video_files, start=1):
            print(f"\n── Processing video {i}/{len(video_files)}: {video_path.name} ──")
            config.input_video_path  = str(video_path)
            config.output_frames_dir = f"output/frames_{i}"
            config.output_video_path = f"output/animation_output_{i}.mp4"
            run_pipeline(config)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user. Partial results may be saved.")
        exit()
    except:
        print("\nAn error occurred during pipeline execution.")
        raise
