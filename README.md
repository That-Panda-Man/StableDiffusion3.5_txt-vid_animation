# SD3.5 Animated Style Transfer Pipeline

A pure-Python video style transfer pipeline using Stable Diffusion 3.5 (and/or SD3 Medium).
It extracts Canny edge frames from an input video, generates AI-styled frames with scheduled prompts, and compiles them into a smoothly dissolving output video.

Three pipeline modes are supported:

| Mode | Base Model | ControlNet | Notes |
|---|---|---|---|
| `controlnet_sd35` | SD3.5 Large | InstantX SD3.5 Canny | Best quality, needs matching weights |
| `controlnet_sd3` | SD3 Medium | InstantX SD3 Canny | Works reliably, lower quality |
| `img2img` | SD3.5 Large | None | Most compatible fallback |

---

## Requirements

- **NVIDIA GPU** with at least 16 GB VRAM recommended (24 GB for ControlNet modes)
- **CUDA 12.x** drivers installed
- **Conda** (Miniconda or Anaconda)
- **Hugging Face account** — the SD3.5 Large and SD3 Medium models are gated and require you to:
  1. Accept the licence on [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
  2. Accept the licence on [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) (if using SD3 mode)
  3. Generate a Hugging Face access token and run `huggingface-cli login`

---

## Installation

### 1. Create the Conda environment

```bash
conda create -n sd35_anim python=3.11 -y
conda activate sd35_anim
```

### 2. Install PyTorch via pip

Visit **[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)** to select the correct build for your OS and CUDA version. The command below is for **CUDA 12.1** — replace the index URL if you are using a different CUDA version.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install all other dependencies via Conda

```bash
conda install -c conda-forge \
    diffusers \
    transformers \
    accelerate \
    opencv \
    pillow \
    numpy \
    tqdm \
    huggingface_hub \
    -y
```

### 4. Log in to Hugging Face

```bash
hf auth login
```

Paste your access token when prompted. This is required to download the gated SD3.5 weights automatically on first run.

---

## Project Structure

```
animation_V2/
├── V2_sd35_anim_pipeline_style_transfer.py   # Main script
├── input_videos/                              # Place input videos here
│   └── input_video_001.mp4                   # Must follow naming: input_video_*.mp4
├── style_image.png                            # Optional style reference image
├── placeholder_rename_to_style_image.png      # Example style image placeholder
└── output/                                    # Created automatically on first run
    ├── frames_1/                              # Canny + generated frames (per video)
    └── animation_output_1.mp4                 # Final output video (per video)
```

---

## Usage

### 1. Add your input video(s)

Place one or more videos inside the `input_videos/` folder. Files **must** follow the naming convention:

```
input_videos/input_video_001.mp4
input_videos/input_video_002.mp4
...
```

### 2. (Optional) Add a style image

Place a style reference image in the project root and set its filename in `PipelineConfig`:

```python
style_image_path = "style_image.png"
```

Set `style_image_path = ""` to disable IP-Adapter style transfer.

### 3. Configure the pipeline

Open `V2_sd35_anim_pipeline_style_transfer.py` and edit the `PipelineConfig` block near the bottom of the file. Key settings:

```python
config = PipelineConfig(
    # Pipeline mode: "controlnet_sd35" | "controlnet_sd3" | "img2img"
    pipeline_mode         = "controlnet_sd35",

    # Model IDs (adjust for your chosen mode — see table above)
    sd3_model_id          = "stabilityai/stable-diffusion-3.5-large",
    controlnet_model_id   = "stabilityai/stable-diffusion-3.5-large-controlnet-canny",

    # Output resolution, must be in multiples of 16
    width   = 768,
    height  = 432,

    # Sampling quality vs. speed
    steps   = 30,
    cfg     = 3.5,

    # Your four keyframe prompts that determine the visual progression
    prompt_start     = "a misty forest at dawn, golden light, cinematic, 8k",
    prompt_start_mid = "a sunlit forest at midday, vibrant greens, cinematic, 8k",
    prompt_end_mid   = "a forest at golden hour, warm amber tones, cinematic, 8k",
    prompt_end       = "a moonlit forest at night, silver light, cinematic, 8k",
)
```

#### Choosing a pipeline mode

| Your hardware / weights | Recommended mode |
|---|---|
| SD3.5 Large ControlNet weights available | `controlnet_sd35` |
| Only SD3 Medium ControlNet weights | `controlnet_sd3` |
| Low VRAM or no ControlNet weights | `img2img` |

> **Note:** Mixing SD3.5 Large with the SD3 Medium ControlNet will cause a tensor size mismatch error (`size 2432 vs 1536`). Use `img2img` mode if in doubt.

### 4. Run the script

```bash
conda activate sd35_anim
python V2_sd35_anim_pipeline_style_transfer.py
```

The script will:
1. Scan `input_videos/` for all `input_video_*.mp4` files and process them sequentially
2. Extract Canny edge frames from each video
3. Generate styled frames using the configured SD3.5 pipeline
4. Compile the generated frames into a dissolve-transitioned output video

Results are saved to `output/animation_output_1.mp4`, `output/animation_output_2.mp4`, etc.

---

## VRAM Tips

If you run out of VRAM, uncomment one of the memory-optimisation lines in `load_pipeline()`:

```python
# pipe.enable_model_cpu_offload()        # moderate savings, minimal speed loss
# pipe.enable_sequential_cpu_offload()   # maximum savings (~4 GB VRAM), slowest
```

---

## Resuming a Partial Run

The script automatically skips already-generated frames. If a run is interrupted, simply re-run the same command and it will continue from where it left off.

---

## Models Used

| Model | Hugging Face Hub |
|---|---|
| Stable Diffusion 3.5 Large | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| SD3.5 ControlNet Canny | [stabilityai/stable-diffusion-3.5-large-controlnet-canny](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-controlnet-canny) |
| SD3 Medium | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) |
| SD3 ControlNet Canny | [InstantX/SD3-Controlnet-Canny](https://huggingface.co/InstantX/SD3-Controlnet-Canny) |
| IP-Adapter (style transfer) | [InstantX/SD3.5-Large-IP-Adapter](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) |
| SigLIP Image Encoder | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |
