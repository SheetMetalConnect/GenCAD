"""GenCAD Demo — Full pipeline visualization: Image to parametric CAD."""

import os
import glob
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import (
    GaussianDiffusion1D, ResNetDiffusion, VanillaCADTransformer,
    CLIP, ResNetImageEncoder, GenCADClipAdapter,
)
from utils import logits2vec
from config import ConfigAE
from cadlib.macro import EOS_IDX, MAX_TOTAL_LEN, ALL_COMMANDS
from cadlib.visualize import vec2CADsolid
from OCC.Extend.DataExchange import write_stl_file


# ---------------------------------------------------------------------------
# Device + global model state
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

DIFFUSION = None
CLIP_ADAPTER = None
CAD_DECODER = None


def load_models():
    global DIFFUSION, CLIP_ADAPTER, CAD_DECODER

    print(f"[GenCAD] Device: {DEVICE}")

    # Diffusion prior
    resnet_params = dict(
        d_in=256, n_blocks=10, d_main=2048, d_hidden=2048,
        dropout_first=0.1, dropout_second=0.1, d_out=256,
    )
    model = ResNetDiffusion(**resnet_params)
    DIFFUSION = GaussianDiffusion1D(
        model, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False,
    )
    ckpt = torch.load("model/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt", map_location="cpu")
    DIFFUSION.load_state_dict(ckpt["model"])
    DIFFUSION = DIFFUSION.to(DEVICE).eval()
    print("[GenCAD] Diffusion prior loaded")

    # CCIP
    cfg_cad = ConfigAE(phase="test", device=DEVICE, overwrite=False)
    cad_encoder = VanillaCADTransformer(cfg_cad)
    image_encoder = ResNetImageEncoder(network="resnet-18")
    clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=256)
    clip_ckpt = torch.load("model/ckpt/ccip_sketch_ckpt_epoch300.pth", map_location="cpu")
    clip.load_state_dict(clip_ckpt["model_state_dict"])
    clip.eval()
    CLIP_ADAPTER = GenCADClipAdapter(clip=clip).to(DEVICE)
    print("[GenCAD] CCIP loaded")

    # CAD decoder
    config = ConfigAE(exp_name="test", phase="test", batch_size=1, device=DEVICE, overwrite=False)
    CAD_DECODER = VanillaCADTransformer(config).to(DEVICE)
    cad_ckpt = torch.load("model/ckpt/ae_ckpt_epoch1000.pth", map_location="cpu")
    CAD_DECODER.load_state_dict(cad_ckpt["model_state_dict"])
    CAD_DECODER.eval()
    print("[GenCAD] CAD decoder loaded")
    print("[GenCAD] All models ready.")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def apply_canny(pil_image):
    """Canny edge detection — returns PIL RGB image."""
    rgb = np.array(pil_image.convert("RGB"))
    edges = cv2.Canny(rgb, 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=2))


# ---------------------------------------------------------------------------
# CAD sequence description
# ---------------------------------------------------------------------------

def describe_cad_sequence(out_vec):
    """Human-readable summary of the decoded CAD commands."""
    commands = out_vec[:, 0]
    counts = {"Line": 0, "Arc": 0, "Circle": 0, "Ext": 0}
    for cmd in commands:
        idx = int(cmd)
        if 0 <= idx < len(ALL_COMMANDS):
            name = ALL_COMMANDS[idx]
            if name in counts:
                counts[name] += 1
    parts = []
    if counts["Line"]:
        parts.append(f"{counts['Line']} lines")
    if counts["Arc"]:
        parts.append(f"{counts['Arc']} arcs")
    if counts["Circle"]:
        parts.append(f"{counts['Circle']} circles")
    if counts["Ext"]:
        parts.append(f"{counts['Ext']} extrusions")
    return ", ".join(parts) if parts else "empty sequence"


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def score_cad_vec(cad_vec):
    """Score a CAD sequence by geometric complexity. Higher = more interesting."""
    counts = {"Line": 0, "Arc": 0, "Circle": 0, "Ext": 0}
    for cmd in cad_vec[:, 0]:
        idx = int(cmd)
        if 0 <= idx < len(ALL_COMMANDS):
            name = ALL_COMMANDS[idx]
            if name in counts:
                counts[name] += 1
    # Reward: more sketch elements + more extrusions = more complex geometry
    return counts["Line"] + counts["Arc"] * 2 + counts["Circle"] * 2 + counts["Ext"] * 5


def single_diffusion_attempt(image_embed):
    """Run one diffusion + decode attempt. Returns (cad_vec, score) or None."""
    latent = DIFFUSION.sample(cond=image_embed)
    latent = latent.unsqueeze(0)

    with torch.no_grad():
        outputs = CAD_DECODER(None, None, z=latent, return_tgt=False)
        batch_out_vec = logits2vec(outputs, device=DEVICE)
        begin_loop_vec = np.full(
            (batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64,
        )
        begin_loop_vec[:, :, 0] = 4  # SOL
        auto_batch = np.concatenate(
            [begin_loop_vec, batch_out_vec], axis=1,
        )[:, :MAX_TOTAL_LEN, :]

    out_vec = auto_batch[0]
    out_command = out_vec[:, 0]

    # Must have valid EOS
    try:
        seq_len = out_command.tolist().index(EOS_IDX)
    except ValueError:
        return None

    cad_vec = out_vec[:seq_len]

    # Must produce valid geometry
    try:
        out_shape = vec2CADsolid(cad_vec.astype(float))
    except Exception:
        return None

    return (cad_vec, out_shape, score_cad_vec(cad_vec))


def run_inference(image, num_attempts=10, progress=gr.Progress()):
    """
    Best-of-N pipeline: run N diffusion attempts, pick the most complex valid result.
    Returns: (canny_image, stl_path, status_text)
    """
    if image is None:
        return None, None, "Upload an image first."

    t0 = time.time()
    pil_image = image

    # Step 1: Edge detection
    progress(0.05, desc="Step 1/5 — Edge detection")
    canny_img = apply_canny(pil_image)
    tensor = PREPROCESS(canny_img).unsqueeze(0).to(DEVICE)

    # Step 2: CLIP embedding (once — reused across all attempts)
    progress(0.10, desc="Step 2/5 — CLIP image encoding")
    image_embed = CLIP_ADAPTER.embed_image(tensor, normalization=False)

    # Step 3+4: Best-of-N diffusion + decode
    candidates = []
    for i in range(num_attempts):
        pct = 0.15 + (i / num_attempts) * 0.70
        progress(pct, desc=f"Step 3/5 — Attempt {i+1}/{num_attempts} (diffusion + decode)...")
        result = single_diffusion_attempt(image_embed)
        if result is not None:
            candidates.append(result)

    if not candidates:
        elapsed = time.time() - t0
        return (
            canny_img, None,
            f"All {num_attempts} attempts failed to produce valid geometry. "
            f"Try a different image. ({elapsed:.1f}s)"
        )

    # Pick highest-scoring candidate
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_vec, best_shape, best_score = candidates[0]
    summary = describe_cad_sequence(best_vec)

    # Step 5: Export STL
    progress(0.90, desc="Step 5/5 — Exporting best result as STL")
    stl_dir = tempfile.mkdtemp()
    stl_path = os.path.join(stl_dir, "generated.stl")
    write_stl_file(
        best_shape, stl_path, mode="binary",
        linear_deflection=0.5, angular_deflection=0.3,
    )

    elapsed = time.time() - t0
    status = (
        f"Best of {len(candidates)}/{num_attempts} valid results\n"
        f"Generated: {summary} (score: {best_score})\n"
        f"Time: {elapsed:.1f}s | Device: {DEVICE}"
    )
    return canny_img, stl_path, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app():
    example_images = sorted(glob.glob("examples/*.png"))

    with gr.Blocks(
        title="GenCAD — Image to CAD",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css="""
            .pipeline-step { border-left: 3px solid #3b82f6; padding-left: 12px; margin-bottom: 8px; }
            footer { display: none !important; }
        """,
    ) as app:

        gr.Markdown(
            "# GenCAD — Image to CAD\n"
            "### Generate parametric 3D CAD models from 2D images\n\n"
            "Upload a photo, sketch, or rendered view of a mechanical part. "
            "GenCAD extracts edges, encodes the image via a contrastive model, "
            "runs a 500-step diffusion process to generate a CAD latent, then decodes it "
            "into real parametric geometry (sketch profiles + extrusions with boolean operations).\n\n"
            "The output is a B-rep solid — not a mesh approximation — exported as STL for visualization and downstream use."
        )

        with gr.Row(equal_height=True):
            # --- Left: input ---
            with gr.Column(scale=1):
                gr.Markdown("#### 1. Input")
                input_image = gr.Image(
                    label="Drop an image here",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=300,
                )
                attempts_slider = gr.Slider(
                    minimum=1, maximum=50, value=20, step=1,
                    label="Attempts (best-of-N)",
                    info="More attempts = better results. With GPU: ~1s per attempt.",
                )
                generate_btn = gr.Button(
                    "Generate CAD Model", variant="primary", size="lg",
                )
                gr.Markdown(
                    "<small>Accepts PNG, JPG, or any image format. "
                    "Best results with clean renders or technical sketches of single parts.</small>"
                )

            # --- Center: pipeline intermediate ---
            with gr.Column(scale=1):
                gr.Markdown("#### 2. Edge Detection")
                canny_output = gr.Image(
                    label="Canny edges — this is what the neural network sees",
                    type="pil",
                    height=300,
                    interactive=False,
                )
                status_box = gr.Textbox(
                    label="Pipeline Log",
                    lines=3,
                    interactive=False,
                    placeholder="Waiting for input...",
                )

            # --- Right: 3D result ---
            with gr.Column(scale=2):
                gr.Markdown("#### 3. Generated CAD Model")
                model_viewer = gr.Model3D(
                    label="Interactive 3D viewer — drag to rotate, scroll to zoom",
                    height=400,
                    clear_color=[0.92, 0.92, 0.92, 1.0],
                )
                stl_download = gr.File(label="Download STL File")

        # --- How it works ---
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "**Image in, CAD out — in five steps:**\n\n"
                "1. **Edge Detection** — Canny filter extracts the contours your part\n"
                "2. **Image Encoding** — ResNet-18 compresses the edges into a 256-dimensional vector\n"
                "3. **Diffusion Sampling** — 500 denoising steps generate a CAD-space latent from that vector "
                "(repeated N times with different noise for best-of-N selection)\n"
                "4. **CAD Decoding** — A transformer converts the latent into parametric sketch commands "
                "(lines, arcs, circles) and extrusion operations with boolean cuts/fuses\n"
                "5. **Solid Construction** — OpenCASCADE builds the B-rep solid and exports STL\n\n"
                "On GPU each attempt takes about 1 second. "
                "The slider controls how many attempts to try — more attempts, better results."
            )

        # --- History ---
        gr.Markdown("### Generation History")
        history_gallery = gr.Gallery(
            label="Previous results (most recent first)",
            columns=6,
            height=140,
            object_fit="contain",
        )
        history_state = gr.State([])

        def generate(image, attempts, history):
            canny, stl_path, status = run_inference(image, num_attempts=int(attempts))
            # Save canny to a temp file for the gallery (Gradio 6.x needs file paths)
            history = list(history) if history else []
            if canny is not None:
                canny_path = os.path.join(tempfile.mkdtemp(), "canny.png")
                canny.save(canny_path)
                history.insert(0, canny_path)
                history = history[:18]
            return canny, stl_path, status, stl_path, history, history

        generate_btn.click(
            fn=generate,
            inputs=[input_image, attempts_slider, history_state],
            outputs=[
                canny_output, model_viewer, status_box, stl_download,
                history_gallery, history_state,
            ],
        )

        # --- Examples ---
        if example_images:
            gr.Markdown("---\n### Example Images\n*Click to load, then hit Generate*")
            gr.Examples(
                examples=[[img] for img in example_images],
                inputs=input_image,
                examples_per_page=20,
            )

        # --- Credits ---
        gr.Markdown(
            "---\n"
            "<small>\n\n"
            "**GenCAD** — Image-conditioned CAD Generation with Transformer-based "
            "Contrastive Representation and Diffusion Priors\n\n"
            "**Research:** Md Ferdous Alam, Fabian Asprion, Stefan Maier "
            "([TMLR 2025](https://openreview.net/pdf?id=e817c1wEZ6) "
            "| [arXiv 2409.16294](https://arxiv.org/abs/2409.16294))\n\n"
            "**Web demo & deployment:** Luke van Enkhuizen / "
            "[SheetMetalConnect](https://github.com/SheetMetalConnect/GenCAD)\n\n"
            "</small>"
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
