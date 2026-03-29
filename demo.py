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

def run_inference(image, progress=gr.Progress()):
    """
    Full pipeline: image -> canny -> CLIP -> diffusion -> decode -> 3D solid -> STL.
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

    # Step 2: CLIP embedding
    progress(0.10, desc="Step 2/5 — CLIP image encoding")
    image_embed = CLIP_ADAPTER.embed_image(tensor, normalization=False)

    # Step 3: Diffusion (bulk of the time)
    progress(0.15, desc="Step 3/5 — Diffusion sampling (500 steps)...")
    latent = DIFFUSION.sample(cond=image_embed)
    latent = latent.unsqueeze(0)

    # Step 4: Decode CAD sequence
    progress(0.80, desc="Step 4/5 — Decoding CAD commands")
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

    # Find EOS
    try:
        seq_len = out_command.tolist().index(EOS_IDX)
    except ValueError:
        elapsed = time.time() - t0
        return (
            canny_img, None,
            f"No valid CAD sequence (no EOS). Try again — diffusion is stochastic. ({elapsed:.1f}s)"
        )

    cad_vec = out_vec[:seq_len]
    summary = describe_cad_sequence(cad_vec)

    # Step 5: Build 3D geometry + export
    progress(0.90, desc="Step 5/5 — Building 3D geometry + STL export")
    try:
        out_shape = vec2CADsolid(cad_vec.astype(float))
    except Exception:
        elapsed = time.time() - t0
        return (
            canny_img, None,
            f"Decoded: {summary} — but geometry construction failed. Try again. ({elapsed:.1f}s)"
        )

    stl_dir = tempfile.mkdtemp()
    stl_path = os.path.join(stl_dir, "generated.stl")
    write_stl_file(
        out_shape, stl_path, mode="binary",
        linear_deflection=0.5, angular_deflection=0.3,
    )

    elapsed = time.time() - t0
    status = f"Generated: {summary}\nTime: {elapsed:.1f}s | Device: {DEVICE}"
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
            "Upload an image of a mechanical part or pick an example below. "
            "The pipeline runs edge detection, CLIP encoding, 500-step diffusion, "
            "and CAD sequence decoding to produce a parametric 3D solid.\n\n"
            "**Stochastic generation** — hit Generate multiple times for different results from the same image."
        )

        with gr.Row(equal_height=True):
            # --- Left: input ---
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=300,
                )
                generate_btn = gr.Button(
                    "Generate CAD", variant="primary", size="lg",
                )

            # --- Center: pipeline intermediate ---
            with gr.Column(scale=1):
                canny_output = gr.Image(
                    label="Edge Detection (model input)",
                    type="pil",
                    height=300,
                    interactive=False,
                )
                status_box = gr.Textbox(
                    label="Pipeline Status",
                    lines=3,
                    interactive=False,
                )

            # --- Right: 3D result ---
            with gr.Column(scale=2):
                model_viewer = gr.Model3D(
                    label="3D Result (rotate / zoom / pan)",
                    height=400,
                    clear_color=[0.92, 0.92, 0.92, 1.0],
                )
                stl_download = gr.File(label="Download STL")

        # --- History ---
        gr.Markdown("---\n### Generation History")
        gr.Markdown(
            "*Each generation uses different diffusion noise. "
            "Compare results and download the best one.*"
        )
        history_gallery = gr.Gallery(
            label="Previous results (most recent first)",
            columns=8,
            height=120,
            object_fit="contain",
        )
        history_state = gr.State([])

        def generate_with_history(image, history):
            canny, stl_path, status = run_inference(image)
            download = stl_path
            history = list(history) if history else []
            if canny is not None:
                history.insert(0, canny)
                history = history[:24]
            return canny, stl_path, status, download, history, history

        generate_btn.click(
            fn=generate_with_history,
            inputs=[input_image, history_state],
            outputs=[
                canny_output, model_viewer, status_box,
                stl_download, history_gallery, history_state,
            ],
        )

        # --- Examples ---
        if example_images:
            gr.Markdown("---\n### Example Images")
            gr.Examples(
                examples=[[img] for img in example_images],
                inputs=input_image,
                label="Click to load, then hit Generate",
                examples_per_page=20,
            )

        gr.Markdown(
            "<br><small>GenCAD — "
            "[Paper (TMLR 2025)](https://openreview.net/pdf?id=e817c1wEZ6) · "
            "[arXiv](https://arxiv.org/abs/2409.16294) · "
            "[GitHub](https://github.com/SheetMetalConnect/GenCAD)"
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
