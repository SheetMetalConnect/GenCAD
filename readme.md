<p align="center">
  <h1 align="center">GenCAD</h1>
  <h4 align="center">Image-conditioned Computer-Aided Design Generation with Transformer-based Contrastive Representation and Diffusion Priors</h4>
</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=e817c1wEZ6">
    <img src="https://img.shields.io/badge/Paper-TMLR%202025-4b44ce.svg">
  </a>
  <a href="https://arxiv.org/abs/2409.16294">
    <img src="https://img.shields.io/badge/arXiv-2409.16294-b31b1b.svg">
  </a>
  <a href="https://gencad.github.io/">
    <img src="https://img.shields.io/badge/Project%20Page-Link-blue">
  </a>
</p>

---

<p align="center">
  <img src="assets/fig_10.png" alt="GenCAD Demo" width="700"/>
</p>

---

## Overview

GenCAD generates 3D CAD models from 2D images using a three-stage pipeline:

1. **Edge Detection** — Canny filter extracts edges from the input image
2. **CLIP Encoding + Diffusion Prior** — image is encoded via a contrastive model, then a diffusion prior generates a CAD latent (500 sampling steps)
3. **CAD Decoder** — transformer decodes the latent into parametric CAD commands (lines, arcs, circles, extrusions with boolean operations)

The output is a real parametric CAD solid (B-rep via OpenCASCADE), not a mesh. STL export is available for visualization and 3D printing.

---

## Quick Start — Web UI (Docker)

The fastest way to run GenCAD is with the built-in Gradio web interface.

### 1. Prerequisites

- Docker or Podman
- ~30 GB disk space (dataset + checkpoints + container image)
- No GPU required (runs on CPU, supports CUDA and Apple Silicon MPS)

### 2. Download Data and Checkpoints

**Dataset** — download from [Google Drive](https://drive.google.com/drive/folders/1M0dPr5kILGY9HTRCHox1vLLDhhxJWl_C?usp=sharing) and place in `data/`:

```bash
# Expected structure after download + extraction:
data/
  cad_vec/          # CAD vector sequences
  embeddings/       # Pre-computed embeddings (.h5 files)
  images/           # Rendered CAD images (organized in folders 0000-0099)
  sketches/         # Sketch images
```

**Pretrained Models** — download from [Google Drive](https://drive.google.com/drive/folders/1Ej7wdtlqT5P-SoUf3gsZXD8b78XqhiI5?usp=sharing) and place in `model/ckpt/`:

```bash
# Required checkpoints:
model/ckpt/
  ae_ckpt_epoch1000.pth                        # CAD autoencoder (CSR)
  ccip_sketch_ckpt_epoch300.pth                # CLIP adapter (CCIP)
  sketch_cond_diffusion_ckpt_epoch1000000.pt   # Diffusion prior
```

### 3. Build and Run

```bash
git clone https://github.com/SheetMetalConnect/GenCAD
cd GenCAD

# Build the container (~18 GB image, takes 5-10 min first time)
docker build -t gencad:latest .

# Run the web UI on port 7860
docker run -d --name gencad-web \
  -p 7860:7860 \
  -v ./data:/app/data:ro \
  -v ./model/ckpt:/app/model/ckpt:ro \
  -v ./examples:/app/examples:ro \
  gencad:latest

# Open http://localhost:7860 in your browser
```

Models take ~15-30 seconds to load on first start. Check logs with `docker logs -f gencad-web`.

### 4. Using the Web UI

1. Upload a PNG image of a mechanical part, or click one of the example images
2. Click **Generate CAD**
3. Watch the pipeline steps: edge detection → CLIP encoding → diffusion → CAD decoding
4. View the generated 3D model in the interactive viewer (rotate/zoom/pan)
5. Download the STL file

Generation is **stochastic** — the diffusion prior uses random noise, so each run produces a different result. Click Generate multiple times and compare in the history gallery.

---

## Podman (Linux / Fedora / RHEL)

For Podman, use `--format docker` for the build and `:Z` for SELinux volume labels:

```bash
podman build --format docker -t gencad:latest .

podman run -d --name gencad-web \
  -p 7860:7860 \
  -v ./data:/app/data:Z \
  -v ./model/ckpt:/app/model/ckpt:Z \
  -v ./examples:/app/examples:Z \
  gencad:latest
```

---

## CLI Inference (no web UI)

For batch processing or headless servers:

```bash
# Put your input images (PNG) in a folder
mkdir my_images
cp your_part_photo.png my_images/

# Run inference with STL + rendered image export
docker run --rm \
  -v ./data:/app/data:ro \
  -v ./model/ckpt:/app/model/ckpt:ro \
  -v ./my_images:/app/my_images \
  gencad:latest conda run -n gencad_env \
  xvfb-run --server-args="-screen 0 2048x2048x24" \
  python inference_gencad.py -image_path my_images -export_stl -export_img

# Results appear in my_images/stls/ and my_images/generated_images/
```

---

## Manual Setup (conda + pip)

For development or if you prefer not to use Docker:

```bash
# 1. Create conda environment
conda create -n gencad_env python=3.10 -y
conda activate gencad_env

# 2. Install pythonocc-core (must be via conda, not pip)
conda install -c conda-forge pythonocc-core=7.9.0

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the web UI
python demo.py

# Or run CLI inference
python inference_gencad.py -image_path path/to/images -export_stl -export_img
```

**Note for headless Linux servers:** prefix commands with `xvfb-run --server-args="-screen 0 2048x2048x24"` to enable offscreen OpenGL rendering.

**Note for macOS (Apple Silicon):** the code automatically uses MPS acceleration. No xvfb needed — native display is used.

---

## Training

### CSR Model (CAD Sequence Representation)
```bash
python train_gencad.py csr -name test -gpu 0
```
With checkpoint:
```bash
python train_gencad.py csr -name test -gpu 0 -ckpt "model/ckpt/ae_ckpt_epoch1000.pth"
```

### CCIP Model (Contrastive CAD-Image Pretraining)
```bash
python train_gencad.py ccip -name test -gpu 0 -cad_ckpt "model/ckpt/ae_ckpt_epoch1000.pth"
```

### Diffusion Prior
```bash
python train_gencad.py dp -name test -gpu 0 \
  -cad_emb 'data/embeddings/cad_embeddings.h5' \
  -img_emb 'data/embeddings/sketch_embeddings.h5'
```

---

## STL Visualization

Convert STL files to PNG renders:
```bash
python stl2img.py -src path/to/stl/files -dst path/to/save/images
```

---

## Architecture

```
Input Image (PNG)
     │
     ▼
┌─────────────┐
│ Canny Edge  │  Edge detection (100/200 thresholds)
│ Detection   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ResNet-18   │  Image encoder (CCIP)
│ CLIP Adapter│  → 256-dim image embedding
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Diffusion   │  500-step denoising (ResNet backbone)
│ Prior       │  image embedding → CAD latent (256-dim)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Transformer │  CAD sequence decoder
│ Decoder     │  latent → commands (Line, Arc, Circle, Extrude)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ OpenCASCADE │  B-rep solid construction
│ (pythonocc) │  → STL export / 3D visualization
└─────────────┘
```

### Device Support

The code automatically selects the best available device:
1. **Apple Silicon (MPS)** — Metal Performance Shaders
2. **NVIDIA GPU (CUDA)** — any CUDA-capable GPU
3. **CPU** — fallback, ~10 seconds per image

---

## Troubleshooting

### "cannot find EOS"
This is expected model behavior, not a bug. The model doesn't produce a valid CAD sequence for every input. The success rate depends on how similar the input is to the training data (DeepCAD dataset). Try:
- Different input images
- Running the same image multiple times (stochastic generation)
- Using rendered CAD views or technical sketches rather than photographs

### Docker build is slow
The first build downloads ~18 GB of dependencies (PyTorch, CUDA libs, conda packages). Subsequent builds use cached layers and complete in seconds.

### `.dockerignore` — what's excluded
The `data/` and `model/ckpt/` directories are excluded from the Docker build context and mounted as volumes at runtime. This keeps the build context small (~3 MB) and the build fast.

### Podman SHELL warning
If you see `SHELL is not supported for OCI image format`, use `--format docker`:
```bash
podman build --format docker -t gencad:latest .
```

### SELinux "Permission denied"
On Fedora/RHEL with SELinux, add `:Z` to volume mounts:
```bash
-v ./data:/app/data:Z
```

---

## Evaluation

Coming soon.

---

## Citation

```bibtex
@article{alam2024gencad,
  title={Image-conditioned Computer-Aided Design Generation with Transformer-based Contrastive Representation and Diffusion Priors},
  author={Alam, Md Ferdous and Asprion, Fabian and Maier, Stefan},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2025}
}
```
