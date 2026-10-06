# Windows Installation Guide

## Portable / embedded ComfyUI

Assuming:

```text
C:\ComfyUI\
├── python_embeded\
├── models\
└── custom_nodes\
```

Put STM here:

```text
C:\ComfyUI\custom_nodes\ComfyUI-STM\
```

Then from the ComfyUI folder:

```powershell
.\python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-STM\requirements.txt
```

Restart ComfyUI after installation.

## Do not replace ComfyUI's Torch

STM intentionally does not require a separate Torch installation in `requirements.txt`.

Use the Torch build already provided by ComfyUI so CUDA/ROCm compatibility stays aligned with the rest of your installation.

## Model directories

### TRELLIS.2

```text
C:\ComfyUI\models\microsoft\TRELLIS.2-4B\
```

Expected top level:

```text
texturing_pipeline.json
ckpts\
```

### DINOv3

```text
C:\ComfyUI\models\facebook\dinov3-vitl16-pretrain-lvd1689m\
```

Accept the model's Hugging Face license before downloading.

### FLUX.2 [klein] 4B

```text
C:\ComfyUI\models\diffusion_models\flux-2-klein-4b.safetensors
C:\ComfyUI\models\text_encoders\qwen_3_4b.safetensors
C:\ComfyUI\models\vae\flux2-vae.safetensors
```

An FP8 diffusion file can also be used when your ComfyUI workflow expects it.

## GGUF

GGUF is optional. If you use a GGUF loader such as ComfyUI-GGUF, install that loader separately and put the corresponding GGUF model in the folder expected by that loader.

STM itself does not require GGUF.

## Existing TRELLIS.2 custom nodes

If you have another TRELLIS.2 texturing pack that registers the legacy node:

```text
Trellis2MeshTexturingStandalone
```

do not leave duplicate copies active without checking their node IDs and imports. Disable the older copy if ComfyUI reports duplicate/ambiguous registration.

## Verification

After restart, confirm that the ComfyUI node search contains:

```text
STM
```

and that at least the following nodes are present:

- STM — Load Models
- STM — Load Mesh
- STM — DINOv3
- STM — Geometry / Dual Grid
- STM — Shape Encoder
- STM — Texture Generation
- STM — Texture Decoder
- STM — Texture Projection
- STM — Apply Texture
