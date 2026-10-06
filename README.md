# STM: Surface Texture Mapper

**ComfyUI custom nodes for TRELLIS.2 mesh texturing, UV-aware texture projection, and optional FLUX.2 [klein] 4B texture enhancement.**

STM is designed to keep the core TRELLIS.2 texturing path usable without the non-commercial / separately restricted native components used by some other wrappers. The project uses native PyTorch implementations for the included geometry, sparse sampling, attention, and projection stages and does **not** bundle `nvdiffrast`, the `o_voxel` binary, CuMesh, or MeshLib.

> **Commercial-use status**
>
> The **STM source code is released under MIT**. The recommended model stack is commercially usable, but it is not correct to describe every possible use as completely unrestricted: **DINOv3 is under Meta's DINOv3 License and has additional end-use and redistribution conditions**. See [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md) and [`MODEL_LICENSES.md`](MODEL_LICENSES.md).
>
> For ordinary commercial 3D assets, games, visualization, texturing, and similar use cases, use the recommended model versions listed below and comply with their individual licenses.

## What STM does

STM breaks the TRELLIS.2 texturing process into modular ComfyUI nodes:

```text
STM — Load Mesh
      │
      ├── STM — UV Unwrap
      │
      ├── STM — Geometry / Dual Grid
      │        └── STM — Shape Encoder
      │
      ├── Load Image → STM — DINOv3
      │
      ├──────────────→ STM — Texture Generation
      │                         │
      │                         └── STM — Texture Decoder
      │
      └────────────────────────→ STM — Texture Projection
                                   │
                                   └── STM — Apply Texture → MESH
```

STM also includes an optional enhancement path for **already-textured meshes**:

```text
Textured MESH
   ↓
STM — Render Views (unlit)
   ↓
STM — Split Tiles
   + reference image
   ↓
FLUX.2 [klein] 4B
   ↓
STM — Merge Tiles
   ↓
STM — Project Views (Detail)
   ↓
Updated textured MESH
```

The enhancer is image-model agnostic. FLUX.2 [klein] 4B is the recommended example because the 4B model is Apache-2.0 licensed.

## Install

### 1. Install the node pack

Copy the `ComfyUI-STM` folder into:

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-STM/
```

If you have an older node pack that registers the legacy `Trellis2MeshTexturingStandalone` node with the same node ID, disable or remove that older copy to avoid duplicate registration and ambiguous imports.

### 2. Install Python dependencies

Use the Python interpreter that belongs to your ComfyUI installation.

For **Windows portable / embedded ComfyUI**:

```powershell
cd C:\path\to\ComfyUI
.\python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-STM\requirements.txt
```

For a normal Python environment:

```bash
python -m pip install -r requirements.txt
```

**Do not install a separate Torch build just for STM.** ComfyUI normally provides the correct PyTorch/CUDA/ROCm stack for the installation. Installing another Torch build into the same environment can break the rest of ComfyUI.

### 3. Restart ComfyUI

Restart ComfyUI completely after installation so the STM node mappings are loaded.

## Models

**Model weights are not included in this repository. Download them from their original sources and keep them outside GitHub.**

### TRELLIS.2-4B

Recommended directory:

```text
ComfyUI/
└── models/
    └── microsoft/
        └── TRELLIS.2-4B/
            ├── texturing_pipeline.json
            └── ckpts/
                ├── shape_enc/
                ├── tex_dec/
                └── flow/
```

Use the official Microsoft `TRELLIS.2-4B` model package and preserve its model files and license terms.

STM's `STM — Load Models` node expects the folder containing `texturing_pipeline.json` and the corresponding `ckpts/` tree.

### DINOv3 ViT-L/16

Recommended directory:

```text
ComfyUI/
└── models/
    └── facebook/
        └── dinov3-vitl16-pretrain-lvd1689m/
            └── <DINOv3 model files>
```

Source:

- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

You must accept Meta's **DINOv3 License** on Hugging Face before downloading/using the weights.

### FLUX.2 [klein] 4B enhancement model

For the native ComfyUI diffusion loader, use:

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── flux-2-klein-4b.safetensors
    ├── text_encoders/
    │   └── qwen_3_4b.safetensors
    └── vae/
        └── flux2-vae.safetensors
```

ComfyUI may also provide an FP8 variant such as:

```text
ComfyUI/models/diffusion_models/flux-2-klein-4b-fp8.safetensors
```

Official sources:

- FLUX.2: https://github.com/black-forest-labs/flux2
- FLUX.2 [klein] 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B

The **4B** model is the commercial-use model. Do **not** swap it for:

- `FLUX.2 [klein] 9B`
- `FLUX.2 [klein] 9B KV`
- `FLUX.2 [dev]`

Those model families use the FLUX Non-Commercial License.

#### Optional GGUF route

You can use compatible ComfyUI-GGUF nodes and GGUF versions of the 4B components when you specifically need quantized loading. GGUF is an optional loading format; STM itself does not require the GGUF custom node pack.

## Using the native ComfyUI TRELLIS.2 loader

STM can also be used as modular geometry / projection stages around the official native TRELLIS.2 ComfyUI nodes.

For that setup, keep your normal ComfyUI model loaders for the TRELLIS.2 VAE / diffusion model and use STM for the mesh-side stages:

```text
STM — Load Mesh
  ↓
STM — Mesh Encoder (Native Latent)
  ↓
native TRELLIS.2 conditioning / sampler
  ↓
native TRELLIS.2 VAE decode
  ↓
STM — Texture Projection
  ↓
STM — Apply Texture
```

The native encoder path is intentionally implemented without the `o_voxel` binary.

## Main STM nodes

The node category is **STM**.

Key nodes include:

- `STM — Load Mesh`
- `STM — Load Models`
- `STM — DINOv3`
- `STM — UV Unwrap`
- `STM — Geometry / Dual Grid`
- `STM — Shape Encoder`
- `STM — Texture Generation`
- `STM — Texture Decoder`
- `STM — Texture Projection`
- `STM — Apply Texture`
- `STM — Render Views`
- `STM — Split Tiles`
- `STM — Merge Tiles`
- `STM — Project Views (Detail)`

The legacy compatibility node `Trellis2MeshTexturingStandalone` is also retained.

## Texture enhancement notes

The enhancement system is intended for any UV-mapped mesh that already has a usable base-colour texture.

It can refine:

- fine surface detail
- sharper edges
- coloured symbols / letters / ornaments
- details supplied by a separate reference image

The pipeline renders the existing texture as **unlit** views before sending tiles to the image model, then projects only the recovered detail back onto the mesh. This helps avoid baking new directional lighting into the albedo.

Use the supplied workflow(s) from your own local workflow collection or build the graph from the nodes above. Workflow JSON files are intentionally not required for the custom node package itself.

## Commercial / licensing notes

The project code is MIT. Third-party model and dependency licenses still apply.

| Component | License | Commercial use |
|---|---|---|
| STM source code | MIT | Yes |
| Microsoft TRELLIS.2 code/model | MIT | Yes, subject to MIT terms |
| Meta DINOv3 ViT-L/16 | DINOv3 License | Commercial use permitted, with additional conditions |
| FLUX.2 [klein] 4B | Apache-2.0 | Yes |
| Qwen3-4B | Apache-2.0 | Yes |
| FLUX.2 VAE | Apache-2.0 | Yes |

See:

- [`LICENSE`](LICENSE)
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- [`MODEL_LICENSES.md`](MODEL_LICENSES.md)
- [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md)

### Important

This repository does **not** grant rights to:

- input meshes or textures you do not own or have permission to use
- reference photographs or images you do not have rights to use
- trademarks, logos, characters, or other third-party IP appearing in an input/reference image
- other custom nodes installed in the user's ComfyUI environment

Model licenses govern the model; they do not automatically clear the underlying content used as inputs.

## What this repository intentionally does not use

The STM core does not require or bundle these components from other TRELLIS.2 wrappers:

```text
nvdiffrast
o_voxel native binary
CuMesh
MeshLib
OpenCV-based texturing path
```

The included geometry / projection implementations use PyTorch tensor operations instead.

Optional accelerators may be detected at runtime (for example, FlashAttention, xFormers, or FlexGEMM), but they are **not required for the default path**. Their own licenses apply if installed separately.

## Attribution

STM was developed as a modular continuation of the author's own:

**Trellis2MeshTexturingStandalone**

The repository also contains portions adapted from or derived from work in:

- Microsoft `TRELLIS.2`
- Visual Bruno `ComfyUI-Trellis2`

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the attribution and license notices.

## Troubleshooting

### Nodes do not appear

Restart ComfyUI and check the console for the STM import error.

### Old TRELLIS.2 node keeps loading

A second node pack may be registering the same legacy node ID. Disable the older `ComfyUI-Trellis2MeshTexturing` / equivalent copy.

### DINOv3 download/authentication fails

Accept the DINOv3 license on Hugging Face, then download the model files into the exact `ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m` folder.

### Enhancement workflow cannot find FLUX

Check the ComfyUI model directories:

```text
models/diffusion_models/
models/text_encoders/
models/vae/
```

and verify the filenames match the loader node you are using.

## License

STM source code is released under the MIT License. See [`LICENSE`](LICENSE).
