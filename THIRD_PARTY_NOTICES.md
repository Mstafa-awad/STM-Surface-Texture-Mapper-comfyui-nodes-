# Third-Party Notices

This file records the third-party code/model sources that are relevant to the STM repository. Model weights are **not bundled** with this GitHub repository.

## 1. Microsoft TRELLIS.2

Repository:

https://github.com/microsoft/TRELLIS.2

The upstream TRELLIS.2 project is released under the MIT License. STM contains code derived from the TRELLIS.2 implementation and re-expresses required pieces using the project's native PyTorch paths.

Relevant adapted areas in STM include TRELLIS.2-style:

- sparse structure / flow utilities
- scalar/flow timestep helpers
- sparse-convolution and transformer building blocks
- sparse VAE components
- dual-grid / o-voxel algorithm concepts implemented with tensor operations

STM does **not** ship the upstream `o_voxel` compiled extension.

Copyright notice required by the upstream MIT license:

> Copyright (c) Microsoft Corporation.

Upstream license:

https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE

## 2. Visual Bruno — ComfyUI-Trellis2

Repository:

https://github.com/visualbruno/ComfyUI-Trellis2

STM retains/adapts functionality originating in the Visual Bruno ComfyUI TRELLIS.2 texturing/node implementation. The upstream repository is MIT licensed.

This attribution is retained here even though STM removes/replaces several external components used by the broader Visual Bruno package.

Upstream license:

https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/LICENSE

## 3. Trellis2MeshTexturingStandalone

`Trellis2MeshTexturingStandalone` is the author's own codebase. It is not a third-party dependency.

STM keeps the legacy node ID / compatibility interface so existing workflows can continue to refer to:

`Trellis2MeshTexturingStandalone`

The current STM code is the user's own project and is licensed under the repository `LICENSE`.

## 4. FLUX.2 [klein] 4B

The recommended enhancement model is downloaded separately from:

https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

Black Forest Labs lists FLUX.2 [klein] 4B under Apache-2.0. Do not distribute or use the 9B/9B-KV/dev model families as though they had the same license; those are under a non-commercial FLUX license.

Official model overview:

https://github.com/black-forest-labs/flux2

## 5. Qwen3-4B

The recommended FLUX.2 text encoder is:

https://huggingface.co/Qwen/Qwen3-4B

Qwen3-4B is Apache-2.0 licensed.

## 6. DINOv3

The recommended image encoder is:

https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

DINOv3 is distributed under Meta's DINOv3 License, not MIT or Apache-2.0. Its license grants commercial use but includes additional conditions on redistribution, trade controls, prohibited end uses, and related obligations.

The full license should be reviewed before commercial deployment:

https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md

## 7. Python runtime dependencies

STM does not vendor Python wheels or copy third-party site-packages into this repository. Users install dependencies from their package manager / ComfyUI environment. The upstream licenses of those packages remain applicable to those installed packages.

The current direct requirements include:

- NumPy
- SciPy
- Pillow
- safetensors
- Transformers
- tqdm
- psutil
- trimesh

The core commercial-use question for STM is therefore separated into:
1. the STM source license and adapted source notices, and
2. the licenses of the separately downloaded model weights and separately installed runtime packages.

Do not copy a dependency's source or binary into this repository without carrying its own license and notice requirements.

## 8. Optional acceleration packages

STM can detect optional accelerators in environments where they are installed. They are not required by the default path.

Examples include:

- FlashAttention — BSD 3-Clause
- xFormers — BSD-style license
- FlexGEMM — permissive upstream license; see its own repository/package metadata

If you redistribute one of these packages inside a larger distribution, include its exact upstream notice and license text.
