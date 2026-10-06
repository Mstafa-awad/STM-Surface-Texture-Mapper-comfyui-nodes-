# STM License / Dependency Audit

**Audit basis:** uploaded project ZIP examined on 2026-10-06, plus current upstream license pages checked against the repositories/model cards referenced below.

## Executive result

### Source-code release

**PASS — STM can be published as an MIT-licensed GitHub project, provided the included upstream attribution notices stay with the adapted portions.**

The original uploaded ZIP already contained clear signs that STM is based on:

- the author's `Trellis2MeshTexturingStandalone`
- Microsoft TRELLIS.2
- Visual Bruno's `ComfyUI-Trellis2`

The original ZIP did **not** contain the referenced `docs/LICENSE_AUDIT.md`; this GitHub-ready package now includes it.

### Commercial output

**PASS for ordinary commercial use with a DINOv3 license qualifier.**

The recommended stack is commercially usable, but there is no honest basis for calling it "unrestricted commercial use in every field" because Meta's DINOv3 License contains additional end-use restrictions and redistribution conditions.

For normal commercial texture / asset generation, use the recommended model versions and follow their licenses.

## Repository inspection

The uploaded ZIP contains Python source only plus:

- `README.md`
- `LICENSE`
- `requirements.txt`

No model weights were bundled.

Static import inspection found no active imports for:

- `nvdiffrast`
- `o_voxel`
- `cumesh`
- `meshlib`
- `pymeshlab`
- `rembg`
- OpenCV (`cv2`)

The source does mention some of these packages in documentation/comments because the project intentionally removed or replaced them. A textual mention is not a runtime dependency.

The optional runtime backends in the source are separated from the default path and are not listed as required packages in `requirements.txt`.

## Why this is safer than simply copying a TRELLIS.2 wrapper

The upstream Microsoft TRELLIS.2 texturing stack and many ComfyUI wrappers can involve native components such as:

- `o_voxel`
- `nvdiffrast`
- CuMesh
- MeshLib
- other optional/native acceleration packages

STM's included path replaces the relevant native geometry/sampling work with PyTorch implementations and does not bundle those components.

This matters because a repository being MIT licensed does not automatically make every dependency of the repository commercially unrestricted.

## Upstream source licenses

### Microsoft TRELLIS.2

Source:

https://github.com/microsoft/TRELLIS.2

License: MIT.

STM preserves Microsoft attribution in the repository notices.

### Visual Bruno ComfyUI-Trellis2

Source:

https://github.com/visualbruno/ComfyUI-Trellis2

License: MIT.

STM records the adaptation/attribution separately in `THIRD_PARTY_NOTICES.md`.

### Author's Trellis2MeshTexturingStandalone

This is the author's own source, not a third-party license obligation.

## Model audit

### TRELLIS.2-4B

Microsoft TRELLIS.2: MIT.

Use the official model release and keep its license information with any model redistribution.

### DINOv3 ViT-L/16

Meta DINOv3 License.

Important: the license expressly grants a worldwide, royalty-free limited license including commercial use, but also imposes conditions covering redistribution, trade controls, prohibited end uses, and other requirements.

This is the only recommended model in the standard STM path that prevents us from using the simpler phrase "unrestricted commercial license."

### FLUX.2 [klein] 4B

Black Forest Labs lists FLUX.2 [klein] 4B as Apache-2.0.

### Qwen3-4B

Qwen3-4B is Apache-2.0.

### FLUX.2 VAE

The FLUX.2 distribution uses Apache-2.0 for the 4B family components recommended here.

### Non-commercial FLUX models

The FLUX.2 9B / 9B-KV / dev families are not commercial-use recommendations and must not be substituted into a project advertised as using a commercial-safe 4B model.

## Direct runtime dependencies

`requirements.txt` contains:

- numpy
- scipy
- Pillow
- safetensors
- transformers
- tqdm
- psutil
- trimesh

These are installed as independent packages. Their own licenses apply. STM does not vendor them.

The fact that a dependency has a license obligation does **not** by itself make STM outputs non-commercial. The key question is whether a dependency imposes a non-commercial restriction on the use case. The direct packages above are used as ordinary runtime libraries, not as bundled model content.

## Output-rights disclaimer

A "commercially usable model" is not the same thing as an automatic copyright clearance for every generated asset.

The user is responsible for rights in:

- input meshes
- source textures
- reference photos
- logos/trademarks
- copyrighted characters
- third-party assets
- other content supplied to the pipeline

## Checked upstream references

- Microsoft TRELLIS.2: https://github.com/microsoft/TRELLIS.2
- Visual Bruno ComfyUI-Trellis2: https://github.com/visualbruno/ComfyUI-Trellis2
- FLUX.2 official repository: https://github.com/black-forest-labs/flux2
- FLUX.2 [klein] 4B model: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- DINOv3 ViT-L/16 license: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md

## Final publication recommendation

Use the wording:

> "STM is MIT-licensed software. The recommended model stack is commercially usable, subject to the individual model licenses. DINOv3 has additional end-use and redistribution conditions."

Do **not** use the wording:

> "Every use of every STM workflow is unrestricted commercially."

That statement would be broader than the DINOv3 license allows.
