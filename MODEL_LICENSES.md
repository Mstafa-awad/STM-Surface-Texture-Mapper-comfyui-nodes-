# Model Licenses and Commercial-Use Matrix

This file is a practical model checklist for STM. It does not replace the upstream model licenses.

## Recommended models

| Model | Recommended source | License | Commercial use |
|---|---|---|---|
| Microsoft TRELLIS.2-4B | https://github.com/microsoft/TRELLIS.2 | MIT | Yes, subject to MIT terms |
| Meta DINOv3 ViT-L/16 | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 License | Yes, but subject to additional restrictions/conditions |
| FLUX.2 [klein] 4B | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B | Apache-2.0 | Yes |
| Qwen3-4B | https://huggingface.co/Qwen/Qwen3-4B | Apache-2.0 | Yes |
| FLUX.2 VAE | distributed by Black Forest Labs with FLUX.2 | Apache-2.0 | Yes |

## Important DINOv3 limitation

Meta's DINOv3 license provides a worldwide, royalty-free limited license, including commercial use, but it is not an unrestricted MIT/Apache-style license.

The license includes conditions relating to:

- compliance with applicable law and trade controls
- prohibited ITAR / military / warfare / nuclear / espionage / weapons-related end uses
- restrictions around reverse engineering
- redistribution of DINO materials and derivatives
- acceptance of the agreement

Read the exact license before using the DINOv3 path for a regulated or restricted project.

## Models to avoid for commercial work

Do **not** use these as the commercial recommendation:

| Model family | License status |
|---|---|
| FLUX.2 [klein] 9B | FLUX Non-Commercial License |
| FLUX.2 [klein] 9B KV | FLUX Non-Commercial License |
| FLUX.2 [dev] | FLUX Non-Commercial License |

For the official FLUX.2 model matrix, see:

https://github.com/black-forest-labs/flux2

## Outputs

A model license controls the permissions and obligations attached to your use of that model. It does not automatically give you ownership or permission for the input material you feed into it.

You should separately have the required rights for:

- source meshes
- input/reference images
- logos and trademarks
- characters and branded material
- textures or photographs supplied by third parties

