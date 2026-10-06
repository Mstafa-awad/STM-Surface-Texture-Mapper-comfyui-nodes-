"""Shape encoder of TRELLIS.2 (flexible dual grid -> structured latent).

Derived from microsoft/TRELLIS.2 ``trellis2/models/sc_vaes/fdg_vae.py`` (MIT, (c) Microsoft Corporation).
Only the *encoder* is kept: it is all the Mesh Texturing node needs.  The decoder (mesh extraction)
imported ``o_voxel``, whose compiled extension contains code derived from a non-commercial
licence, and was removed together with that dependency.
"""
from typing import *
import torch
import torch.nn as nn
from ...modules import sparse as sp
from .sparse_unet_vae import SparseUnetVaeEncoder


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        use_fp8: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
            use_fp8,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
