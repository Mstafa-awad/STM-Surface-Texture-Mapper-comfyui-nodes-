FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'      # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
from .. import config as _sparse_config


def __getattr__(name):                              # TORCH_CONV_CHUNK_BYTES lives in the shared sparse config
    if name == 'TORCH_CONV_CHUNK_BYTES':
        return _sparse_config.TORCH_CONV_CHUNK_BYTES
    raise AttributeError(name)
