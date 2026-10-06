import os

from .handle import StmModels

try:
    import folder_paths
except Exception:                                      # pragma: no cover - outside ComfyUI
    folder_paths = None


def models_dir() -> str:
    return folder_paths.models_dir if folder_paths is not None else os.path.join(os.getcwd(), "models")


class STMLoadModels:
    """Creates the shared model handle. No weights are read here: each stage loads (and releases) its own model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_folder": ("STRING", {"default": os.path.join("microsoft", "TRELLIS.2-4B"),
                                            "tooltip": "Folder below ComfyUI/models with texturing_pipeline.json and ckpts/ (microsoft/TRELLIS.2-4B, MIT)."}),
                "use_fp8": ("BOOLEAN", {"default": False, "tooltip": "Use the fp8 checkpoints (ckpts_fp8 / pipeline_fp8.json) if present: about half the VRAM."}),
                "keep_models_loaded": (["auto", "yes", "no"], {"default": "auto",
                                       "tooltip": "auto: park released models in RAM only on machines with >= 24 GiB RAM."}),
            },
            "optional": {
                "dino_folder": ("STRING", {"default": os.path.join("facebook", "dinov3-vitl16-pretrain-lvd1689m"),
                                           "tooltip": "DINOv3 ViT-L/16 folder below ComfyUI/models (gated: accept Meta's DINOv3 License on Hugging Face)."}),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load"
    CATEGORY = "STM"

    def load(self, model_folder, use_fp8, keep_models_loaded, dino_folder=None):
        root = models_dir()
        path = model_folder if os.path.isabs(model_folder) else os.path.join(root, model_folder)
        dino = dino_folder or os.path.join("facebook", "dinov3-vitl16-pretrain-lvd1689m")
        dino = dino if os.path.isabs(dino) else os.path.join(root, dino)
        keep = {"auto": None, "yes": True, "no": False}[keep_models_loaded]
        return (StmModels.from_folder(path, dino_path=dino, use_fp8=use_fp8, keep_models_loaded=keep),)


NODE_CLASS_MAPPINGS = {"Trellis2TexturingLoader": STMLoadModels}
NODE_DISPLAY_NAME_MAPPINGS = {"Trellis2TexturingLoader": "STM — Load Models"}
