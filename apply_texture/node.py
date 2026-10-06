from .stage import apply_texture


class STMApplyTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH",), "base_color": ("IMAGE",)},
                "optional": {"metallic": ("IMAGE",), "roughness": ("IMAGE",)}}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, mesh, base_color, metallic=None, roughness=None):
        return (apply_texture(mesh, base_color, metallic, roughness),)


NODE_CLASS_MAPPINGS = {"STM_ApplyTexture": STMApplyTexture}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_ApplyTexture": "STM — Apply Texture"}
