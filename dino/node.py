from .stage import extract_dino_features, images_from_comfy, preview_images


class STMDINOv3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "image_size": ([512, 1024, 1536, 2048, 4096], {"default": 1024, "tooltip": "Size the image is resized to before DINOv3. TRELLIS.2 was trained with 512 and 1024; 1536/2048/4096 are experimental: 2.25x / 4x / 16x more image tokens (slower, more VRAM) and quality is not validated."}),
                "max_views": ("INT", {"default": 4, "min": 1, "max": 16}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Foreground mask (white = object): crops around the object and sets the background to black (official TRELLIS.2 preprocessing)."}),
                "preprocess_image": ("BOOLEAN", {"default": False, "tooltip": "Apply that preprocessing using the image's alpha channel. Automatic when a mask is connected."}),
                "crop_to_main_object": ("BOOLEAN", {"default": False, "tooltip": "Crop around the largest masked region only (ignores particles / stray mask blobs that would otherwise shrink the object in the crop)."}),
            },
        }

    RETURN_TYPES = ("STM_DINO_FEATURES", "IMAGE")
    RETURN_NAMES = ("dino_features", "processed_image")
    FUNCTION = "run"
    CATEGORY = "STM"

    def run(self, models, image, image_size, max_views, mask=None, preprocess_image=False, crop_to_main_object=False):
        from ..load_models.handle import coerce_models
        from ..runtime.stage import log
        views = images_from_comfy(image, mask, bool(preprocess_image), int(max_views), bool(crop_to_main_object))
        for i, v in enumerate(views):
            side = v.size[0]
            note = ""
            if int(image_size) > 1.25 * side:
                note = (f" -> WARNING: upsampled {int(image_size) / side:.1f}x: no extra detail, only more tokens. "
                        f"Use image_size {512 if side < 768 else 1024} unless the object itself is larger in the source image.")
            log(f"DINOv3 view {i + 1}: object crop {side}x{side}px -> {image_size}px{note}")
        features = extract_dino_features(coerce_models(models), views, int(image_size))
        return (features, preview_images(views, int(image_size)))


NODE_CLASS_MAPPINGS = {"STM_DINOv3": STMDINOv3}
NODE_DISPLAY_NAME_MAPPINGS = {"STM_DINOv3": "STM — DINOv3"}
