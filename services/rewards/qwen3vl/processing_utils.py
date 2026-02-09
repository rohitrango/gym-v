from __future__ import annotations

import logging

# Match slime's default behavior for models without patch_size.
DEFAULT_PATCH_SIZE = 14


def process_vision_info(prompt, processor):
    if processor is None:
        return {"images": [], "videos": []}

    from qwen_vl_utils import process_vision_info as _process_vision_info

    image_patch_size = None
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_patch_size = getattr(image_processor, "patch_size", None)
    if image_patch_size is None:
        logging.getLogger(__name__).info(
            "Using default patch size: %s", DEFAULT_PATCH_SIZE
        )
        image_patch_size = DEFAULT_PATCH_SIZE

    images, videos = _process_vision_info(prompt, image_patch_size=image_patch_size)
    return {"images": images, "videos": videos}
