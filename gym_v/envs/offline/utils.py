import base64
import io
import json
import re

from PIL import Image

from gym_v.logger import get_logger

logger = get_logger()


try:
    import orjson

    def fast_json_loads(s: str) -> dict:
        return orjson.loads(s)
except ImportError:
    logger.warning("orjson not installed, falling back to native json")

    def fast_json_loads(s: str) -> dict:
        return json.loads(s)


def normalize_em(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def decode_base64_image(image_data: str) -> Image.Image:
    img_str = image_data.split(";base64,")[-1]
    image_bytes = base64.b64decode(img_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
