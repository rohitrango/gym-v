from __future__ import annotations

import base64
from io import BytesIO
import os
from typing import Any
import urllib.parse
import urllib.request

from PIL import Image

from gym_v.logger import get_logger

logger = get_logger()


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    return base64.b64decode(value)


def _looks_like_base64(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 16 or len(stripped) % 4 != 0:
        return False
    for ch in stripped:
        if ch.isalnum() or ch in "+/=":
            continue
        return False
    return True


def _is_image_header(data: bytes) -> bool:
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return True
    if data.startswith(b"RIFF") and b"WEBP" in data[8:16]:
        return True
    return False


def _maybe_decode_base64(value: str) -> bytes | None:
    if value.startswith("data:"):
        return _decode_base64(value)
    if not _looks_like_base64(value):
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception:
        return None
    return decoded if _is_image_header(decoded) else None


def _fetch_url(url: str, *, timeout_s: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read()


def image_to_base64(
    image: Image.Image,
    *,
    image_format: str = "PNG",
    data_url: bool = True,
) -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if data_url:
        return f"data:image/{image_format.lower()};base64,{encoded}"
    return encoded


def video_to_base64(
    video: bytes | bytearray | memoryview | str | os.PathLike[str],
    *,
    data_url: bool = False,
    mime_type: str = "video/mp4",
) -> str:
    if isinstance(video, (bytes, bytearray, memoryview)):
        payload = bytes(video)
    else:
        with open(os.fspath(video), "rb") as f:
            payload = f.read()
    encoded = base64.b64encode(payload).decode("utf-8")
    if data_url:
        return f"data:{mime_type};base64,{encoded}"
    return encoded


def convert_local_image_or_video_to_base_64(
    value: Any,
    *,
    media_type: str | None = None,
    image_format: str = "PNG",
    data_url: bool = True,
    mime_type: str = "video/mp4",
) -> Any:
    if isinstance(value, (list, tuple)):
        return [
            convert_local_image_or_video_to_base_64(
                item,
                media_type=media_type,
                image_format=image_format,
                data_url=data_url,
                mime_type=mime_type,
            )
            for item in value
        ]
    if media_type is None:
        media_type = "image" if isinstance(value, Image.Image) else "video"
    if media_type == "image":
        if isinstance(value, Image.Image):
            return image_to_base64(value, image_format=image_format, data_url=data_url)
        if isinstance(value, (bytes, bytearray, memoryview)):
            image = Image.open(BytesIO(bytes(value)))
            return image_to_base64(image, image_format=image_format, data_url=data_url)
        image = Image.open(os.fspath(value))
        return image_to_base64(image, image_format=image_format, data_url=data_url)
    if media_type == "video":
        return video_to_base64(value, data_url=data_url, mime_type=mime_type)
    raise ValueError(f"Unsupported media_type: {media_type!r}")


_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def _looks_like_image_path(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("data:image/"):
        return True
    if lowered.startswith(("http://", "https://")):
        path = urllib.parse.urlparse(lowered).path
    else:
        path = lowered
    _, ext = os.path.splitext(path)
    return ext in _IMAGE_EXTENSIONS


def _infer_media_type(value: Any) -> str:
    if isinstance(value, Image.Image):
        return "image"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "video"
    if isinstance(value, str):
        if _looks_like_image_path(value):
            return "image"
        if _maybe_decode_base64(value) is not None:
            return "image"
        return "video"
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            if _infer_media_type(item) == "image":
                return "image"
        return "video"
    return "video"


def convert_base64_to_local_image_or_video(
    value: Any,
    *,
    media_type: str | None = None,
) -> Any:
    if media_type is None:
        media_type = _infer_media_type(value)
    if media_type == "image":
        if value is None:
            return value
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, Image.Image) for item in value):
                return list(value)
            return _load_images_from_urls([str(item) for item in value])
        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(value))).convert("RGB")
        if isinstance(value, str):
            images = _load_images_from_urls([value])
            return images[0] if len(images) == 1 else images
        return value
    if media_type == "video":
        if value is None:
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, (list, tuple)):
            return _load_videos_from_urls(list(value))
        if isinstance(value, (str, os.PathLike)):
            videos = _load_videos_from_urls([value])
            return videos[0] if len(videos) == 1 else videos
        return value
    raise ValueError(f"Unsupported media_type: {media_type!r}")


def _load_images_from_urls(image_urls: list[str]) -> list[Image.Image]:
    """Load images from data URLs, raw base64 strings, HTTP(S) URLs, or local paths."""
    if not image_urls:
        return []

    images: list[Image.Image] = []
    for idx, url in enumerate(image_urls):
        try:
            if url.startswith("data:image/"):
                log_desc = f"data URI (length: {len(url)})"
                logger.info("Loading image %d from %s", idx + 1, log_desc)
                _, _, base64_data = url.partition(",")
                img_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            elif url.startswith(("http://", "https://")):
                log_desc = f"URL: {url[:80]}{'...' if len(url) > 80 else ''}"
                logger.info("Downloading image %d from %s", idx + 1, log_desc)
                img_data = _fetch_url(url)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            else:
                decoded = _maybe_decode_base64(url)
                if decoded is not None:
                    log_desc = f"raw base64 string (length: {len(url)})"
                    logger.info("Loading image %d from %s", idx + 1, log_desc)
                    image = Image.open(BytesIO(decoded)).convert("RGB")
                else:
                    path = os.path.expanduser(url)
                    log_desc = f"local path: {path}"
                    logger.info("Loading image %d from %s", idx + 1, log_desc)
                    image = Image.open(path).convert("RGB")
            images.append(image)
            logger.info("Image %d loaded successfully: %s", idx + 1, image.size)
        except Exception as exc:
            error_url = f"<data of length {len(url)}" if len(url) > 100 else url
            logger.error("Failed to load image %d from %s: %s", idx + 1, error_url, exc)
            raise RuntimeError(f"Failed to load image {idx + 1}: {exc}") from exc

    return images


def _load_videos_from_urls(video_urls: list[Any]) -> list[Any]:
    """Load videos from bytes-like, data URLs, raw base64 strings, HTTP(S) URLs, or local paths."""
    if not video_urls:
        return []

    videos: list[Any] = []
    for idx, item in enumerate(video_urls):
        url: str | None = None
        try:
            if item is None:
                videos.append(None)
                continue
            if isinstance(item, (bytes, bytearray, memoryview)):
                payload = bytes(item)
                logger.info(
                    "Loading video %d from in-memory bytes (%d bytes)",
                    idx + 1,
                    len(payload),
                )
                videos.append(payload)
                continue
            if isinstance(item, os.PathLike):
                item = os.fspath(item)
                if isinstance(item, bytes):
                    item = os.fsdecode(item)
            if not isinstance(item, str):
                raise TypeError(f"Unsupported video item type: {type(item)!r}")

            url = item
            if url.startswith("data:"):
                log_desc = f"data URI (length: {len(url)})"
                logger.info("Loading video %d from %s", idx + 1, log_desc)
                payload = _decode_base64(url)
            elif url.startswith(("http://", "https://")):
                log_desc = f"URL: {url[:80]}{'...' if len(url) > 80 else ''}"
                logger.info("Downloading video %d from %s", idx + 1, log_desc)
                payload = _fetch_url(url)
            else:
                payload = None
                if _looks_like_base64(url):
                    try:
                        payload = base64.b64decode(url, validate=True)
                        log_desc = f"raw base64 string (length: {len(url)})"
                        logger.info("Loading video %d from %s", idx + 1, log_desc)
                    except Exception:
                        payload = None
                if payload is None:
                    path = os.path.expanduser(url)
                    log_desc = f"local path: {path}"
                    logger.info("Loading video %d from %s", idx + 1, log_desc)
                    with open(path, "rb") as f:
                        payload = f.read()
            videos.append(payload)
            logger.info(
                "Video %d loaded successfully: %d bytes", idx + 1, len(payload)
            )
        except Exception as exc:
            if url is None:
                error_url = f"<{type(item).__name__}>"
            else:
                error_url = f"<data of length {len(url)}" if len(url) > 100 else url
            logger.error("Failed to load video %d from %s: %s", idx + 1, error_url, exc)
            raise RuntimeError(f"Failed to load video {idx + 1}: {exc}") from exc

    return videos
