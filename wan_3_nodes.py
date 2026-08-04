"""ComfyUI nodes for the Wan 3.0 API through Muapi.

The pack mirrors the four Wan 3.0 model contracts currently published by
Muapi:

* text-to-image from a prompt;
* instruction-based image editing from a prompt and source image;
* text-to-video from a prompt; and
* image-to-video from a prompt and source image.

Each request uses Muapi's submit-then-poll API pattern and returns media in a
form that can be previewed or passed to one of the included saver nodes. The
implementation is intentionally self-contained so it can be installed into
``ComfyUI/custom_nodes`` without a separate Wan SDK.
"""

import io
import json
import mimetypes
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch
from PIL import Image


BASE_URL = os.getenv(
    "WAN_3_API_BASE_URL",
    os.getenv("MUAPI_API_BASE_URL", "https://api.muapi.ai/api/v1"),
)
POLL_INTERVAL = float(os.getenv("WAN_3_POLL_INTERVAL", "5"))
MAX_WAIT = float(os.getenv("WAN_3_MAX_WAIT", "1800"))

_MEDIA_KEYS = (
    "url",
    "image",
    "image_url",
    "video",
    "video_url",
    "file_url",
    "output",
    "outputs",
    "images",
    "data",
    "result",
    "media",
    "content",
    "file",
)


def _api_url(endpoint: str) -> str:
    return f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"


def _load_api_key(api_key_input: str = "") -> str:
    """Resolve a key from the node, environment, or Muapi CLI config."""
    if api_key_input and str(api_key_input).strip():
        return str(api_key_input).strip()

    for env_name in ("MUAPI_API_KEY", "WAN_3_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    config_path = os.path.expanduser("~/.muapi/config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            value = config.get("api_key") or config.get("MUAPI_API_KEY") or ""
            if str(value).strip():
                return str(value).strip()
        except (OSError, ValueError, TypeError):
            pass

    raise RuntimeError(
        "No Muapi key found. Paste one into the API Key node, set MUAPI_API_KEY, "
        "or run `muapi auth configure --api-key YOUR_KEY`."
    )


def _check_response(response: requests.Response) -> None:
    if response.status_code == 401:
        raise RuntimeError("Muapi authentication failed. Check the API key.")
    if response.status_code == 402:
        raise RuntimeError("Muapi rejected the request because the account lacks credits.")
    if response.status_code == 429:
        raise RuntimeError("Muapi rate limit reached. Retry the workflow later.")
    if response.ok:
        return

    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:500]
    raise RuntimeError(f"Muapi HTTP {response.status_code}: {detail}")


def _output_value(value: Any) -> Optional[str]:
    """Find the first HTTP media URL in a known Muapi response shape.

    The traversal intentionally ignores arbitrary dictionary keys such as
    ``urls.get``. Those keys can contain a prediction URL while the job is
    still processing, which must not be mistaken for generated media.
    """
    if isinstance(value, str):
        value = value.strip()
        if value.lower().startswith(("http://", "https://")):
            return value
        return None
    if isinstance(value, dict):
        for key in _MEDIA_KEYS:
            if key in value:
                found = _output_value(value[key])
                if found:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _output_value(item)
            if found:
                return found
    return None


def _has_media_output(data: Dict[str, Any]) -> bool:
    for key in ("image", "image_url", "video", "video_url", "outputs", "images"):
        if _output_value(data.get(key)):
            return True
    output = data.get("output")
    return isinstance(output, (str, list, tuple)) or (
        isinstance(output, dict)
        and any(key in output for key in ("image", "image_url", "video", "video_url", "outputs", "images"))
    )


def _submit(api_key: str, endpoint: str, payload: Dict[str, Any]) -> str:
    try:
        response = requests.post(
            _api_url(endpoint),
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not submit the Muapi request: {exc}") from exc

    _check_response(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Muapi returned a non-JSON submission response.") from exc

    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"Muapi submission did not include request_id: {data}")
    return str(request_id)


def _poll(api_key: str, request_id: str, label: str) -> Dict[str, Any]:
    deadline = time.monotonic() + MAX_WAIT
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                _api_url(f"predictions/{request_id}/result"),
                headers={"x-api-key": api_key},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not poll Muapi request {request_id}: {exc}") from exc

        _check_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Muapi returned a non-JSON polling response.") from exc

        status = str(data.get("status", "")).strip().lower()
        print(f"[{label}] {status or 'unknown'} — {request_id}")
        if status in {"completed", "complete", "succeeded", "success", "done"}:
            return data
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            error = data.get("error") or data.get("message") or data
            raise RuntimeError(f"{label} generation failed: {error}")
        if not status and _has_media_output(data):
            return data
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Timed out waiting for {label} request {request_id}.")


def _output_url(result: Dict[str, Any], media_kind: str) -> str:
    preferred = (
        ("image", "image_url", "images")
        if media_kind == "image"
        else ("video", "video_url", "outputs")
    )
    for key in preferred + ("output", "url", "file_url", "outputs", "images"):
        found = _output_value(result.get(key))
        if found:
            return found
    raise RuntimeError(f"Muapi result did not contain an output {media_kind} URL: {result}")


def _pil_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _download_image(image_url: str):
    try:
        response = requests.get(image_url, timeout=180)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            return _pil_to_tensor(image)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download the generated image: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Muapi returned an unreadable image: {exc}") from exc


def _first_frame(video_url: str):
    """Download a generated video and return its first frame as IMAGE."""
    temp_path = None
    try:
        import cv2

        response = requests.get(video_url, timeout=180, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            temp_path = handle.name
            for chunk in response.iter_content(1024 * 32):
                if chunk:
                    handle.write(chunk)

        capture = cv2.VideoCapture(temp_path)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("the downloaded video had no readable first frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0)
    except Exception as exc:
        print(f"[Wan 3.0] Could not decode first frame: {exc}")
        return torch.zeros(1, 64, 64, 3)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _image_bytes(image_tensor: Any) -> io.BytesIO:
    if image_tensor is None:
        raise ValueError("An image input is required.")
    tensor = image_tensor.detach() if hasattr(image_tensor, "detach") else image_tensor
    if getattr(tensor, "dim", lambda: 0)() == 4:
        tensor = tensor[0]
    array = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError("Expected a ComfyUI IMAGE tensor with shape [H, W, C].")
    array = array.astype(np.float32)
    if array.size and float(np.max(array)) > 1.0:
        array /= 255.0
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] < 3:
        raise ValueError("The image input must have at least one color channel.")
    array = np.clip(array[..., :3], 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB").save(
        buffer, format="JPEG", quality=95
    )
    buffer.seek(0)
    return buffer


def _upload_file(api_key: str, filename: str, content: Any, mime: str) -> str:
    try:
        response = requests.post(
            _api_url("upload_file"),
            headers={"x-api-key": api_key},
            files={"file": (filename, content, mime)},
            timeout=600,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not upload the image to Muapi: {exc}") from exc

    _check_response(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Muapi returned a non-JSON upload response.") from exc
    url = _output_value(data)
    if not url:
        raise RuntimeError(f"Muapi image upload did not return a URL: {data}")
    return url


def _upload_image(api_key: str, image_tensor: Any) -> str:
    return _upload_file(api_key, "image.jpg", _image_bytes(image_tensor), "image/jpeg")


def _input_directory() -> Optional[str]:
    try:
        import folder_paths

        return folder_paths.get_input_directory()
    except Exception:
        return None


def _resolve_image_ref(api_key: str, reference: Any) -> Optional[str]:
    """Resolve a URL or local ComfyUI/input path to a hosted image URL."""
    if not reference or not str(reference).strip():
        return None
    reference = str(reference).strip().strip('"').strip("'")
    if reference.lower().startswith(("http://", "https://")):
        return reference

    path = pathlib.Path(os.path.expanduser(reference))
    if not path.is_file():
        input_dir = _input_directory()
        if input_dir:
            candidate = pathlib.Path(input_dir) / reference
            if candidate.is_file():
                path = candidate
    if not path.is_file():
        raise RuntimeError(
            f"Image reference not found: {reference!r}. Use an http(s) URL, an absolute "
            "path, or a file in ComfyUI/input."
        )

    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    try:
        with path.open("rb") as handle:
            return _upload_file(api_key, path.name, handle, mime)
    except OSError as exc:
        raise RuntimeError(f"Could not read image reference {path}: {exc}") from exc


def _prompt_input(default: str):
    return (
        "STRING",
        {
            "multiline": True,
            "default": default,
        },
    )


def _image_source_inputs():
    return {
        "api_key": ("STRING", {"multiline": False, "default": ""}),
        "image": ("IMAGE",),
        "image_url": (
            "STRING",
            {
                "multiline": False,
                "default": "",
                "tooltip": "Remote URL or local path; used when IMAGE is not connected.",
            },
        ),
    }


class _Wan30GenerationNode:
    endpoint = ""
    label = "Wan 3.0"
    media_kind = "image"

    def _complete(self, api_key: str, payload: Dict[str, Any]):
        print(f"[{self.label}] Submitting to {self.endpoint}...")
        request_id = _submit(api_key, self.endpoint, payload)
        result = _poll(api_key, request_id, self.label)
        media_url = _output_url(result, self.media_kind)
        print(f"[{self.label}] Done — request_id={request_id}")
        if self.media_kind == "image":
            return _download_image(media_url), media_url, request_id
        return media_url, _first_frame(media_url), request_id


class Wan30ApiKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Muapi key. Get one at muapi.ai → Dashboard → API Keys.",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"

    def run(self, api_key):
        return (_load_api_key(api_key),)


class Wan30TextToImage(_Wan30GenerationNode):
    endpoint = "wan3.0-text-to-image"
    label = "Wan 3.0 T2I"
    media_kind = "image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": _prompt_input(
                    "An editorial product photograph with soft studio lighting and precise detail"
                )
            },
            "optional": {"api_key": ("STRING", {"multiline": False, "default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"

    def run(self, prompt, api_key=""):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty.")
        return self._complete(_load_api_key(api_key), {"prompt": prompt})


class _Wan30ImageSourceNode(_Wan30GenerationNode):
    media_kind = "image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt": _prompt_input("Preserve the subject and make a subtle, polished change")},
            "optional": _image_source_inputs(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"

    def run(self, prompt, api_key="", image=None, image_url=""):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty.")
        key = _load_api_key(api_key)
        source_url = _upload_image(key, image) if image is not None else _resolve_image_ref(key, image_url)
        if not source_url:
            raise ValueError("Connect an IMAGE or provide image_url to the Wan 3.0 Image Edit node.")
        return self._complete(
            key,
            {"prompt": prompt, "image_url": source_url},
        )


class Wan30ImageEdit(_Wan30ImageSourceNode):
    endpoint = "wan3.0-image-edit"
    label = "Wan 3.0 Image Edit"


class Wan30TextToVideo(_Wan30GenerationNode):
    endpoint = "wan3.0-text-to-video"
    label = "Wan 3.0 T2V"
    media_kind = "video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": _prompt_input(
                    "A cinematic aerial shot of mist moving through a mountain valley at sunrise"
                )
            },
            "optional": {"api_key": ("STRING", {"multiline": False, "default": ""})},
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"

    def run(self, prompt, api_key=""):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty.")
        return self._complete(_load_api_key(api_key), {"prompt": prompt})


class Wan30ImageToVideo(_Wan30GenerationNode):
    endpoint = "wan3.0-image-to-video"
    label = "Wan 3.0 I2V"
    media_kind = "video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": _prompt_input(
                    "The camera slowly pushes in as a breeze moves through the subject's hair"
                )
            },
            "optional": _image_source_inputs(),
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"

    def run(self, prompt, api_key="", image=None, image_url=""):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty.")
        key = _load_api_key(api_key)
        source_url = _upload_image(key, image) if image is not None else _resolve_image_ref(key, image_url)
        if not source_url:
            raise ValueError("Connect an IMAGE or provide image_url to the Wan 3.0 Image-to-Video node.")
        return self._complete(
            key,
            {"prompt": prompt, "image_url": source_url},
        )


NODE_CLASS_MAPPINGS = {
    "Wan30ApiKey": Wan30ApiKey,
    "Wan30TextToImage": Wan30TextToImage,
    "Wan30ImageEdit": Wan30ImageEdit,
    "Wan30TextToVideo": Wan30TextToVideo,
    "Wan30ImageToVideo": Wan30ImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan30ApiKey": "🔑 Wan 3.0 API Key",
    "Wan30TextToImage": "🖼️ Wan 3.0 Text-to-Image",
    "Wan30ImageEdit": "🖌️ Wan 3.0 Image Edit",
    "Wan30TextToVideo": "🎬 Wan 3.0 Text-to-Video",
    "Wan30ImageToVideo": "🎬 Wan 3.0 Image-to-Video",
}
