"""ComfyUI output nodes for downloading Wan 3.0 media URLs."""

import io
import os

import numpy as np
import requests
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _output_directory():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.path.expanduser("~"), "comfyui_output")


def _safe_subfolder(value, fallback="wan_3_0"):
    value = (value or fallback).strip()
    normalized = os.path.normpath(value)
    if normalized in ("", "."):
        return fallback
    if os.path.isabs(normalized) or normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("save_subfolder must stay inside ComfyUI's output directory.")
    return normalized


def _safe_prefix(value, fallback="wan_3_0"):
    prefix = os.path.basename((value or fallback).strip())
    return prefix or fallback


def _next_path(output_dir, prefix, extension):
    file_number = 1
    filepath = os.path.join(output_dir, f"{prefix}_{file_number:05d}{extension}")
    while os.path.exists(filepath):
        file_number += 1
        filepath = os.path.join(output_dir, f"{prefix}_{file_number:05d}{extension}")
    return filepath


class Wan30ImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "save_subfolder": ("STRING", {"default": "wan_3_0"}),
                "filename_prefix": ("STRING", {"default": "wan_3_0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filepath")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"
    OUTPUT_NODE = True

    def run(self, image_url, save_subfolder, filename_prefix):
        if not image_url or not image_url.strip().lower().startswith(("http://", "https://")):
            return self._error("image_url must be an http(s) URL.")

        filepath = None
        try:
            save_subfolder = _safe_subfolder(save_subfolder)
            filename_prefix = _safe_prefix(filename_prefix)
            output_dir = os.path.join(_output_directory(), save_subfolder)
            os.makedirs(output_dir, exist_ok=True)
            filepath = _next_path(output_dir, filename_prefix, ".png")

            print(f"[Wan 3.0 Saver] Downloading {image_url[:100]}...")
            response = requests.get(image_url.strip(), timeout=600)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image = image.convert("RGB")
                image.save(filepath, format="PNG")
                tensor = self._to_tensor(image)

            filename = os.path.basename(filepath)
            preview = {"filename": filename, "subfolder": save_subfolder, "type": "output"}
            print(f"[Wan 3.0 Saver] Saved {filename}")
            return {"ui": {"images": [preview]}, "result": (tensor, filepath)}
        except Exception as exc:
            if filepath and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            return self._error(str(exc))

    @staticmethod
    def _to_tensor(image):
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array.copy()).unsqueeze(0)

    @staticmethod
    def _error(message):
        print(f"[Wan 3.0 Saver] ERROR: {message}")
        return {
            "ui": {"text": [message]},
            "result": (torch.zeros(1, 64, 64, 3), "ERROR"),
        }


class Wan30VideoSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"multiline": False, "default": ""}),
                "save_subfolder": ("STRING", {"default": "wan_3_0"}),
                "filename_prefix": ("STRING", {"default": "wan_3_0"}),
            },
            "optional": {
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 30}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("frames", "filepath", "frame_count")
    FUNCTION = "run"
    CATEGORY = "🌀 Wan 3.0"
    OUTPUT_NODE = True

    def run(
        self,
        video_url,
        save_subfolder,
        filename_prefix,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
    ):
        if not video_url or not video_url.strip().lower().startswith(("http://", "https://")):
            return self._error("video_url must be an http(s) URL.")

        filepath = None
        try:
            save_subfolder = _safe_subfolder(save_subfolder)
            filename_prefix = _safe_prefix(filename_prefix)
            output_dir = os.path.join(_output_directory(), save_subfolder)
            os.makedirs(output_dir, exist_ok=True)
            filepath = _next_path(output_dir, filename_prefix, ".mp4")

            print(f"[Wan 3.0 Saver] Downloading {video_url[:100]}...")
            response = requests.get(video_url.strip(), stream=True, timeout=600)
            response.raise_for_status()
            with open(filepath, "wb") as handle:
                for chunk in response.iter_content(1024 * 32):
                    if chunk:
                        handle.write(chunk)

            frames, count = self._load_frames(
                filepath,
                int(frame_load_cap),
                int(skip_first_frames),
                int(select_every_nth),
            )
            filename = os.path.basename(filepath)
            preview = {
                "filename": filename,
                "subfolder": save_subfolder,
                "type": "output",
                "format": "video/mp4",
            }
            print(f"[Wan 3.0 Saver] Saved {filename} — {count} frame(s)")
            return {"ui": {"gifs": [preview]}, "result": (frames, filepath, count)}
        except Exception as exc:
            if filepath and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            return self._error(str(exc))

    @staticmethod
    def _load_frames(path, cap, skip, every):
        try:
            import cv2

            frames = []
            raw_index = 0
            capture = cv2.VideoCapture(path)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if raw_index < skip:
                    raw_index += 1
                    continue
                if (raw_index - skip) % every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    frames.append(rgb)
                    if cap > 0 and len(frames) >= cap:
                        break
                raw_index += 1
            capture.release()
            if not frames:
                raise RuntimeError("No readable frames were found in the downloaded video.")
            return torch.from_numpy(np.stack(frames)), len(frames)
        except Exception as exc:
            print(f"[Wan 3.0 Saver] Frame decode failed: {exc}")
            return torch.zeros(1, 64, 64, 3), 1

    @staticmethod
    def _error(message):
        print(f"[Wan 3.0 Saver] ERROR: {message}")
        return {
            "ui": {"text": [message]},
            "result": (torch.zeros(1, 64, 64, 3), "ERROR", 0),
        }


NODE_CLASS_MAPPINGS = {
    "Wan30ImageSaver": Wan30ImageSaver,
    "Wan30VideoSaver": Wan30VideoSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan30ImageSaver": "🖼️ Wan 3.0 Save Image",
    "Wan30VideoSaver": "🎬 Wan 3.0 Save Video",
}
