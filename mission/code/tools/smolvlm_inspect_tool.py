"""Smolagents tool that inspects each SO-101 camera feed with SmolVLM."""

from __future__ import annotations

import json
import logging
import re
import time
import gc  # <--- Added for memory cleanup
from typing import Any

import cv2
import torch
from PIL import Image
from smolagents import Tool

from inference.smolvlm_segmentation import (
    DEFAULT_MODEL_ID,
    SmolVLMSegmentationConfig,
    load_model,
)
from tools.smolvlm_segmentation_tool import DEFAULT_CAMERA_LAYOUT

logger = logging.getLogger(__name__)
_GRID_TOKEN_PATTERN = re.compile(r"<row_\d+_col_\d+>", re.IGNORECASE)

DEFAULT_PROMPT = "Describe what you see in this camera feed."
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_FRAME_WIDTH = DEFAULT_CAMERA_LAYOUT["top"]["width"]
DEFAULT_FRAME_HEIGHT = DEFAULT_CAMERA_LAYOUT["top"]["height"]


class SmolVLMInspectTool(Tool):
    """Capture all SO-101 cameras and ask SmolVLM to describe each view."""

    name = "capture_scene_inspection"
    output_type = "string"
    description = (
        "Capture the top/side/front SO-101 camera feeds, describe the scene in each using SmolVLM "
        "with a short prompt, and return the combined context."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Instruction for SmolVLM when describing the camera view.",
            "default": DEFAULT_PROMPT,
            "nullable": True,
        },
        "delay_seconds": {
            "type": "number",
            "description": "Seconds to wait before capturing frames.",
            "default": 3.0,
            "nullable": True,
        },
        "max_new_tokens": {
            "type": "number",
            "description": "Maximum tokens SmolVLM should emit per view.",
            "default": DEFAULT_MAX_NEW_TOKENS,
            "nullable": True,
        },
        "model_id": {
            "type": "string",
            "description": "SmolVLM checkpoint to load.",
            "default": DEFAULT_MODEL_ID,
            "nullable": True,
        },
        "device": {
            "type": "string",
            "description": "Torch device override (e.g. cuda, cpu).",
            "default": "",
            "nullable": True,
        },
        "torch_dtype": {
            "type": "string",
            "description": "Torch dtype override passed to the model loader.",
            "default": "auto",
            "nullable": True,
        },
        "frame_width": {
            "type": "number",
            "description": "Requested width for each captured frame.",
            "default": DEFAULT_FRAME_WIDTH,
            "nullable": True,
        },
        "frame_height": {
            "type": "number",
            "description": "Requested height for each captured frame.",
            "default": DEFAULT_FRAME_HEIGHT,
            "nullable": True,
        },
    }
    outputs = {
        "context": {
            "type": "string",
            "description": "Concatenated descriptions for each camera for agent context.",
        },
        "descriptions_json": {
            "type": "string",
            "description": "Structured JSON with each camera description.",
        },
        "descriptions": {
            "type": "object",
            "description": "Per-camera description dictionary as parsed directly from SmolVLM.",
        },
    }
    def __init__(self, agent_model=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_model = agent_model

    def forward(
        self,
        prompt: str | None = None,
        delay_seconds: float | None = None,
        max_new_tokens: int | None = None,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> dict[str, Any]:
        prompt_text = (prompt or DEFAULT_PROMPT).strip()
        if not prompt_text:
            prompt_text = DEFAULT_PROMPT
        if len(prompt_text) > 200:
            logger.debug("Truncating oversized prompt for SmolVLM inspection")
            prompt_text = DEFAULT_PROMPT
        wait_seconds = max(0.0, float(delay_seconds or 0.0))
        width = int(frame_width or DEFAULT_FRAME_WIDTH)
        height = int(frame_height or DEFAULT_FRAME_HEIGHT)
        
        # Initialize variables for cleanup scope
        processor = None
        model = None

        if self.agent_model and hasattr(self.agent_model, 'offload'):
            self.agent_model.offload()

        try:
            logger.info("⏳ [SmolVLM] Loading model to GPU...")
            cfg = SmolVLMSegmentationConfig(
                prompt=prompt_text,
                model_id=model_id or DEFAULT_MODEL_ID,
                device=device or None,
                torch_dtype=torch_dtype or "auto",
            )
            processor, model, resolved_device = load_model(cfg)

            camera_map = self._build_camera_map()
            if wait_seconds:
                logger.info("Waiting %.2fs before capturing inspection frames", wait_seconds)
                time.sleep(wait_seconds)

            descriptions: dict[str, dict[str, str]] = {}
            tokens = int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS)
            
            for name, index in camera_map.items():
                logger.info("Capturing %s camera (index %s) for inspection", name, index)
                frame = self._capture_frame(index, width, height)
                
                # Run Inference
                description = _describe_view(
                    processor=processor,
                    model=model,
                    device=resolved_device,
                    image=frame,
                    prompt=prompt_text,
                    max_new_tokens=tokens,
                )
                descriptions[name] = {"description": description}

            context = _format_context(descriptions)
            return {
                "context": context,
                "descriptions_json": json.dumps(descriptions, indent=2),
                "descriptions": descriptions,
            }

        except Exception as e:
            logger.error(f"Error during SmolVLM inspection: {e}")
            raise e

        finally:
            # --- AGGRESSIVE VRAM CLEANUP ---
            logger.info("🧹 [SmolVLM] Unloading model and clearing VRAM...")
            
            if model is not None:
                # Move weights to CPU first
                model.to("cpu")
            
            # Delete references to allow Python GC to pick them up
            del model
            del processor
            
            # Force Garbage Collection
            gc.collect()
            
            # Clear CUDA Cache (The most important part for VRAM)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ [SmolVLM] Cleanup complete. VRAM allocated: {free_mem:.2f} GB")

            if self.agent_model and hasattr(self.agent_model, 'reload'):
                self.agent_model.reload()

    @staticmethod
    def _build_camera_map() -> dict[str, int]:
        return {
            "top": DEFAULT_CAMERA_LAYOUT["top"]["index"],
            "side": DEFAULT_CAMERA_LAYOUT["side"]["index"],
            "front": DEFAULT_CAMERA_LAYOUT["front"]["index"],
        }

    @staticmethod
    def _capture_frame(index: int, width: int, height: int, warmup_frames: int = 5) -> Image.Image:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera at index {index} could not be opened")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(max(0, warmup_frames)):
            cap.read()

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from camera index {index}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)


def _describe_view(
    processor,
    model,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(message, add_generation_prompt=True)
    inputs = processor(text=formatted_prompt, images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return _clean_model_response(outputs[0])


def _clean_model_response(raw_text: str) -> str:
    text = raw_text.strip()
    lowered = text.lower()
    assistant_marker = "assistant:"
    marker_idx = lowered.rfind(assistant_marker)
    if marker_idx != -1:
        text = text[marker_idx + len(assistant_marker) :]
    text = _GRID_TOKEN_PATTERN.sub("", text)
    return text.strip()


def _format_context(descriptions: dict[str, dict[str, str]]) -> str:
    lines = ["TOON", "scene_descriptions[<=3]{view,description}:"]
    for view, info in descriptions.items():
        description = info.get("description", "").replace("\n", " ")
        sanitized = " ".join(description.strip().split())
        lines.append(f"  {view},{sanitized}")
    lines.append("END TOON")
    return "\n".join(lines)
