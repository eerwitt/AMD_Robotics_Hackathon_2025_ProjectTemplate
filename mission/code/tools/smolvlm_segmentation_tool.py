"""Smolagents tool that snapshots SO-101 cameras and summarizes the scene in TOON."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from smolagents import Tool

from inference.smolvlm_segmentation import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    SmolVLMSegmentationConfig,
    load_model,
    run as run_segmentation,
)

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_LAYOUT = {
    "top": {"index": 6, "width": 640, "height": 480},
    "side": {"index": 4, "width": 640, "height": 480},
    "front": {"index": 2, "width": 640, "height": 480},
}
SEGMENTATION_DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
SEGMENTATION_DEFAULT_DEVICE: str | None = None
SEGMENTATION_DEFAULT_TORCH_DTYPE = "auto"
SEGMENTATION_DEFAULT_FRAME_WIDTH = DEFAULT_CAMERA_LAYOUT["top"]["width"]
SEGMENTATION_DEFAULT_FRAME_HEIGHT = DEFAULT_CAMERA_LAYOUT["top"]["height"]
OUTPUT_DIR = "outputs/smolvlm/cameras"


class SmolVLMSegmentationTool(Tool):
    """Capture frames from the default cameras and return a TOON scene summary. Use this tool to understand what is physically around the environment of the robot."""

    name = "capture_scene_segmentation"
    output_type = "string"
    description = (
        "Capture the top/side/front SO-101 camera feeds, run SmolVLM segmentation on each, "
        "and return a combined TOON block describing the detected objects and environments."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Types of objects to search for or general request about what's in the environment.",
            "default": DEFAULT_PROMPT,
            "nullable": True,
        },
        "delay_seconds": {
            "type": "number",
            "description": "Seconds to wait before capturing frames.",
            "default": 3.0,
            "nullable": True,
        },
    }
    outputs = {
        "toon": {
            "type": "string",
            "description": "TOON table summarizing all detected objects across the cameras.",
        },
        "summaries_json": {
            "type": "string",
            "description": "JSON blob containing the per-camera segmentation metadata.",
        },
        "output_dir": {
            "type": "string",
            "description": "Directory where annotated frames were saved.",
        },
    }

    def forward(
        self,
        prompt: str | None = None,
        delay_seconds: float | None = None,
    ) -> dict[str, Any]:
        prompt_text = prompt or DEFAULT_PROMPT
        wait_seconds = max(0.0, float(delay_seconds) if delay_seconds is not None else 3.0)
        output_dir_path = Path(OUTPUT_DIR)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        cfg = SmolVLMSegmentationConfig(
            prompt=prompt_text,
            output_path=str(output_dir_path / "placeholder.png"),
            model_id=SEGMENTATION_DEFAULT_MODEL_ID
        )
        processor, model, resolved_device = load_model(cfg)

        camera_map = self._build_camera_map()
        width = SEGMENTATION_DEFAULT_FRAME_WIDTH
        height = SEGMENTATION_DEFAULT_FRAME_HEIGHT

        if wait_seconds:
            logger.info("Waiting %.2fs before capturing frames", wait_seconds)
            time.sleep(wait_seconds)

        summaries: dict[str, Any] = {}
        combined_objects: list[dict[str, Any]] = []
        for name, index in camera_map.items():
            logger.info("Capturing %s camera at index %s", name, index)
            frame = self._capture_frame(index=index, width=width, height=height)

            camera_output = output_dir_path / f"{name}.png"
            camera_cfg = replace(cfg, output_path=str(camera_output))
            result = run_segmentation(
                camera_cfg,
                image=frame,
                processor=processor,
                model=model,
                device=resolved_device,
            )
            normalized = self._normalize_result(result)
            summaries[name] = normalized
            combined_objects.extend(self._objects_for_view(name, normalized))
            logger.info("Annotated %s camera frame saved to %s", name, camera_output)

        toon = self._format_toon(combined_objects)
        return {
            "toon": toon,
            "summaries_json": json.dumps(summaries, indent=2),
            "output_dir": str(output_dir_path),
        }

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

    @staticmethod
    def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
        normalized_objects = []
        for obj in result.get("objects") or []:
            point = obj.get("point") or (None, None)
            try:
                x_val = float(point[0])
                y_val = float(point[1])
            except (TypeError, ValueError):
                continue
            normalized_objects.append(
                {
                    "label": str(obj.get("label") or "object"),
                    "description": str(obj.get("reason") or obj.get("description") or ""),
                    "point_x": x_val,
                    "point_y": y_val,
                }
            )

        return {
            "label": str(result.get("label") or ""),
            "reason": str(result.get("reason") or ""),
            "objects": normalized_objects,
            "output_path": result.get("output_path") or "",
            "raw_response": result.get("raw_response") or "",
        }

    @staticmethod
    def _objects_for_view(view_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for obj in summary.get("objects", []):
            label = SmolVLMSegmentationTool._sanitize_field(obj.get("label", "object"))
            description = obj.get("description") or summary.get("reason") or ""
            objects.append(
                {
                    "label": f"{view_name}.{label}",
                    "point_x": obj.get("point_x", 0.5),
                    "point_y": obj.get("point_y", 0.5),
                    "description": SmolVLMSegmentationTool._sanitize_field(description or "object of interest"),
                }
            )
        return objects

    @staticmethod
    def _format_toon(objects: list[dict[str, Any]]) -> str:
        rows = objects or [
            {
                "label": "scene.none",
                "point_x": 0.5,
                "point_y": 0.5,
                "description": "No segmentation candidates returned",
            }
        ]
        lines = [
            "  "
            + ",".join(
                [
                    obj["label"],
                    f"{float(obj['point_x']):.4f}",
                    f"{float(obj['point_y']):.4f}",
                    SmolVLMSegmentationTool._sanitize_field(obj["description"]),
                ]
            )
            for obj in rows
        ]
        table = "\n".join(lines)
        return (
            "TOON\n"
            "objects[<=9]{label,point_x,point_y,description}:\n"
            f"{table}\n"
            "END TOON"
        )

    @staticmethod
    def _sanitize_field(value: str) -> str:
        cleaned = str(value).replace("\n", " ").strip()
        return " ".join(cleaned.split()).replace(",", ";")
