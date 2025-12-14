#!/usr/bin/env python3
"""Capture frames from the SO-101 camera layout and describe each view with SmolVLM."""

from __future__ import annotations

import argparse
import json
import re
import time

import cv2
import torch
from PIL import Image

from inference.smolvlm_segmentation import (
    DEFAULT_MODEL_ID,
    SmolVLMSegmentationConfig,
    load_model,
)

DEFAULT_CAMERA_LAYOUT = {
    "top": {"index": 6, "width": 640, "height": 480},
    "side": {"index": 4, "width": 640, "height": 480},
    "front": {"index": 2, "width": 640, "height": 480},
}

_GRID_TOKEN_PATTERN = re.compile(r"<row_\d+_col_\d+>", re.IGNORECASE)


def _capture_frame(index: int, width: int, height: int, warmup_frames: int = 5) -> Image.Image:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera at index {index} could not be opened")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(warmup_frames):
        cap.read()

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from camera index {index}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture frames from all robot cameras and ask SmolVLM to describe what it sees."
    )
    parser.add_argument(
        "--prompt",
        default="Describe what you see in this camera feed.",
        help="Instruction passed to SmolVLM.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        help="How long to wait before capturing frames.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="SmolVLM checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override for the model (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype override passed to the model loader.",
    )
    parser.add_argument(
        "--top-index",
        type=int,
        default=DEFAULT_CAMERA_LAYOUT["top"]["index"],
        help="Camera index for the top camera.",
    )
    parser.add_argument(
        "--side-index",
        type=int,
        default=DEFAULT_CAMERA_LAYOUT["side"]["index"],
        help="Camera index for the side camera.",
    )
    parser.add_argument(
        "--front-index",
        type=int,
        default=DEFAULT_CAMERA_LAYOUT["front"]["index"],
        help="Camera index for the front camera.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Width to request from each camera stream.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=480,
        help="Height to request from each camera stream.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per description.",
    )
    return parser.parse_args()


def _describe_image(
    processor,
    model,
    device,
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
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return _clean_model_response(outputs[0])


def _clean_model_response(raw_text: str) -> str:
    """Drop template bookkeeping (User rows, grid tokens) and keep the assistant reply."""

    text = raw_text.strip()
    lowered = text.lower()
    assistant_marker = "assistant:"
    marker_idx = lowered.rfind(assistant_marker)
    if marker_idx != -1:
        text = text[marker_idx + len(assistant_marker) :]
    text = _GRID_TOKEN_PATTERN.sub("", text)
    return text.strip()


def main() -> None:
    args = _parse_args()

    base_cfg = SmolVLMSegmentationConfig(
        prompt=args.prompt,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    processor, model, device = load_model(base_cfg)

    camera_map = {
        "top": args.top_index,
        "side": args.side_index,
        "front": args.front_index,
    }

    print(f"Waiting {args.delay_seconds:.1f}s before capturing frames...")
    time.sleep(max(0.0, args.delay_seconds))

    descriptions = {}
    for name, index in camera_map.items():
        print(f"Capturing from {name} camera (index {index})")
        frame = _capture_frame(index, args.frame_width, args.frame_height)
        description = _describe_image(
            processor,
            model,
            device,
            frame,
            args.prompt,
            args.max_new_tokens,
        )
        descriptions[name] = {
            "description": description,
        }
        print(f"{name}: {description}")

    print(json.dumps(descriptions, indent=2))


if __name__ == "__main__":
    main()
