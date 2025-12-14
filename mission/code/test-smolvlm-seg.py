#!/usr/bin/env python3
"""Capture frames from the default SO-101 cameras and run SmolVLM segmentation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
from PIL import Image

from inference.smolvlm_segmentation import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    SmolVLMSegmentationConfig,
    load_model,
    run as run_segmentation,
)

DEFAULT_CAMERA_LAYOUT = {
    "top": {"index": 6, "width": 640, "height": 480},
    "side": {"index": 4, "width": 640, "height": 480},
    "front": {"index": 2, "width": 640, "height": 480},
}


def _capture_frame(index: int, width: int, height: int, warmup_frames: int = 5) -> Image.Image:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera at index {index} could not be opened")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Grab a few frames so auto-exposure settles.
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
        description="Wait a few seconds, capture frames from all robot cameras, and run SmolVLM segmentation."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Segmentation instruction to pass to SmolVLM.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        help="How long to wait before capturing frames.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/smolvlm/cameras",
        help="Directory where annotated frames are stored.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = SmolVLMSegmentationConfig(
        prompt=args.prompt,
        output_path=str(output_dir / "placeholder.png"),
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

    summaries = {}
    for name, index in camera_map.items():
        print(f"Capturing from {name} camera (index {index})")
        frame = _capture_frame(index, args.frame_width, args.frame_height)

        camera_output = output_dir / f"{name}.png"
        cfg = replace(base_cfg, output_path=str(camera_output))
        result = run_segmentation(
            cfg,
            image=frame,
            processor=processor,
            model=model,
            device=device,
        )
        summaries[name] = result
        print(f"Saved annotated frame for {name} camera to {camera_output}")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
