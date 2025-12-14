#!/usr/bin/env python3
"""SmolVLM inference helper that turns segmentation prompts into highlighted images."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText as _AutoModelForVLM
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_OUTPUT_PATH = "outputs/smolvlm/segmented.png"
DEFAULT_PROMPT = (
    "Search the scene for the most contextually relevant object to interact with, describe it, "
    "and provide a single representative point on that object."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are an agent controlling a robot. Identify up to three objects the robot should care about "
    "and respond strictly in Token-Oriented Object Notation (TOON) using the template:\n"
    "TOON\n"
    'objects[<=3]{label,point_x,point_y,description}:\n'
    "  label,point_x,point_y,description\n"
    "END TOON\n"
    "Coordinates must be floats between 0 and 1 relative to the provided image. Do not include prose "
    "outside the TOON block."
)
CODE_BLOCK_PATTERN = re.compile(r"^```(?:json|toon)?\s*(.*?)```$", re.DOTALL)
TOON_BLOCK_PATTERN = re.compile(r"TOON\s*(.*?)\s*END TOON", re.IGNORECASE | re.DOTALL)
TOON_TABLE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)\[(?P<count><=?\d+)\]\{(?P<fields>[^}]*)\}:\s*$",
    re.IGNORECASE,
)


@dataclass
class SmolVLMSegmentationConfig:
    """Configuration describing how to run SmolVLM segmentation inference."""

    image_path: str | None = None
    prompt: str = DEFAULT_PROMPT
    output_path: str = DEFAULT_OUTPUT_PATH
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 128
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    device: str | None = None
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    use_flash_attention_on_cuda: bool = True
    point_radius: int = 14
    point_color: tuple[int, int, int] = field(default_factory=lambda: (255, 80, 80))
    label_background: tuple[int, int, int, int] = field(
        default_factory=lambda: (0, 0, 0, 160)
    )
    label_text_color: tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))
    response_format: str = "toon"
    format_retry_attempts: int = 2


def load_model(
    cfg: SmolVLMSegmentationConfig,
) -> tuple[AutoProcessor, Any, torch.device]:
    """Load the processor/model pair for repeated inference."""

    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.torch_dtype, device)
    attn_impl = cfg.attn_implementation or _default_attn_impl(
        device, cfg.use_flash_attention_on_cuda
    )
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = _AutoModelForVLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        _attn_implementation=attn_impl,
    ).to(device)
    model.eval()
    return processor, model, device


def run(
    cfg: SmolVLMSegmentationConfig,
    image: Image.Image | None = None,
    *,
    processor: AutoProcessor | None = None,
    model: Any | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run SmolVLM on the provided image and save the highlighted output."""

    if image is None:
        if not cfg.image_path:
            raise ValueError("cfg.image_path must be provided when image is None")
        image_path = Path(cfg.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.info("Loaded image %s with size %s", image_path, image.size)
    else:
        logger.info("Running SmolVLM on in-memory image with size %s", image.size)

    proc = processor
    mdl = model
    dev = _normalize_device_arg(device)
    if proc is None or mdl is None:
        proc, mdl, dev = load_model(cfg)
    elif dev is None:
        dev = next(mdl.parameters()).device

    model_input_size = _infer_model_input_size(proc)

    messages = _build_messages(cfg.system_prompt, cfg.prompt, image.size)
    attempts = max(1, cfg.format_retry_attempts)
    generated_text = ""
    parsed: dict[str, Any] | None = None

    for attempt in range(attempts):
        generated_text = _generate_model_response(
            proc,
            mdl,
            dev,
            image,
            messages,
            cfg.max_new_tokens,
        )
        logger.info("Model raw response: %s", generated_text)

        try:
            if cfg.response_format.lower() == "toon":
                parsed = _parse_toon_output(generated_text)
            else:
                parsed = _parse_model_output(generated_text)
            break
        except ValueError as exc:
            logger.warning("Failed to parse response on attempt %d: %s", attempt + 1, exc)
            if attempt >= attempts - 1:
                raise
            messages.extend(
                _build_retry_messages(cfg.response_format, generated_text)
            )

    if parsed is None:
        raise RuntimeError("Parsing failed unexpectedly despite retries")
    model_point = _normalized_point_to_pixels(parsed["point"], model_input_size)
    px_point = _map_model_point_to_original(
        model_point, image.size, model_input_size
    )
    annotated = _draw_highlight(
        base=image,
        point=px_point,
        label=parsed["label"],
        reason=parsed["reason"],
        radius=cfg.point_radius,
        point_color=cfg.point_color,
        label_background=cfg.label_background,
        label_text_color=cfg.label_text_color,
    )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    logger.info("Saved highlighted image to %s", output_path)

    return {
        "point": {"x": px_point[0], "y": px_point[1]},
        "label": parsed["label"],
        "reason": parsed["reason"],
        "raw_response": generated_text,
        "parsed_response": parsed,
        "objects": parsed.get("objects"),
        "output_path": str(output_path),
    }


def _resolve_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype '{dtype_name}'")
    return dtype


def _default_attn_impl(device: torch.device, enable_flash: bool) -> str:
    has_flashattn = True
    try:
        import flash_attn
    except ImportError:
        has_flashattn = False
    if has_flashattn and enable_flash and device.type == "cuda":
        return "flash_attention_2"
    return "eager"


def _normalize_device_arg(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _build_messages(
    system_prompt: str, user_prompt: str, image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    width, height = image_size
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{user_prompt.strip()}\n"
                        f"The image resolution is {width} by {height} pixels. "
                        "Respond only with the requested TOON block."
                    ),
                },
            ],
        },
    ]


def _build_retry_messages(response_format: str, bad_response: str) -> list[dict[str, Any]]:
    if response_format.lower() == "toon":
        text = (
            "The previous reply was invalid. Respond again using ONLY this TOON template:\n"
            "TOON\nobjects[<=3]{label,point_x,point_y,description}:\n  <rows>\nEND TOON\n"
            "Fill each row with a label, normalized point_x/point_y floats, and a short description."
        )
    else:
        text = (
            "The previous reply was invalid. Respond using ONLY valid JSON matching the requested schema."
        )
    return [
        {"role": "assistant", "content": [{"type": "text", "text": bad_response}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _parse_model_output(text: str) -> dict[str, Any]:
    """Extract the JSON blob and normalize its contents."""

    cleaned = text.strip()
    candidates = list(_iter_json_candidates(cleaned))
    data = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict):
                data = parsed
                break

    if not isinstance(data, dict):
        if last_error:
            raise ValueError(
                f"Could not parse JSON response from model: {cleaned}"
            ) from last_error
        raise ValueError(f"Could not parse JSON response from model: {cleaned}")

    point = data.get("point") or data.get("coordinates")
    if (
        not isinstance(point, Sequence)
        or isinstance(point, (str, bytes))
        or len(point) != 2
    ):
        raise ValueError(f"Model response missing point coordinates: {data}")

    try:
        normalized_point = (float(point[0]), float(point[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid point coordinates: {point}") from exc

    label = (
        data.get("label")
        or data.get("object")
        or data.get("target")
        or "object"
    )
    reason = (
        data.get("reason")
        or data.get("rationale")
        or data.get("explanation")
        or ""
    )

    return {
        "label": str(label),
        "reason": str(reason),
        "point": normalized_point,
        "json": data,
        "objects": [
            {"label": str(label), "reason": str(reason), "point": normalized_point}
        ],
    }


def _parse_toon_output(text: str) -> dict[str, Any]:
    """Parse the TOON response emitted by the model."""

    cleaned = _strip_code_fence(text)
    block_match = TOON_BLOCK_PATTERN.search(cleaned)
    if not block_match:
        raise ValueError(f"Could not find TOON block in response: {text}")
    content = block_match.group(1)

    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    header_idx = None
    header_match = None
    for idx, line in enumerate(lines):
        match = TOON_TABLE_PATTERN.match(line.strip())
        if match:
            header_idx = idx
            header_match = match
            break

    if header_match is None or header_idx is None:
        raise ValueError(f"TOON block missing objects table header: {text}")

    fields = [field.strip() for field in header_match.group("fields").split(",") if field.strip()]
    if not fields:
        raise ValueError(f"No fields declared in TOON table: {text}")

    data_lines = lines[header_idx + 1 :]
    if not data_lines:
        raise ValueError(f"No rows found in TOON table: {text}")

    objects = []
    for line in data_lines:
        row = [part.strip() for part in line.split(",")]
        if len(row) < len(fields):
            row.extend([""] * (len(fields) - len(row)))
        row_dict = {field: value for field, value in zip(fields, row)}

        label = (
            row_dict.get("label")
            or row_dict.get("name")
            or row_dict.get("object")
            or ""
        ).strip()
        reason = (
            row_dict.get("description")
            or row_dict.get("reason")
            or row_dict.get("notes")
            or ""
        ).strip()
        try:
            x_val = float(row_dict.get("point_x") or row_dict.get("x"))
            y_val = float(row_dict.get("point_y") or row_dict.get("y"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid coordinates in row: {line}") from exc
        objects.append({"label": label or "object", "reason": reason, "point": (x_val, y_val)})

    if not objects:
        raise ValueError(f"Could not parse TOON response from model: {text}")

    primary = objects[0]
    return {
        "label": primary["label"],
        "reason": primary["reason"],
        "point": primary["point"],
        "objects": objects,
    }


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Yield increasingly relaxed substrings that might contain valid JSON."""

    stripped = _strip_code_fence(text)
    if stripped:
        yield stripped
    yield text

    for candidate in _balanced_brace_slices(text):
        yield candidate


def _strip_code_fence(text: str) -> str:
    match = CODE_BLOCK_PATTERN.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _balanced_brace_slices(text: str) -> Iterable[str]:
    depth = 0
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : idx + 1]
                    start = None


def _generate_model_response(
    processor: AutoProcessor,
    model,
    device: torch.device,
    image: Image.Image,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    formatted_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _normalized_point_to_pixels(point: tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    x_norm, y_norm = point

    if math.isfinite(x_norm) and 0.0 <= x_norm <= 1.0:
        x = x_norm * (width - 1)
    else:
        x = x_norm
    if math.isfinite(y_norm) and 0.0 <= y_norm <= 1.0:
        y = y_norm * (height - 1)
    else:
        y = y_norm

    x = max(0, min(int(round(x)), width - 1))
    y = max(0, min(int(round(y)), height - 1))
    return (x, y)


def _map_model_point_to_original(
    model_point: tuple[int, int],
    original_size: tuple[int, int],
    model_size: tuple[int, int],
) -> tuple[int, int]:
    orig_w, orig_h = original_size
    model_w, model_h = model_size
    if orig_w == 0 or orig_h == 0:
        return (0, 0)

    if orig_w >= orig_h:
        scale = model_h / float(orig_h)
    else:
        scale = model_w / float(orig_w)

    resized_w = orig_w * scale
    resized_h = orig_h * scale
    crop_x = max(0.0, (resized_w - model_w) / 2.0)
    crop_y = max(0.0, (resized_h - model_h) / 2.0)

    x_model, y_model = model_point
    x_resized = x_model + crop_x
    y_resized = y_model + crop_y

    x_orig = x_resized / scale
    y_orig = y_resized / scale

    x_clamped = max(0, min(int(round(x_orig)), orig_w - 1))
    y_clamped = max(0, min(int(round(y_orig)), orig_h - 1))
    return (x_clamped, y_clamped)


def _infer_model_input_size(processor: AutoProcessor) -> tuple[int, int]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return (384, 384)

    crop_size = getattr(image_processor, "crop_size", None)
    if isinstance(crop_size, dict):
        width = crop_size.get("width")
        height = crop_size.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return (width, height)

    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        width = size.get("width")
        height = size.get("height")
        shortest = size.get("shortest_edge")
        if isinstance(width, int) and isinstance(height, int):
            return (width, height)
        if isinstance(shortest, int):
            return (shortest, shortest)
    if isinstance(size, int):
        return (size, size)

    return (384, 384)


def _draw_highlight(
    base: Image.Image,
    point: tuple[int, int],
    label: str,
    reason: str,
    radius: int,
    point_color: tuple[int, int, int],
    label_background: tuple[int, int, int, int],
    label_text_color: tuple[int, int, int],
) -> Image.Image:
    """Draw a circular highlight at the predicted point and annotate with the label."""

    annotated = base.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    x, y = point
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=(*point_color, 128), outline=point_color, width=3)

    font = ImageFont.load_default()
    label_text = label or "object"
    if reason:
        label_text += f": {reason}"

    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 4
    rect = (
        x + radius + padding,
        y - text_height // 2 - padding,
        x + radius + padding + text_width + 2 * padding,
        y + text_height // 2 + padding,
    )
    draw.rectangle(rect, fill=label_background)
    draw.text(
        (rect[0] + padding, rect[1] + padding),
        label_text,
        fill=label_text_color,
        font=font,
    )
    return annotated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLM segmentation inference.")
    parser.add_argument("image", help="Path to the RGB image to analyze.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Segmentation prompt describing which object to highlight.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to store the annotated visualization.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face checkpoint to load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype to run the model with.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SmolVLMSegmentationConfig(
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    result = run(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
