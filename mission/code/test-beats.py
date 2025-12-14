#!/usr/bin/env python3
"""Make mel-spectrogram and chroma images per sliding audio window."""

from __future__ import annotations

import argparse
import json
import itertools
import logging
import random
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    ACTION,
    OBS_STR,
    build_dataset_frame,
    hw_to_dataset_features,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from mutagen import File as MutagenFile

from beats_utils import (
    feature_to_image,
    sliding_windows,
    samples_for_duration,
    format_run_filename,
)

LOG = logging.getLogger(__name__)

SERVO_PLACEHOLDER = [
    {
        "name": "so101_teach_joint",
        "angle": 0.0,
        "velocity": 0.0,
        "effort": 0.0,
    }
]
SERVO_NAMES = [entry["name"] for entry in SERVO_PLACEHOLDER]
SERVO_PLACEHOLDER_VALUES = {entry["name"]: entry["angle"] for entry in SERVO_PLACEHOLDER}
TASK_PREFIX = "Dance to the beat of "
STATE_FEATURE_DIM = 20
STATE_FEATURE_NAMES = tuple(f"state_{idx}" for idx in range(STATE_FEATURE_DIM))
STATE_VALUES_TEMPLATE = {name: 0.0 for name in STATE_FEATURE_NAMES}
DEFAULT_ANIMATION_DIR = Path("outputs/keyframe_animations")


@dataclass(slots=True)
class AnimationKeyframe:
    """Simple holder for a keyframe in a saved animation."""

    time: float
    positions: dict[str, float]


@dataclass(slots=True)
class AnimationClip:
    """Description of a recorded animation that can be replayed."""

    name: str
    duration: float
    keyframes: list[AnimationKeyframe]


def _load_animation_clips(animation_dir: Path) -> tuple[list[AnimationClip], set[str]]:
    """Parse every JSON animation in *animation_dir*."""

    animations: list[AnimationClip] = []
    servo_names: set[str] = set()
    if not animation_dir.exists():
        return animations, servo_names

    for path in sorted(animation_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            LOG.warning("Failed to read animation %s: %s", path.name, exc)
            continue

        keyframes: list[AnimationKeyframe] = []
        for item in payload.get("keyframes", []):
            if not isinstance(item, Mapping):
                continue
            positions = {}
            for motor, value in item.get("positions", {}).items():
                if not isinstance(motor, str):
                    continue
                try:
                    positions[motor] = float(value)
                except (TypeError, ValueError):
                    continue
            if not positions:
                continue
            servo_names.update(positions.keys())
            time_value = float(item.get("time", 0.0))
            keyframes.append(AnimationKeyframe(time=time_value, positions=positions))

        if not keyframes:
            continue

        keyframes.sort(key=lambda kf: kf.time)
        duration = float(payload.get("duration", keyframes[-1].time))
        duration = max(duration, keyframes[-1].time)
        clip_name = str(payload.get("name") or path.stem)
        animations.append(AnimationClip(name=clip_name, duration=duration, keyframes=keyframes))

    return animations, servo_names


def _interpolate_animation_pose(time_value: float, keyframes: list[AnimationKeyframe]) -> dict[str, float]:
    """Linear interpolation between animation keyframes."""

    if not keyframes:
        return {}
    if time_value <= keyframes[0].time:
        return {name: float(value) for name, value in keyframes[0].positions.items()}
    if time_value >= keyframes[-1].time:
        return {name: float(value) for name, value in keyframes[-1].positions.items()}

    for left, right in zip(keyframes, keyframes[1:]):
        if left.time <= time_value <= right.time:
            span = right.time - left.time
            alpha = 0.0 if span == 0 else (time_value - left.time) / span
            result: dict[str, float] = {}
            left_positions = left.positions
            right_positions = right.positions
            union_names = left_positions.keys() | right_positions.keys()
            for name in union_names:
                start = left_positions.get(name, 0.0)
                end = right_positions.get(name, start)
                result[name] = float(start + (end - start) * alpha)
            return result

    return {name: float(value) for name, value in keyframes[-1].positions.items()}


def _zero_pose(servo_names: list[str]) -> dict[str, float]:
    """Create a zeroed servo pose dict for the requested names."""

    return {name: 0.0 for name in servo_names}


def _extract_frame_text(frame: object) -> str | None:
    """Pull a single string from a mutagen ID3 frame, if any text exists."""
    if frame is None:
        return None
    text = getattr(frame, "text", None)
    if isinstance(text, (list, tuple)) and text:
        return str(text[0])
    if isinstance(text, str):
        return text
    return str(frame)


def describe_task_from_metadata(audio_path: Path) -> str:
    """Return a task description built from the audio metadata when available."""
    metadata = MutagenFile(str(audio_path))
    if metadata is None:
        subject = audio_path.stem
    else:
        tags = getattr(metadata, "tags", None)
        subject = None
        if tags:
            for key in ("TIT2", "©nam", "TPE1", "©ART", "title", "artist"):
                frame = tags.get(key)
                if frame is None:
                    continue
                candidate = _extract_frame_text(frame)
                if candidate:
                    subject = candidate
                    break
        if subject is None:
            subject = audio_path.stem
    return f"{TASK_PREFIX}{subject}"


def positive_float(value: str) -> float:
    """Require a positive floating point number for argparse."""
    converted = float(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return converted


def positive_int(value: str) -> int:
    """Require a positive integer for argparse."""
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return converted


def non_negative_int(value: str) -> int:
    """Require a non-negative integer for argparse."""
    converted = int(value)
    if converted < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return converted


def parse_args() -> argparse.Namespace:
    """Collect CLI configuration for the extractor."""
    parser = argparse.ArgumentParser(
        description="Produce mel and chroma images for each sliding window of audio."
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="+",
        help="Path(s) to the source audio files that will be analyzed.",
    )
    parser.add_argument(
        "--run-number",
        type=non_negative_int,
        default=0,
        help="Run identifier that is embedded in every filename.",
    )
    parser.add_argument(
        "--window-duration",
        type=positive_float,
        default=3.0,
        help="Seconds captured in each sliding window.",
    )
    parser.add_argument(
        "--hop-duration",
        type=positive_float,
        default=1 / 30,
        help="Seconds between successive windows.",
    )
    parser.add_argument(
        "--sr",
        type=positive_int,
        default=22050,
        help="Target sampling rate for loading the audio.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Width and height in pixels for the output images.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description assigned to every dataset frame; defaults to metadata-based description.",
    )
    parser.add_argument(
        "--episode-duration",
        type=positive_float,
        default=20.0,
        help="Seconds of audio that go into each episode before saving and starting a new one.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, write the mel/chroma PNGs to disk; otherwise they are only stored in the dataset.",
    )
    parser.add_argument(
        "--real-run",
        action="store_true",
        help="Drive servo values from saved keyframe animations (outputs/keyframe_animations) instead of the placeholder pose.",
    )
    parser.add_argument(
        "--animation-dir",
        type=Path,
        default=DEFAULT_ANIMATION_DIR,
        help="Directory containing JSON animations exported by test-keyframe-gui.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID under which the dataset will be written (defaults to `eerwitt/xvla-beats-run{run}` ).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="If set, explicitly create the dataset at this root instead of the HF cache.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags that will be applied when pushing to the Hugging Face hub.",
    )
    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Do not push the dataset to the Hugging Face hub after recording.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Mark the dataset as private when pushing to the Hugging Face hub.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Set up INFO-level logging and minimize noisy dependency output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("librosa").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    args = parse_args()
    audio_paths = args.audio
    target_size = tuple(args.image_size)
    output_dir = Path("outputs") / "beats"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "Starting run %d over %d audio file(s) -> window=%.3fs hop=%.3fs size=%dx%d (debug=%s)",
        args.run_number,
        len(audio_paths),
        args.window_duration,
        args.hop_duration,
        target_size[0],
        target_size[1],
        args.debug,
    )

    real_run_active = args.real_run
    animation_dir = args.animation_dir or DEFAULT_ANIMATION_DIR
    animations: list[AnimationClip] = []
    servo_names_candidate: set[str] = set()
    if real_run_active:
        animations, servo_names_candidate = _load_animation_clips(animation_dir)
        if not animations or not servo_names_candidate:
            LOG.warning(
                "Real-run requested but %s contains no usable animation poses; falling back to placeholder actions.",
                animation_dir,
            )
            real_run_active = False
        else:
            random.shuffle(animations)
            LOG.info("Loaded %d animation(s) from %s", len(animations), animation_dir)

    initial_animation_pose: dict[str, float] | None = None
    if real_run_active and animations and animations[0].keyframes:
        initial_animation_pose = {
            name: float(value)
            for name, value in animations[0].keyframes[0].positions.items()
        }
    servo_names = sorted(servo_names_candidate) if real_run_active else SERVO_NAMES

    dataset_repo = (
        args.dataset_repo_id
        if args.dataset_repo_id is not None
        else f"eerwitt/xvla-beats-run{args.run_number:03d}"
    )
    dataset_root = args.dataset_root or HF_LEROBOT_HOME / dataset_repo
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    image_shape = (target_size[1], target_size[0], 3)
    obs_hw_features = {name: float for name in STATE_FEATURE_NAMES}
    obs_hw_features.update({"mel": image_shape, "chroma": image_shape})
    obs_features = hw_to_dataset_features(obs_hw_features, OBS_STR, use_video=True)
    action_features = hw_to_dataset_features({name: float for name in servo_names}, ACTION)
    features = {**obs_features, **action_features}
    fps = max(1, int(round(1.0 / args.hop_duration)))
    dataset = LeRobotDataset.create(
        repo_id=dataset_repo,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="test-beats",
        use_videos=True,
    )
    LOG.info("Recording dataset as %s -> %s (videos enabled)", dataset_repo, dataset_root)
    if not args.debug:
        LOG.info("Debug flag not set; mel/chroma PNGs are kept in-memory and not written to %s", output_dir)

    animation_iter = itertools.cycle(animations) if real_run_active else None
    current_animation: AnimationClip | None = None
    animation_playing = False
    animation_start_time = 0.0
    animation_duration = 0.0
    last_sent_pose = _zero_pose(servo_names)
    if real_run_active and initial_animation_pose:
        last_sent_pose = {
            name: initial_animation_pose.get(name, 0.0)
            for name in servo_names
        }
    frame_ind = 0

    def flush_episode_if_needed() -> None:
        buffer = dataset.episode_buffer
        if buffer is None or buffer.get("size", 0) == 0:
            return
        dataset.save_episode()

    total_time_offset = 0.0
    for audio_path in audio_paths:
        LOG.info("Processing %s", audio_path)
        y, sr = librosa.load(str(audio_path), sr=int(args.sr), mono=True)
        LOG.info(
            "Loaded %.2f seconds (%d samples) at %d Hz", len(y) / sr, len(y), sr
        )
        window_samples = samples_for_duration(args.window_duration, sr)
        hop_samples = samples_for_duration(args.hop_duration, sr)

        beat_frames: list[int] = []
        onset_envelope = np.array([0.0], dtype=np.float32)
        tempo = 0.0
        use_real_run = real_run_active
        if use_real_run:
            tempo, idx_arr = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_samples)
            beat_frames = [int(int(idx)) for idx in idx_arr.tolist()]
            if not beat_frames:
                LOG.warning(
                    "No beats detected in %s; falling back to placeholder actions.",
                    audio_path.name,
                )
                use_real_run = False
            else:
                raw_onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_samples)
                max_energy = float(np.max(raw_onset)) if raw_onset.size else 1.0
                if max_energy <= 0:
                    max_energy = 1.0
                onset_envelope = (raw_onset / max_energy).astype(np.float32)
                LOG.info(
                    "Detected tempo %.1f BPM with %d beats (hop=%d samples) for %s",
                    tempo,
                    len(beat_frames),
                    hop_samples,
                    audio_path.name,
                )

        task_description = args.task if args.task else describe_task_from_metadata(audio_path)

        audio_frame_ind = 0
        beat_index = 0
        episode_reference_time = total_time_offset
        for start_sample, window in sliding_windows(y, window_samples, hop_samples):
            start_time = total_time_offset + start_sample / sr
            end_time = total_time_offset + (start_sample + window_samples) / sr
            if args.episode_duration and start_time >= episode_reference_time + args.episode_duration:
                flush_episode_if_needed()
                episode_reference_time = start_time

            base_name = f"run{args.run_number:03d}_frame{frame_ind:05d}"
            mel_name = format_run_filename(args.run_number, frame_ind, "mel")
            chroma_name = format_run_filename(args.run_number, frame_ind, "chroma")
            mel_img = feature_to_image(window, sr, "mel", target_size)
            chroma_img = feature_to_image(window, sr, "chroma", target_size)
            LOG.info(
                "Frame %05d [%05d:%05d samples -> %.3fs-%.3fs s]: saving %s_(mel|chroma).png",
                frame_ind,
                start_sample,
                start_sample + window_samples,
                start_time,
                end_time,
                base_name,
            )
            if args.debug:
                mel_img.save(output_dir / mel_name)
                chroma_img.save(output_dir / chroma_name)
            observation_values = {
                "mel": np.asarray(mel_img, dtype=np.uint8),
                "chroma": np.asarray(chroma_img, dtype=np.uint8),
                **STATE_VALUES_TEMPLATE,
            }
            if use_real_run:
                is_beat = beat_index < len(beat_frames) and audio_frame_ind == beat_frames[beat_index]
                if is_beat:
                    beat_index += 1

                canonical_pose = dict(last_sent_pose)
                if animation_playing and current_animation is not None:
                    elapsed = start_time - animation_start_time
                    canonical_pose = _interpolate_animation_pose(elapsed, current_animation.keyframes)
                    if elapsed >= animation_duration:
                        animation_playing = False
                        current_animation = None
                elif is_beat and animations:
                    current_animation = next(animation_iter) if animation_iter is not None else None
                    if current_animation is not None:
                        animation_duration = max(
                            current_animation.duration,
                            current_animation.keyframes[-1].time if current_animation.keyframes else 0.0,
                        )
                        animation_start_time = start_time
                        canonical_pose = _interpolate_animation_pose(0.0, current_animation.keyframes)
                        animation_playing = animation_duration > 0.0 and bool(current_animation.keyframes)
                        if not animation_playing:
                            current_animation = None

                expanded_canonical = {
                    name: canonical_pose.get(name, last_sent_pose.get(name, 0.0)) for name in servo_names
                }
                energy_index = min(audio_frame_ind, len(onset_envelope) - 1) if onset_envelope.size else 0
                energy = float(onset_envelope[energy_index]) if onset_envelope.size else 0.0
                energy = max(0.0, min(1.0, energy))

                action_values = {}
                for name in servo_names:
                    base = last_sent_pose.get(name, 0.0)
                    target = expanded_canonical.get(name, base)
                    action_values[name] = float(base + (target - base) * energy)
                last_sent_pose = dict(action_values)
            else:
                action_values = dict(SERVO_PLACEHOLDER_VALUES)
            observation_frame = build_dataset_frame(dataset.features, observation_values, OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_values, ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": task_description})
            frame_ind += 1
            audio_frame_ind += 1

        flush_episode_if_needed()
        total_time_offset += len(y) / sr

    flush_episode_if_needed()
    dataset.meta._flush_metadata_buffer()
    dataset.finalize()
    LOG.info("Completed %d frames (dataset saved to %s)", frame_ind, dataset_root)

    if not args.no_push_to_hub:
        tag_list = (
            [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            if args.tags
            else None
        )
        dataset.push_to_hub(tags=tag_list, private=args.private)
        LOG.info("Uploaded %s to the Hugging Face hub", dataset_repo)
    else:
        LOG.info("Skipping upload because --no-push-to-hub was set")


if __name__ == "__main__":
    main()
