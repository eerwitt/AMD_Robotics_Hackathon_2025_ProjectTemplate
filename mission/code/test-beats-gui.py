#!/usr/bin/env python3
"""Generate and preview test-beats output with synchronized audio controls."""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import librosa
import pygame
import pyarrow.parquet as pq

from lerobot.motors import MotorsBus
from lerobot.robots import so101_follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import register_third_party_plugins

DEFAULT_ROBOT_ID = "so101_follower"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"

from beats_utils import (
    feature_to_image,
    format_run_filename,
    samples_for_duration,
    sliding_windows,
)

LOG = logging.getLogger(__name__)

CALIBRATION_SNAPSHOT_DIR = Path("outputs/calibration_snapshots")
STARTUP_SEQUENCE: tuple[tuple[str, float], ...] = (
    ("wakeup-step-01.json", 2.0),
    ("wakeup-step-02.json", 3.0),
    ("wakeup-step-03.json", 2.0),
)
WINK_SERVO_CANDIDATES = ("gripper", "wrist_roll", "wrist_flex")
WINK_AMOUNT = 200
WINK_LEFT_DURATION = 0.5
WINK_RIGHT_DURATION = 0.3
WINK_RESET_DURATION = 0.5


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("librosa").setLevel(logging.WARNING)


def positive_float(value: str) -> float:
    converted = float(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return converted


def non_negative_int(value: str) -> int:
    converted = int(value)
    if converted < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and generate test-beats runs.")
    parser.add_argument(
        "audio",
        type=Path,
        help="MP3 (or pygame-supported) file that drives the generation and playback.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs") / "beats",
        help="Directory where the mel/chroma images are stored.",
    )
    parser.add_argument(
        "--run-number",
        type=non_negative_int,
        default=None,
        help="Run index to overwrite/preview; defaults to incrementing the latest run.",
    )
    parser.add_argument(
        "--window-duration",
        type=positive_float,
        default=3.0,
        help="Duration (seconds) of the sliding window passed to librosa.",
    )
    parser.add_argument(
        "--hop-duration",
        type=positive_float,
        default=1 / 30,
        help="Seconds between successive windows.",
    )
    parser.add_argument(
        "--sr",
        type=positive_float,
        default=22050.0,
        help="Target sampling rate for the audio generation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Dimensions pixels for each camera image.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional path to a LeRobot test-beats dataset for servo timeline visualization.",
    )
    parser.add_argument(
        "--robot-enable",
        action="store_true",
        help="Connect to the SO-101 follower and replay servo positions while audio plays.",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default=None,
        help="Serial port for the follower robot (overrides default).",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default=None,
        help="Identifier for the follower robot (overrides default).",
    )
    parser.add_argument(
        "--robot-calibrate",
        action="store_true",
        help="Run calibration routine immediately after connecting to the follower.",
    )
    return parser.parse_args()


def find_latest_run(output_dir: Path) -> int:
    best = -1
    if not output_dir.exists():
        return best
    for entry in output_dir.iterdir():
        if not entry.name.startswith("run"):
            continue
        prefix = entry.name.split("_", 1)[0]
        if not prefix.startswith("run"):
            continue
        candidate = prefix[3:]
        if not candidate.isdigit():
            continue
        run_idx = int(candidate)
        best = max(best, run_idx)
    return best


def remove_run_files(output_dir: Path, run_number: int) -> None:
    if not output_dir.exists():
        return
    prefix = f"run{run_number:03d}_"
    for entry in output_dir.iterdir():
        if entry.name.startswith(prefix):
            entry.unlink()


class FrameProducer(threading.Thread):
    def __init__(
        self,
        audio_path: Path,
        run_number: int,
        output_dir: Path,
        window_duration: float,
        hop_duration: float,
        sr: int,
        image_size: Tuple[int, int],
        frame_buffers: Dict[int, Tuple[bytes, bytes]],
        buffer_lock: threading.Lock,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.run_number = run_number
        self.output_dir = output_dir
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.sr = sr
        self.image_size = image_size
        self.frame_buffers = frame_buffers
        self.buffer_lock = buffer_lock
        self.stop_event = stop_event
        self.frames_generated = 0
        self.completed = threading.Event()

    def run(self) -> None:
        y, sr = librosa.load(str(self.audio_path), sr=self.sr, mono=True)
        window_samples = samples_for_duration(self.window_duration, sr)
        hop_samples = samples_for_duration(self.hop_duration, sr)
        LOG.info(
            "Producer loaded %.2f seconds (%d samples) at %d Hz", len(y) / sr, len(y), sr
        )
        frame_idx = 0
        for start_sample, window in sliding_windows(y, window_samples, hop_samples):
            if self.stop_event.is_set():
                break
            base_name = f"run{self.run_number:03d}_frame{frame_idx:05d}"
            mel_img = feature_to_image(window, sr, "mel", self.image_size)
            chroma_img = feature_to_image(window, sr, "chroma", self.image_size)
            mel_name = format_run_filename(self.run_number, frame_idx, "mel")
            chroma_name = format_run_filename(self.run_number, frame_idx, "chroma")
            mel_path = self.output_dir / mel_name
            chroma_path = self.output_dir / chroma_name
            mel_img.save(mel_path)
            chroma_img.save(chroma_path)
            mel_bytes = mel_img.convert("RGB").tobytes()
            chroma_bytes = chroma_img.convert("RGB").tobytes()
            with self.buffer_lock:
                self.frame_buffers[frame_idx] = (mel_bytes, chroma_bytes)
            LOG.info(
                "Generated frame %03d (%s + %s)", frame_idx, mel_path.name, chroma_path.name
            )
            frame_idx += 1
            self.frames_generated = frame_idx
        self.completed.set()


def create_placeholder(size: Tuple[int, int], label: str, font: pygame.font.Font) -> pygame.Surface:
    surface = pygame.Surface(size)
    surface.fill((20, 20, 20))
    text = font.render(label, True, (220, 220, 220))
    surface.blit(text, text.get_rect(center=(size[0] / 2, size[1] / 2)))
    return surface.convert()


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


@dataclass(slots=True)
class ServoHistory:
    """Stores servo trajectories extracted from a LeRobot dataset."""

    servo_names: list[str]
    values: list[list[float]]
    timestamps: list[float]
    fps: float

    def frame_for_time(self, current_time: float) -> int:
        if not self.values:
            return 0
        if self.timestamps:
            idx = bisect.bisect_right(self.timestamps, current_time) - 1
        else:
            idx = int(current_time * self.fps)
        idx = max(0, min(len(self.values) - 1, idx))
        return idx


class ServoChart:
    """Draws servo trajectories inside a fixed area."""

    CHART_BG = (24, 24, 24)
    CHART_BORDER = (90, 90, 90)
    LINE_COLOR = (90, 180, 240)
    HIGHLIGHT_COLOR = (255, 170, 60)
    LABEL_COLOR = (220, 220, 220)
    VALUE_COLOR = (255, 200, 120)

    def __init__(self, history: ServoHistory, font: pygame.font.Font):
        self.history = history
        self.font = font
        self.min_vals: list[float] = []
        self.ranges: list[float] = []
        servo_count = len(history.servo_names)
        if servo_count > 0:
            for servo_idx in range(servo_count):
                column = [row[servo_idx] for row in history.values]
                min_val = min(column)
                max_val = max(column)
                rng = max_val - min_val
                self.min_vals.append(min_val)
                self.ranges.append(rng if rng > 0 else 1.0)
        self.norm_values = [
            [
                (row[servo_idx] - self.min_vals[servo_idx]) / self.ranges[servo_idx]
                if self.ranges[servo_idx] > 0
                else 0.5
                for servo_idx in range(len(history.servo_names))
            ]
            for row in history.values
        ]

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, highlight_idx: int) -> None:
        pygame.draw.rect(surface, self.CHART_BG, rect, border_radius=8)
        pygame.draw.rect(surface, self.CHART_BORDER, rect, width=1, border_radius=8)
        title = self.font.render("Servo trajectories", True, self.LABEL_COLOR)
        surface.blit(title, (rect.left + 8, rect.top + 4))

        servo_count = len(self.history.servo_names)
        if servo_count == 0:
            return

        history_len = len(self.norm_values)
        body_top = rect.top + title.get_height() + 6
        usable_height = rect.height - title.get_height() - 10
        row_height = usable_height / servo_count if servo_count else 0
        graph_left = rect.left + 140
        graph_width = max(12, rect.width - (graph_left - rect.left) - 80)

        for servo_idx, name in enumerate(self.history.servo_names):
            row_top = body_top + servo_idx * row_height
            label_surface = self.font.render(name, True, self.LABEL_COLOR)
            label_y = row_top + row_height / 2 - label_surface.get_height() / 2
            surface.blit(label_surface, (rect.left + 6, label_y))

            graph_top = row_top + 6
            graph_height = max(4, row_height - 12)
            graph_bottom = graph_top + graph_height

            if graph_width > 0 and history_len > 0:
                step = max(1, history_len // max(int(graph_width), 1))
                points: list[tuple[float, float]] = []
                denom = max(1, history_len - 1)
                for idx in range(0, history_len, step):
                    rel = idx / denom
                    x = graph_left + rel * graph_width
                    y = graph_bottom - self.norm_values[idx][servo_idx] * graph_height
                    points.append((x, y))
                if len(points) > 1:
                    pygame.draw.lines(surface, self.LINE_COLOR, False, points, 2)

                clamped_idx = max(0, min(history_len - 1, highlight_idx))
                rel = clamped_idx / denom if denom > 0 else 0.0
                highlight_x = graph_left + rel * graph_width
                highlight_y = graph_bottom - self.norm_values[clamped_idx][servo_idx] * graph_height
                pygame.draw.circle(surface, self.HIGHLIGHT_COLOR, (int(highlight_x), int(highlight_y)), 5)

                actual_value = self.history.values[clamped_idx][servo_idx]
                value_surface = self.font.render(f"{actual_value:.1f}", True, self.VALUE_COLOR)
                value_y = row_top + row_height / 2 - value_surface.get_height() / 2
                value_x = rect.right - value_surface.get_width() - 6
                surface.blit(value_surface, (value_x, value_y))


def load_servo_history(dataset_root: Path) -> ServoHistory | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        LOG.debug("Dataset info not found at %s", info_path)
        return None

    try:
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except Exception as exc:
        LOG.warning("Unable to parse dataset info %s: %s", info_path, exc)
        return None

    features = info.get("features", {})
    action_feature = features.get("action")
    if not action_feature:
        return None

    servo_names = action_feature.get("names") or []
    if not servo_names:
        return None

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        LOG.warning("Dataset data directory missing: %s", data_dir)
        return None

    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        return None

    servo_values: list[list[float]] = []
    timestamps: list[float] = []
    for parquet_path in parquet_files:
        try:
            table = pq.read_table(parquet_path, columns=["action", "timestamp"])
        except Exception as exc:
            LOG.warning("Failed to read %s: %s", parquet_path, exc)
            continue
        actions = table.column("action").to_pylist()
        timestamps.extend(table.column("timestamp").to_pylist())
        for row in actions:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                halo = [float(value) for value in row]
            else:
                halo = [float(row)]
            if len(halo) < len(servo_names):
                halo.extend([0.0] * (len(servo_names) - len(halo)))
            elif len(halo) > len(servo_names):
                halo = halo[: len(servo_names)]
            servo_values.append(halo)

    if not servo_values:
        return None

    fps = float(info.get("fps") or 30.0)
    return ServoHistory(servo_names=list(servo_names), values=servo_values, timestamps=timestamps, fps=fps)


def _clamp_motor_value(motor: str, value: int, bus: MotorsBus) -> int:
    calibration = getattr(bus, "calibration", None) or {}
    motor_calib = calibration.get(motor)
    if motor_calib is not None:
        return max(motor_calib.range_min, min(motor_calib.range_max, value))
    motor_info = bus.motors.get(motor)
    if motor_info is not None:
        resolution_table = getattr(bus, "model_resolution_table", {})
        resolution = resolution_table.get(motor_info.model)
        if resolution:
            return max(0, min(resolution - 1, value))
    return max(0, value)


def _load_snapshot_targets(snapshot_path: Path, bus: MotorsBus | None) -> dict[str, int]:
    if not snapshot_path.exists():
        LOG.warning("Startup snapshot missing: %s", snapshot_path)
        return {}
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        LOG.warning("Unable to parse startup snapshot %s: %s", snapshot_path.name, exc)
        return {}

    motors = payload.get("motors")
    if not isinstance(motors, Mapping):
        return {}

    targets: dict[str, int] = {}
    for name, info in motors.items():
        if not isinstance(name, str) or not isinstance(info, Mapping):
            continue
        value = info.get("saved_position")
        if value is None:
            value = info.get("present_position")
        if value is None:
            continue
        try:
            target_value = int(round(float(value)))
        except Exception:
            continue
        if bus is not None:
            target_value = _clamp_motor_value(name, target_value, bus)
        targets[name] = target_value
    return targets


def _sleep_with_event_pump(duration: float) -> bool:
    if duration <= 0:
        return True
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        remaining = end_time - time.perf_counter()
        time.sleep(min(0.02, max(0.0, remaining)))
    return True


def _animate_pose_transition(robot_bus: MotorsBus, targets: dict[str, int], transition_time: float) -> bool:
    if transition_time <= 0 or not targets:
        try:
            robot_bus.sync_write("Goal_Position", targets, normalize=False)
        except Exception as exc:
            LOG.warning("Startup pose write failed: %s", exc)
        return True

    try:
        current = robot_bus.sync_read("Present_Position", normalize=False)
    except Exception as exc:
        LOG.warning("Unable to read motors before startup motion: %s", exc)
        return True

    names = [name for name in targets if name in current]
    if not names:
        return True

    steps = max(2, int(transition_time * 40))
    step_duration = transition_time / steps
    for idx in range(steps):
        alpha = idx / (steps - 1) if steps > 1 else 1.0
        pose: dict[str, int] = {}
        for name in names:
            start_val = float(current[name])
            target_val = targets[name]
            interpolated = int(round(start_val + (target_val - start_val) * alpha))
            pose[name] = _clamp_motor_value(name, interpolated, robot_bus)
        try:
            robot_bus.sync_write("Goal_Position", pose, normalize=False)
        except Exception as exc:
            LOG.warning("Startup pose transition write failed: %s", exc)
            return True
        if idx < steps - 1:
            if not _sleep_with_event_pump(step_duration):
                return False
    return True


def _render_startup_message(
    screen: pygame.Surface,
    font: pygame.font.Font,
    window_size: tuple[int, int],
    message: str,
) -> None:
    screen.fill((10, 10, 10))
    text = font.render(message, True, (240, 240, 240))
    screen.blit(text, text.get_rect(center=(window_size[0] / 2, window_size[1] / 2)))
    pygame.display.flip()


def _choose_wink_servo(robot_bus: MotorsBus) -> str | None:
    for candidate in WINK_SERVO_CANDIDATES:
        if candidate in robot_bus.motors:
            return candidate
    return next(iter(robot_bus.motors.keys()), None)


def _perform_wink_animation(robot_bus: MotorsBus) -> bool:
    servo_name = _choose_wink_servo(robot_bus)
    if servo_name is None:
        return True
    try:
        present = robot_bus.read("Present_Position", servo_name, normalize=False)
    except Exception as exc:
        LOG.warning("Unable to read wink servo %s: %s", servo_name, exc)
        return True

    try:
        base_position = int(round(float(present)))
    except Exception:
        base_position = int(present)

    loops = 3
    for _ in range(loops):
        left_target = _clamp_motor_value(servo_name, base_position - WINK_AMOUNT, robot_bus)
        try:
            robot_bus.sync_write("Goal_Position", {servo_name: left_target}, normalize=False)
        except Exception as exc:
            LOG.warning("Wink animation write failed: %s", exc)
            return True
        if not _sleep_with_event_pump(WINK_LEFT_DURATION):
            return False

        right_target = _clamp_motor_value(servo_name, base_position + WINK_AMOUNT, robot_bus)
        try:
            robot_bus.sync_write("Goal_Position", {servo_name: right_target}, normalize=False)
        except Exception as exc:
            LOG.warning("Wink animation write failed: %s", exc)
            return True
        if not _sleep_with_event_pump(WINK_RIGHT_DURATION):
            return False

    reset_target = _clamp_motor_value(servo_name, base_position, robot_bus)
    try:
        robot_bus.sync_write("Goal_Position", {servo_name: reset_target}, normalize=False)
    except Exception as exc:
        LOG.warning("Failed to reset wink servo: %s", exc)
        return True

    return _sleep_with_event_pump(WINK_RESET_DURATION)


def _play_snapshot_step(
    robot_bus: MotorsBus | None,
    snapshot_path: Path,
    duration: float,
) -> bool:
    duration = max(0.0, duration)
    transition_time = min(duration, 0.6)
    hold_time = max(0.0, duration - transition_time)
    targets = _load_snapshot_targets(snapshot_path, robot_bus)
    if targets and robot_bus and getattr(robot_bus, "is_connected", False):
        if transition_time > 0 and not _animate_pose_transition(robot_bus, targets, transition_time):
            return False
        if hold_time > 0 and not _sleep_with_event_pump(hold_time):
            return False
        return True
    return _sleep_with_event_pump(duration)


def run_startup_sequence(
    robot_bus: MotorsBus | None,
    screen: pygame.Surface,
    font: pygame.font.Font,
    window_size: tuple[int, int],
) -> bool:
    snapshot_dir = CALIBRATION_SNAPSHOT_DIR
    if not snapshot_dir.exists():
        LOG.info("Calibration snapshots directory is missing; startup animation will wait only.")
    for name, duration in STARTUP_SEQUENCE:
        message = f"Startup pose: {name}"
        _render_startup_message(screen, font, window_size, message)

        snapshot_path = snapshot_dir / name
        if not _play_snapshot_step(robot_bus, snapshot_path, duration):
            return False

    _render_startup_message(screen, font, window_size, "Winking follower arm")
    if robot_bus and getattr(robot_bus, "is_connected", False):
        if not _perform_wink_animation(robot_bus):
            return False
    else:
        if not _sleep_with_event_pump(WINK_LEFT_DURATION + WINK_RIGHT_DURATION + WINK_RESET_DURATION):
            return False

    _render_startup_message(screen, font, window_size, "Startup complete; starting music...")
    return _sleep_with_event_pump(0.5)


def main() -> None:
    configure_logging()
    args = parse_args()

    output_dir = args.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_run = find_latest_run(output_dir)
    run_number = args.run_number if args.run_number is not None else upstream_run + 1
    LOG.info("Starting run %03d (outputs=%s)", run_number, output_dir)
    remove_run_files(output_dir, run_number)

    total_duration = librosa.get_duration(filename=str(args.audio))

    dataset_repo_id = f"eerwitt/xvla-beats-run{run_number:03d}"
    dataset_root = args.dataset_root or (HF_LEROBOT_HOME / dataset_repo_id)
    servo_history = load_servo_history(dataset_root)
    servo_chart_height = 0
    if servo_history:
        servo_chart_height = max(140, len(servo_history.servo_names) * 32 + 20)

    robot_bus: MotorsBus | None = None
    robot = None
    if args.robot_enable:
        register_third_party_plugins()
        robot_cfg = so101_follower.SO101FollowerConfig(
            port=args.robot_port or DEFAULT_ROBOT_PORT,
            id=args.robot_id or DEFAULT_ROBOT_ID,
        )
        robot = make_robot_from_config(robot_cfg)
        LOG.info(
            "Connecting to follower %s (calibrate=%s)", robot_cfg.id, args.robot_calibrate
        )
        robot.connect(calibrate=args.robot_calibrate)
        try:
            robot_bus = robot.bus  # type: ignore[assignment]
        except AttributeError as err:
            robot.disconnect()
            raise RuntimeError("Robot instance must expose `.bus` for servo sync.") from err

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(str(args.audio))

    image_size = tuple(args.image_size)
    padding = 20
    timeline_height = 80
    window_width = image_size[0] * 2 + padding * 3
    window_height = image_size[1] + padding * 4 + timeline_height + servo_chart_height
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Run {run_number:03d} Beats")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 20)
    servo_chart = ServoChart(servo_history, font) if servo_history else None

    play_button = pygame.Rect(padding, window_height - padding - 40, 80, 36)
    pause_button = pygame.Rect(padding + 100, window_height - padding - 40, 80, 36)
    restart_button = pygame.Rect(padding + 200, window_height - padding - 40, 80, 36)
    slider_rect = pygame.Rect(
        padding,
        window_height - padding - 90,
        window_width - padding * 2,
        20,
    )

    placeholder = create_placeholder(image_size, "Waiting for frames...", font)
    placeholder_pair = (placeholder, placeholder)

    frame_buffers: Dict[int, Tuple[bytes, bytes]] = {}
    buffer_lock = threading.Lock()
    frame_surfaces: Dict[int, Tuple[pygame.Surface, pygame.Surface]] = {}
    stop_event = threading.Event()

    producer = FrameProducer(
        audio_path=args.audio,
        run_number=run_number,
        output_dir=output_dir,
        window_duration=args.window_duration,
        hop_duration=args.hop_duration,
        sr=int(args.sr),
        image_size=image_size,
        frame_buffers=frame_buffers,
        buffer_lock=buffer_lock,
        stop_event=stop_event,
    )
    producer.start()

    playback_time = 0.0
    playback_start_ms = 0
    playing = False
    slider_dragging = False

    def start_playback(start_time: float) -> None:
        nonlocal playing, playback_time, playback_start_ms
        seek_time = max(0.0, min(total_duration, start_time))
        pygame.mixer.music.stop()
        pygame.mixer.music.play(loops=0, start=seek_time)
        playback_time = seek_time
        playback_start_ms = pygame.time.get_ticks()
        playing = True

    running = True
    startup_success = run_startup_sequence(
        robot_bus,
        screen,
        font,
        (window_width, window_height),
    )
    if startup_success:
        start_playback(0.0)
    else:
        running = False

    try:
        while running:
            now = pygame.time.get_ticks()
            with buffer_lock:
                pending = {
                    idx: frame_buffers[idx]
                    for idx in frame_buffers.keys()
                    if idx not in frame_surfaces
                }

            for frame_idx, (mel_bytes, chroma_bytes) in pending.items():
                mel_surface = pygame.image.frombuffer(mel_bytes, image_size, "RGB").convert()
                chroma_surface = pygame.image.frombuffer(chroma_bytes, image_size, "RGB").convert()
                frame_surfaces[frame_idx] = (mel_surface, chroma_surface)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if slider_rect.collidepoint(event.pos) and slider_rect.width > 0:
                        slider_dragging = True
                        fraction = max(0.0, min(1.0, (event.pos[0] - slider_rect.left) / slider_rect.width))
                        playback_time = fraction * total_duration
                        playing = False
                        pygame.mixer.music.stop()
                    elif play_button.collidepoint(event.pos):
                        start_playback(playback_time)
                    elif pause_button.collidepoint(event.pos):
                        if playing:
                            playing = False
                            playback_time = max(
                                0.0,
                                min(
                                    total_duration,
                                    playback_time + (pygame.time.get_ticks() - playback_start_ms) / 1000.0,
                                ),
                            )
                            pygame.mixer.music.stop()
                        else:
                            start_playback(playback_time)
                    elif restart_button.collidepoint(event.pos):
                        start_playback(0.0)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    slider_dragging = False
                elif event.type == pygame.MOUSEMOTION and slider_dragging and slider_rect.width > 0:
                    fraction = max(0.0, min(1.0, (event.pos[0] - slider_rect.left) / slider_rect.width))
                    playback_time = fraction * total_duration
                    playing = False
                    pygame.mixer.music.stop()

            if playing:
                elapsed = (pygame.time.get_ticks() - playback_start_ms) / 1000.0
                current_time = playback_time + elapsed
                if current_time >= total_duration:
                    current_time = total_duration
                    playing = False
                    pygame.mixer.music.stop()
                    playback_time = total_duration
            else:
                current_time = playback_time

            servo_frame_index = (
                servo_history.frame_for_time(current_time) if servo_history else 0
            )
            if robot_bus and servo_history and not slider_dragging:
                servo_values = servo_history.values[servo_frame_index]
                action_payload = {
                    name: int(round(value)) for name, value in zip(servo_history.servo_names, servo_values)
                }
                try:
                    robot_bus.sync_write("Goal_Position", action_payload, normalize=False)
                except Exception as exc:
                    LOG.debug("Servo replay write failed: %s", exc)

            frame_index = int(current_time / args.hop_duration) if args.hop_duration > 0 else 0
            display_surfaces = placeholder_pair
            if frame_surfaces:
                sorted_frames = sorted(frame_surfaces.keys())
                chosen_frame = sorted_frames[0]
                for candidate in sorted_frames:
                    if candidate <= frame_index:
                        chosen_frame = candidate
                    else:
                        break
                display_surfaces = frame_surfaces.get(chosen_frame, placeholder_pair)

            screen.fill((10, 10, 10))

            status_surface = font.render(
                f"Run {run_number:03d} – Generated {producer.frames_generated} frames",
                True,
                (230, 230, 230),
            )
            screen.blit(status_surface, (padding, padding / 2))

            screen.blit(display_surfaces[0], (padding, padding + 20))
            screen.blit(display_surfaces[1], (padding * 2 + image_size[0], padding + 20))

            if servo_chart and servo_chart_height > 0:
                chart_rect = pygame.Rect(
                    padding,
                    padding + 20 + image_size[1] + 10,
                    window_width - padding * 2,
                    servo_chart_height,
                )
                servo_chart.draw(screen, chart_rect, servo_frame_index)

            pygame.draw.rect(screen, (60, 60, 60), slider_rect, border_radius=6)
            pygame.draw.rect(screen, (120, 120, 120), slider_rect.inflate(-4, -8), border_radius=4)
            if total_duration > 0:
                knob_x = slider_rect.left + max(0.0, min(1.0, current_time / total_duration)) * slider_rect.width
            else:
                knob_x = slider_rect.left
            pygame.draw.circle(screen, (200, 80, 80), (int(knob_x), slider_rect.centery), 10)

            time_surface = small_font.render(
                f"{format_time(current_time)} / {format_time(total_duration)}", True, (220, 220, 220)
            )
            screen.blit(time_surface, (slider_rect.left, slider_rect.top - 22))

            pygame.draw.rect(screen, (40, 120, 40), play_button, border_radius=6)
            pygame.draw.rect(screen, (120, 120, 40), pause_button, border_radius=6)
            pygame.draw.rect(screen, (120, 40, 40), restart_button, border_radius=6)
            for rect, label in [
                (play_button, "Play"),
                (pause_button, "Pause"),
                (restart_button, "Restart"),
            ]:
                label_surface = font.render(label, True, (255, 255, 255))
                screen.blit(label_surface, label_surface.get_rect(center=rect.center))

            pygame.display.flip()
            clock.tick(60)
    finally:
        stop_event.set()
        producer.join(timeout=1.0)
        pygame.mixer.music.stop()
        pygame.quit()
        if robot is not None:
            LOG.info("Disconnecting follower")
            robot.disconnect()


if __name__ == "__main__":
    main()
