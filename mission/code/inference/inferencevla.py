#!/usr/bin/env python3
"""Standalone SmolVLA/RTC launcher.

This script mirrors the worker configuration but does not import any files from
the Hackathon repository. Move it anywhere, install the LeRobot stack plus the
required robot drivers, and run it with the same flags you would normally pass
to `uv run worker`.

To fine-tune SmolVLA for your hardware/tasks:
1. Record demonstrations with `lerobot-record` (matching camera names + state keys) into a Hugging Face dataset.
2. Run `lerobot/scripts/train.py --policy.path=lerobot/smolvla_base --dataset.repo_id=<you>/<dataset> ...` following the VLA recipe.
3. Point `--policy.path` in this script to the fine-tuned checkpoint so inference uses your new weights.

If behavior still looks off, recalibrate the SO-101 arm (run the vendor’s calibration + `robot.reset_position` routines) so the home pose and joint limits match the policy’s assumptions before collecting data or running inference.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from collections import deque
from threading import Event, Lock, Thread
from typing import Optional

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import Robot, RobotConfig, so101_follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging
from torch import Tensor

# Required so that custom robot configs (bi_so101_follower, etc.) are registered
register_third_party_devices()

DEFAULT_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"


def _make_default_cameras() -> dict[str, OpenCVCameraConfig]:
    """Default SO-101 layout; policy rename map will align names."""
    return {
        "top": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30),
        "side": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    """Thread-safe wrapper so the actor/action threads can share the robot."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor) -> None:
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features

    def reset_position(self) -> None:
        with self.lock:
            if hasattr(self.robot, "reset_position"):
                self.robot.reset_position()
            else:
                logger.warning("Robot does not expose reset_position(); skipping reset.")


@dataclass
class StandaloneRTCDemoConfig(HubMixin):
    """Configuration schema compatible with LeRobot's draccus parser."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    duration: float = 60.0
    fps: float = 5.0
    device: Optional[str] = None
    action_queue_size_to_get_new_actions: int = 60
    task: str = "move to the right"
    log_every_n_chunks: int = 5
    log_image_stats: bool = True
    log_action_deltas: bool = True
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_disable_cudagraphs: bool = True
    stagnation_window: int = 10
    stagnation_threshold: float = 0.01
    return_to_start_on_stall: bool = True

    def __post_init__(self) -> None:
        if self.policy is None:
            policy_path = parser.get_path_arg("policy") or DEFAULT_POLICY_PATH
            if not policy_path:
                raise ValueError("--policy.path must be provided")
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

        if self.robot is None:
            self.robot = so101_follower.SO101FollowerConfig(
                port=DEFAULT_ROBOT_PORT,
                id=DEFAULT_ROBOT_ID,
                cameras=_make_default_cameras(),
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        # Allows draccus to load `.path=` arguments directly.
        return ["policy"]


def _apply_torch_compile(policy, cfg: StandaloneRTCDemoConfig):
    if policy.type in ("pi05", "pi0"):
        return policy

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available; skipping compilation")
        return policy

    options = {}
    if cfg.torch_compile_disable_cudagraphs:
        options["triton.cudagraphs"] = False
    policy.predict_action_chunk = torch.compile(  # type: ignore[attr-defined]
        policy.predict_action_chunk,
        backend=cfg.torch_compile_backend,
        mode=cfg.torch_compile_mode,
        options=options or None,
    )
    logger.info("Enabled torch.compile on predict_action_chunk")
    return policy


def _infer_camera_rename_map(policy_cfg: PreTrainedConfig, robot: RobotWrapper) -> dict[str, str]:
    """Infer mapping from robot camera names to policy-expected names."""
    input_features = getattr(policy_cfg, "input_features", {}) or {}
    expected = [
        key.rsplit(".", 1)[-1]
        for key in input_features
        if key.startswith("observation.images.")
    ]
    available = []
    if hasattr(robot.robot, "cameras"):
        available = list(robot.robot.cameras.keys())

    if not expected or not available:
        return {}

    if len(expected) != len(available):
        logger.warning(
            "Camera count mismatch (policy expects %d, robot provides %d). Mapping by order.",
            len(expected),
            len(available),
        )

    rename_map: dict[str, str] = {}
    for old_name, new_name in zip(available, expected):
        if old_name != new_name:
            rename_map[old_name] = new_name

    if rename_map:
        logger.info("Applying camera rename map: %s", rename_map)
    return rename_map


def _request_action_chunks(
    *,
    policy,
    policy_cfg: PreTrainedConfig,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: StandaloneRTCDemoConfig,
    task_override: str,
):
    latency_tracker = LatencyTracker()
    fps = cfg.fps
    time_per_chunk = 1.0 / fps

    dataset_features = hw_to_dataset_features(
        robot.observation_features(), "observation"
    )
    policy_device = policy.config.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {
                "device": getattr(policy_cfg, "device", None) or cfg.device
            }
        },
    )
    camera_rename_map = _infer_camera_rename_map(policy_cfg, robot)

    seen_chunks = 0

    logger.info(
        "Policy input features: %s | output features: %s",
        getattr(policy_cfg, "input_features", "unknown"),
        getattr(policy_cfg, "output_features", "unknown"),
    )
    logger.info("Robot observation features: %s", robot.observation_features())
    logger.info("Robot action features: %s", robot.action_features())

    expected_inputs = set(getattr(policy_cfg, "input_features", {}).keys())
    warned_missing_inputs = False

    get_actions_threshold = (
        cfg.action_queue_size_to_get_new_actions if cfg.rtc.enabled else 0
    )

    while not shutdown_event.is_set():
        if action_queue.qsize() > get_actions_threshold:
            time.sleep(0.05)
            continue

        current_time = time.perf_counter()
        action_index_before_inference = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over()

        inference_latency = latency_tracker.max()
        inference_delay = math.ceil(inference_latency / time_per_chunk)

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        obs_with_policy_features = build_dataset_frame(
            dataset_features, obs_processed, prefix="observation"
        )

        for name, value in obs_with_policy_features.items():
            tensor = torch.from_numpy(value)
            if "image" in name:
                tensor = tensor.float() / 255.0
                tensor = tensor.permute(2, 0, 1).contiguous()
            obs_with_policy_features[name] = tensor.unsqueeze(0).to(policy_device)

        if camera_rename_map:
            for old_name, new_name in camera_rename_map.items():
                old_key = f"observation.images.{old_name}"
                new_key = f"observation.images.{new_name}"
                if old_key in obs_with_policy_features and new_key not in obs_with_policy_features:
                    obs_with_policy_features[new_key] = obs_with_policy_features.pop(old_key)

        task_str = task_override or cfg.task or ""
        obs_with_policy_features["task"] = [task_str]
        obs_with_policy_features["robot_type"] = (
            robot.robot.name if hasattr(robot.robot, "name") else ""
        )

        if expected_inputs and not warned_missing_inputs:
            missing = sorted(expected_inputs - set(obs_with_policy_features.keys()))
            if missing:
                logger.warning("Missing policy inputs: %s", missing)
            warned_missing_inputs = True

        if seen_chunks % cfg.log_every_n_chunks == 0:
            logger.info("Observation keys: %s", list(obs_with_policy_features.keys()))
            if cfg.log_image_stats:
                for name, tensor in obs_with_policy_features.items():
                    if "images" in name:
                        img = tensor.float()
                        stats = (float(img.min()), float(img.max()), float(img.mean()))
                        logger.info("Image %s stats min=%.3f max=%.3f mean=%.3f", name, *stats)
            state_tensor = obs_with_policy_features.get("observation.state")
            if state_tensor is not None:
                state_vals = state_tensor.flatten().tolist()
                logger.info("Observation.state values: %s", [round(v, 4) for v in state_vals])
        preprocessed_obs = preprocessor(obs_with_policy_features)
        actions = policy.predict_action_chunk(
            preprocessed_obs,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_actions,
        )

        original_actions = actions.squeeze(0).clone()
        postprocessed_actions = postprocessor(actions).squeeze(0)
        if seen_chunks % cfg.log_every_n_chunks == 0:
            logger.info(
                "Chunk %d raw action stats: mean=%.4f std=%.4f | postprocessed mean=%.4f std=%.4f",
                seen_chunks,
                float(original_actions.mean()),
                float(original_actions.std()),
                float(postprocessed_actions.mean()),
                float(postprocessed_actions.std()),
            )

        new_latency = time.perf_counter() - current_time
        latency_tracker.add(new_latency)
        new_delay = math.ceil(new_latency / time_per_chunk)

        if (
            cfg.action_queue_size_to_get_new_actions
            < cfg.rtc.execution_horizon + new_delay
        ):
            logger.warning(
                "action_queue_size_to_get_new_actions is too small for the current delay"
            )

        action_queue.merge(
            original_actions, postprocessed_actions, new_delay, action_index_before_inference
        )
        seen_chunks += 1

    logger.info("Action-chunk requester stopped")


def _actor_control(
    *,
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: StandaloneRTCDemoConfig,
):
    action_interval = 1.0 / cfg.fps
    processed_actions = 0
    prev_action_dict = None
    recent_deltas = deque(maxlen=cfg.stagnation_window)
    while not shutdown_event.is_set():
        start = time.perf_counter()
        action = action_queue.get()
        if action is not None:
            action = action.cpu()
            action_dict = {
                key: action[i].item()
                for i, key in enumerate(robot.action_features())
            }
            processed = robot_action_processor((action_dict, None))
            robot.send_action(processed)
            if prev_action_dict is not None:
                delta = sum(
                    abs(action_dict[key] - prev_action_dict.get(key, 0.0))
                    for key in action_dict
                ) / max(1, len(action_dict))
                recent_deltas.append(delta)
                if (
                    cfg.return_to_start_on_stall
                    and len(recent_deltas) == cfg.stagnation_window
                    and all(d < cfg.stagnation_threshold for d in recent_deltas)
                ):
                    avg_delta = sum(recent_deltas) / len(recent_deltas)
                    logger.warning(
                        "Detected stagnation (avg delta %.5f over last %d actions). Returning to start.",
                        avg_delta,
                        cfg.stagnation_window,
                    )
                    robot.reset_position()
                    shutdown_event.set()
                    break

            if processed_actions % cfg.log_every_n_chunks == 0:
                logger.info("Actor sent action dict: %s", action_dict)
                if cfg.log_action_deltas and prev_action_dict:
                    deltas = {
                        key: action_dict[key] - prev_action_dict.get(key, 0.0)
                        for key in action_dict
                    }
                    logger.info("Action delta since last logged step: %s", deltas)
                prev_action_dict = action_dict.copy()
            processed_actions += 1

        dt = time.perf_counter() - start
        time.sleep(max(0.0, (action_interval - dt) - 0.001))

    logger.info("Actor thread stopped")


def _load_policy(cfg: StandaloneRTCDemoConfig):
    policy_cfg = cfg.policy
    assert policy_cfg is not None

    policy_class = get_policy_class(policy_cfg.type)
    config = PreTrainedConfig.from_pretrained(policy_cfg.pretrained_path)

    if policy_cfg.type == "smolvla":
        config.input_features = policy_cfg.input_features
        config.output_features = policy_cfg.output_features

    if policy_cfg.type in ("pi05", "pi0"):
        config.compile_model = cfg.use_torch_compile

    policy = policy_class.from_pretrained(policy_cfg.pretrained_path, config=config)
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()

    device = cfg.device or getattr(policy_cfg, "device", None) or "cpu"
    policy = policy.to(device)
    policy.eval()

    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)

    return policy, policy_cfg


def run(cfg: StandaloneRTCDemoConfig) -> None:
    init_logging()
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    policy, policy_cfg = _load_policy(cfg)

    logger.info("Initializing robot: %s", cfg.robot.type)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_wrapper = RobotWrapper(robot)

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    action_queue = ActionQueue(cfg.rtc)

    requester = Thread(
        target=_request_action_chunks,
        kwargs=dict(
            policy=policy,
            policy_cfg=policy_cfg,
            robot=robot_wrapper,
            robot_observation_processor=robot_observation_processor,
            action_queue=action_queue,
            shutdown_event=shutdown_event,
            cfg=cfg,
            task_override=cfg.task,
        ),
        daemon=True,
        name="GetActions",
    )
    requester.start()

    actor = Thread(
        target=_actor_control,
        kwargs=dict(
            robot=robot_wrapper,
            robot_action_processor=robot_action_processor,
            action_queue=action_queue,
            shutdown_event=shutdown_event,
            cfg=cfg,
        ),
        daemon=True,
        name="Actor",
    )
    actor.start()

    logger.info("Running task '%s' for %.1f seconds", cfg.task, cfg.duration)
    start_time = time.time()
    try:
        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            time.sleep(1.0)
            if int(time.time() - start_time) % 5 == 0:
                logger.info("Action queue size: %d", action_queue.qsize())
    finally:
        shutdown_event.set()
        requester.join(timeout=5.0)
        actor.join(timeout=5.0)
        robot.disconnect()
        logger.info("Robot disconnected")
