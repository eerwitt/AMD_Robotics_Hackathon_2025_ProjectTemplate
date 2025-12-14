#!/usr/bin/env python3
"""Smoothly guide the SO-101 follower back to its calibration midpoint."""

import logging
import time
from dataclasses import dataclass
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.robots import Robot, RobotConfig, so101_follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep

try:
    # Reuse the same defaults as the ACT/SmolVLA demo scripts.
    from inference.inferenceact import DEFAULT_ROBOT_ID, DEFAULT_ROBOT_PORT
except Exception:  # pragma: no cover - only triggered when ACT deps are missing.
    try:
        from inference.inferencevla import DEFAULT_ROBOT_ID, DEFAULT_ROBOT_PORT
    except Exception:  # pragma: no cover - best-effort fallback when both fail.
        DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
        DEFAULT_ROBOT_PORT = "/dev/ttyACM0"


def move_to_calibration_position_smoothly(
    robot: Robot,
    duration_seconds: float = 30.0,
    target_pose: str = "midpoint",
) -> None:
    """Interpolate every joint from its current position to the requested calibration-derived pose."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0 to build an interpolation trajectory.")

    target_pose = (target_pose or "midpoint").lower()
    if target_pose not in {"midpoint", "origin"}:
        raise ValueError("target_pose must be 'midpoint' or 'origin'.")

    if not hasattr(robot, "bus"):
        raise AttributeError("Robot instance must expose a Dynamixel bus via `.bus`.")

    bus = robot.bus  # type: ignore[attr-defined]
    current_calibration = bus.read_calibration()
    if not current_calibration:
        raise RuntimeError("Robot returned an empty calibration map; aborting reset.")

    goal_positions: dict[str, float] = {}
    logging.info("Calibration map (%d joints):", len(current_calibration))
    for motor_name, calibration in current_calibration.items():
        midpoint = (calibration.range_min + calibration.range_max) // 2
        origin = calibration.range_min
        goal = midpoint if target_pose == "midpoint" else origin
        goal_positions[motor_name] = float(goal)
        logging.info(
            "  %s: range_min=%s range_max=%s midpoint=%s origin=%s",
            motor_name,
            calibration.range_min,
            calibration.range_max,
            midpoint,
            origin,
        )
    logging.info("Selected target pose: %s", target_pose)

    current_position_dict = bus.sync_read("Present_Position", normalize=False)
    motor_names = list(current_position_dict.keys())
    logging.info("Current joint positions:")
    for name in motor_names:
        logging.info("  %s: %s", name, current_position_dict[name])

    target_dict = {name: goal_positions.get(name) for name in motor_names}
    logging.info("Target joint positions computed from calibration %s pose:\n%s", target_pose, pformat(target_dict))

    current_position = np.array(
        [current_position_dict[name] for name in motor_names],
        dtype=np.float32,
    )
    target_position = np.array(
        [goal_positions[name] for name in motor_names],
        dtype=np.float32,
    )

    num_steps = max(1, int(duration_seconds * 50))
    step_duration = duration_seconds / num_steps
    logging.info(
        "Moving %d joints toward calibration %s pose over %.2f seconds (%d steps).",
        len(motor_names),
        target_pose,
        duration_seconds,
        num_steps,
    )
    diff_positions = {
        name: goal_positions[name] - current_position_dict[name] for name in motor_names
    }

    MAX_DELTA = 200.0
    logging.info(
        "Applying MAX_DELTA=%.1f ticks; joints beyond this threshold will stay put.",
        MAX_DELTA,
    )
    filtered_targets = target_position.copy()
    skipped_joints: list[str] = []
    for idx, name in enumerate(motor_names):
        delta = diff_positions[name]
        if abs(delta) > MAX_DELTA:
            logging.warning(
                "Skipping %s because |delta|=%.1f exceeds MAX_DELTA=%.1f",
                name,
                delta,
                MAX_DELTA,
            )
            filtered_targets[idx] = current_position[idx]
            skipped_joints.append(name)

    if skipped_joints:
        logging.warning("The following joints will stay at their current positions: %s", skipped_joints)

    trajectory = np.linspace(current_position, filtered_targets, num_steps)
    clipped_target_dict = {
        name: value for name, value in zip(motor_names, filtered_targets.tolist(), strict=False)
    }
    logging.info("Per-joint deltas to %s (after clipping check):\n%s", target_pose, pformat(diff_positions))
    print("Current positions:", pformat(current_position_dict), flush=True)
    print(f"Target {target_pose}:", pformat(target_dict), flush=True)
    print(f"Clipped target {target_pose}:", pformat(clipped_target_dict), flush=True)
    print(f"Delta to {target_pose}:", pformat(diff_positions), flush=True)
    logging.info("Sleeping 10 seconds before executing trajectory so you can interrupt if needed.")
    time.sleep(10)

    for pose in trajectory:
        action_dict = {
            name: int(round(value))
            for name, value in zip(motor_names, pose.tolist(), strict=False)
        }
        bus.sync_write("Goal_Position", action_dict, normalize=False)
        precise_sleep(step_duration)


@dataclass
class ResetRobotConfig:
    """CLI schema so we can reuse LeRobot's parser overrides."""

    robot: RobotConfig | None = None
    duration_seconds: float = 30.0
    calibrate_on_connect: bool = False
    target_pose: str = "midpoint"

    def __post_init__(self) -> None:
        if self.robot is None:
            self.robot = so101_follower.SO101FollowerConfig(
                port=DEFAULT_ROBOT_PORT,
                id=DEFAULT_ROBOT_ID,
            )


@parser.wrap()
def main(cfg: ResetRobotConfig) -> None:
    """Connect to the robot, execute the reset trajectory, and disconnect."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    register_third_party_devices()

    robot = make_robot_from_config(cfg.robot)

    robot_name = getattr(cfg.robot, "id", "robot")
    logging.info("Connecting to %s (calibrate_on_connect=%s)", robot_name, cfg.calibrate_on_connect)
    robot.connect(calibrate=cfg.calibrate_on_connect)

    try:
        move_to_calibration_position_smoothly(
            robot,
            duration_seconds=cfg.duration_seconds,
            target_pose=cfg.target_pose,
        )
    finally:
        logging.info("Disconnecting from %s", robot_name)
        robot.disconnect()


if __name__ == "__main__":
    main()
