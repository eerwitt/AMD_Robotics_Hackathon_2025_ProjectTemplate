"""Smolagents tool for moving the SO-101 follower to saved calibration poses."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from smolagents import Tool

from lerobot.motors import MotorCalibration
from lerobot.robots import so101_follower
from lerobot.robots.utils import make_robot_from_config

logger = logging.getLogger(__name__)

# TODO: investigate bugs in smolagents interpreter for the /dev mapping
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
DEFAULT_SNAPSHOT_DIR = Path("outputs/calibration_snapshots")
DEFAULT_TRANSITION_DURATION = 6.0
POSE_TOON_OPTIONS = (
    "pose_descriptions.toon",
    "positions.toon",
    "pose_context.toon",
)


class SmolPoseMoverTool(Tool):
    """Tool that moves the follower arm robot to different positions, can be used to move front camera to see more of the environment available."""

    name = "move_to_preset_pose"
    output_type = "string"
    description = (
        "Move the SO-101 follower to a stored calibration pose. "
        "Pose names are surfaced via metadata derived from the calibration snapshot files and describe useful camera/arm configurations."
    )
    inputs = {
        "pose_name": {
            "type": "string",
            "description": "Name of the saved pose (derived from snapshot filenames).",
            "nullable": True,
        },
    }
    outputs = {
        "pose_name": {
            "type": "string",
            "description": "The pose that was moved to (or 'home').",
        },
        "success": {
            "type": "boolean",
            "description": "Whether the motion completed without error.",
        },
        "details": {
            "type": "string",
            "description": "Summary or error details.",
        },
    }

    def __init__(self) -> None:
        super().__init__()
        DEFAULT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = DEFAULT_SNAPSHOT_DIR
        self.pose_names: list[str] = []
        self.pose_context_toon: str | None = None
        self.metadata: dict[str, Any] = {}
        self._refresh_metadata()

    def forward(self, pose_name: str | None = None) -> dict[str, Any]:
        request_pose = (pose_name or "").strip()
        if not request_pose:
            return {
                "pose_name": "",
                "success": False,
                "details": "Pose name must be provided. Known poses: "
                + ", ".join(self.pose_names),
            }
        self._refresh_metadata()
        if request_pose not in self.pose_names:
            return {
                "pose_name": request_pose,
                "success": False,
                "details": f"Pose '{request_pose}' is not available. Known poses: {', '.join(self.pose_names)}.",
            }

        duration = max(0.1, DEFAULT_TRANSITION_DURATION)

        cfg = so101_follower.SO101FollowerConfig(port=DEFAULT_ROBOT_PORT, id=DEFAULT_ROBOT_ID)
        robot = make_robot_from_config(cfg)
        logger.info("Connecting follower for pose '%s' using port=%s id=%s", request_pose, cfg.port, cfg.id)
        robot.connect(calibrate=False)
        details = ""
        try:
            bus = robot.bus  # type: ignore[assignment]
            logger.info(
                "Follower bus motors [%d]: %s",
                len(bus.motors),
                ", ".join(sorted(bus.motors.keys())),
            )
            targets = self._targets_for_pose(request_pose, bus)
            if not targets:
                details = f"No motor targets for pose '{request_pose}'."
                return {"pose_name": request_pose, "success": False, "details": details}

            self._execute_transition(bus, targets, duration)
            details = f"Moved to '{request_pose}' over {duration:.1f}s using {len(targets)} joints."
            return {"pose_name": request_pose, "success": True, "details": details}
        except Exception as exc:  # pragma: no cover - hardware integration
            logger.exception("Failed to execute pose '%s'", request_pose)
            return {"pose_name": request_pose, "success": False, "details": str(exc)}
        finally:
            robot.disconnect()

    def _refresh_metadata(self) -> None:
        snapshots = sorted(self.snapshot_dir.glob("*.json"))
        names = sorted(path.stem for path in snapshots)
        self.pose_names = names
        self.pose_context_toon = self._read_optional_toon()
        self.metadata = {
            "pose_names": names,
            "pose_descriptions_toon": self.pose_context_toon or "",
        }

    def _read_optional_toon(self) -> str | None:
        for filename in POSE_TOON_OPTIONS:
            path = self.snapshot_dir / filename
            if path.is_file():
                try:
                    return path.read_text()
                except Exception:
                    logger.warning("Failed to read %s", path)
        return None

    def _targets_for_pose(self, pose: str, bus: so101_follower.SO101FollowerConfig | Any) -> dict[str, int]:
        path = self.snapshot_dir / f"{pose}.json"
        if not path.exists():
            raise FileNotFoundError(f"No snapshot found for pose '{pose}'")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return self._build_pose_targets(payload, bus)

    def _build_pose_targets(self, snapshot_payload: Mapping[str, Any], bus: Any) -> dict[str, int]:
        motors_payload = snapshot_payload.get("motors", {})
        calibration_map: dict[str, MotorCalibration] = bus.read_calibration()
        targets: dict[str, int] = {}
        if not isinstance(motors_payload, Mapping):
            return targets
        for motor_name in bus.motors.keys():
            entry = motors_payload.get(motor_name)
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("saved_position") or entry.get("present_position")
            if value is None:
                continue
            try:
                raw = int(round(float(value)))
            except Exception:
                continue
            targets[motor_name] = self._clamp_motor_value(calibration_map.get(motor_name), bus, motor_name, raw)
        return targets

    @staticmethod
    def _clamp_motor_value(
        calibration: MotorCalibration | None, bus: Any, motor: str, value: int
    ) -> int:
        if calibration is not None:
            return max(calibration.range_min, min(calibration.range_max, value))
        motor_info = bus.motors.get(motor)
        if motor_info:
            resolution = getattr(motor_info, "resolution", None)
            if resolution:
                return max(0, min(resolution - 1, value))
        return value

    def _execute_transition(self, bus: Any, targets: dict[str, int], duration: float) -> None:
        current_positions = bus.sync_read("Present_Position", normalize=False)
        available = [name for name in targets if name in current_positions]
        if not available:
            raise ValueError("No overlapping motors between pose and robot.")
        start = np.array([current_positions[name] for name in available], dtype=np.float32)
        dest = np.array([targets[name] for name in available], dtype=np.float32)
        steps = max(2, int(max(duration, 0.5) * 40))
        step_duration = max(duration / steps, 1e-3)
        for pose in np.linspace(start, dest, steps, dtype=np.float32):
            action = {name: int(round(value)) for name, value in zip(available, pose.tolist(), strict=False)}
            bus.sync_write("Goal_Position", action, normalize=False)
            time.sleep(step_duration)
