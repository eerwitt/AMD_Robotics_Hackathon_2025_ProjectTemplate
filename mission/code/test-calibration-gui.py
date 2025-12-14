#!/usr/bin/env python3
"""Interactive calibration GUI for the SO-101 follower with IK teleop overlay.

This script reuses the range-slider UI from the motor calibration helper but adds
an optional inverse-kinematics teleoperation bridge. When the leader arm is
connected the GUI streams its task-space pose, solves IK with LeRobot's helpers,
and continuously writes joint targets to the follower. Sliders always reflect
the current calibration map plus the live joint readings so it is easy to spot
outliers before saving an updated calibration set.

Run it as:

    python test-calibration-gui.py \
        --robot.port=/dev/ttyACM1 --robot.id=my_awesome_follower_arm \
        --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm

Use ``--enable_teleop=false`` when only the visualization part is needed.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:  # pragma: no cover - pygame is only needed at runtime.
    import pygame
except ImportError as exc:  # pragma: no cover - importing fails during linting.
    raise SystemExit(
        "pygame is required for test-calibration-gui.py. Install it with 'pip install pygame'."
    ) from exc

from lerobot.configs import parser
from lerobot.motors import MotorCalibration, MotorsBus
from lerobot.robots import Robot, RobotConfig, so101_follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config

try:  # pragma: no cover - optional dependency in some setups.
    from lerobot.robots import so101_leader
except Exception:  # pragma: no cover - if leader cfg is not bundled.
    so101_leader = None  # type: ignore[assignment]

DEFAULT_ROBOT_ID = "my_awesome_follower_arm"
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"

DEFAULT_TELEOP_ID = "my_awesome_leader_arm"
DEFAULT_TELEOP_PORT = "/dev/ttyACM1"

BAR_LEN, BAR_THICKNESS = 450, 8
HANDLE_R = 10
BRACKET_W, BRACKET_H = 6, 14
TRI_W, TRI_H = 12, 14

BTN_W, BTN_H = 60, 22
SAVE_W, SAVE_H = 80, 28
LOAD_W = 80
SNAP_W = 120
MOVE_W = 150
CYCLE_W = 120
STOP_W = 80
DD_W, DD_H = 160, 28

TOP_GAP = 80
PADDING_Y, TOP_OFFSET = 70, 60
FONT_SIZE, FPS = 20, 60

BG_COLOR = (30, 30, 30)
BAR_RED, BAR_GREEN = (200, 60, 60), (60, 200, 60)
HANDLE_COLOR, TEXT_COLOR = (240, 240, 240), (250, 250, 250)
TICK_COLOR = (250, 220, 40)
BTN_COLOR, BTN_COLOR_HL = (80, 80, 80), (110, 110, 110)
DD_COLOR, DD_COLOR_HL = (70, 70, 70), (100, 100, 100)
TELEOP_TAG_BG = (54, 54, 54)
STATUS_BG = (45, 45, 45)
PROMPT_BG = (36, 36, 36)
PROMPT_BORDER = (140, 140, 140)
POSE_MENU_BG = (40, 40, 40)
POSE_MENU_BORDER = (130, 130, 130)
POSE_MENU_ITEM = (60, 60, 60)
POSE_MENU_ITEM_HL = (90, 90, 90)
POSE_MENU_TEXT_DIM = (200, 200, 200)
POSE_MENU_ITEM_H = 30
POSE_MENU_MAX_VISIBLE = 8
POSE_MENU_W = 360
SNAPSHOT_DIR_ENV = "SO101_CALIB_SNAPSHOT_DIR"
DEFAULT_SNAPSHOT_DIR = Path("outputs/calibration_snapshots")
STATUS_TIMEOUT_S = 4.0


def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(slots=True)
class RangeValues:
    min_v: int
    pos_v: int
    max_v: int


@dataclass(slots=True)
class TaskSpacePose:
    """Normalized task-space pose used by the IK bridge."""

    position: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, Sequence[float]]:
        payload: dict[str, Sequence[float]] = {"position": list(self.position)}
        if self.orientation_rpy is not None:
            payload["orientation_rpy"] = list(self.orientation_rpy)
        return payload


class RangeSlider:
    """One motor = one slider row."""

    def __init__(
        self,
        motor: str,
        idx: int,
        res: int,
        calibration: MotorCalibration,
        present: int,
        label_pad: int,
        base_y: int,
    ) -> None:
        self.motor = motor
        self.res = res
        self.x0 = 40 + label_pad
        self.x1 = self.x0 + BAR_LEN
        self.y = base_y + idx * PADDING_Y

        self.min_v = calibration.range_min
        self.max_v = calibration.range_max
        self.pos_v = max(self.min_v, min(present, self.max_v))

        self.min_x = self._pos_from_val(self.min_v)
        self.max_x = self._pos_from_val(self.max_v)
        self.pos_x = self._pos_from_val(self.pos_v)

        self.min_btn = pygame.Rect(self.x0 - BTN_W - 6, self.y - BTN_H // 2, BTN_W, BTN_H)
        self.max_btn = pygame.Rect(self.x1 + 6, self.y - BTN_H // 2, BTN_W, BTN_H)

        self.drag_min = self.drag_max = self.drag_pos = False
        self.tick_val = present
        self.font = pygame.font.Font(None, FONT_SIZE)

    def _val_from_pos(self, x: float) -> int:
        return round((x - self.x0) / BAR_LEN * self.res)

    def _pos_from_val(self, v: float) -> float:
        return self.x0 + (v / self.res) * BAR_LEN

    def set_tick(self, v: float) -> None:
        self.tick_val = max(0, min(int(v), self.res))

    def _triangle_hit(self, pos: tuple[int, int]) -> bool:
        tri_top = self.y - BAR_THICKNESS // 2 - 2
        return pygame.Rect(self.pos_x - TRI_W // 2, tri_top - TRI_H, TRI_W, TRI_H).collidepoint(pos)

    def handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.min_btn.collidepoint(e.pos):
                self.min_x, self.min_v = self.pos_x, self.pos_v
                return
            if self.max_btn.collidepoint(e.pos):
                self.max_x, self.max_v = self.pos_x, self.pos_v
                return
            if dist(e.pos, (int(self.min_x), self.y)) <= HANDLE_R:
                self.drag_min = True
            elif dist(e.pos, (int(self.max_x), self.y)) <= HANDLE_R:
                self.drag_max = True
            elif self._triangle_hit(e.pos):
                self.drag_pos = True

        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag_min = self.drag_max = self.drag_pos = False

        elif e.type == pygame.MOUSEMOTION:
            x = e.pos[0]
            if self.drag_min:
                self.min_x = max(self.x0, min(x, self.pos_x))
            elif self.drag_max:
                self.max_x = min(self.x1, max(x, self.pos_x))
            elif self.drag_pos:
                self.pos_x = max(self.min_x, min(x, self.max_x))

            self.min_v = self._val_from_pos(self.min_x)
            self.max_v = self._val_from_pos(self.max_x)
            self.pos_v = self._val_from_pos(self.pos_x)

    def _draw_button(self, surf: pygame.Surface, rect: pygame.Rect, text: str) -> None:
        clr = BTN_COLOR_HL if rect.collidepoint(pygame.mouse.get_pos()) else BTN_COLOR
        pygame.draw.rect(surf, clr, rect, border_radius=4)
        t = self.font.render(text, True, TEXT_COLOR)
        surf.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

    def draw(self, surf: pygame.Surface) -> None:
        name_surf = self.font.render(self.motor, True, TEXT_COLOR)
        surf.blit(
            name_surf,
            (self.min_btn.right - name_surf.get_width(), self.min_btn.y - name_surf.get_height() - 4),
        )

        pygame.draw.rect(surf, BAR_RED, (self.x0, self.y - BAR_THICKNESS // 2, BAR_LEN, BAR_THICKNESS))
        pygame.draw.rect(
            surf,
            BAR_GREEN,
            (self.min_x, self.y - BAR_THICKNESS // 2, self.max_x - self.min_x, BAR_THICKNESS),
        )

        tick_x = self._pos_from_val(self.tick_val)
        pygame.draw.line(
            surf,
            TICK_COLOR,
            (tick_x, self.y - BAR_THICKNESS // 2 - 4),
            (tick_x, self.y + BAR_THICKNESS // 2 + 4),
            2,
        )

        for x, sign in ((self.min_x, +1), (self.max_x, -1)):
            pygame.draw.line(
                surf,
                HANDLE_COLOR,
                (x, self.y - BRACKET_H // 2),
                (x, self.y + BRACKET_H // 2),
                2,
            )
            pygame.draw.line(
                surf,
                HANDLE_COLOR,
                (x, self.y - BRACKET_H // 2),
                (x + sign * BRACKET_W, self.y - BRACKET_H // 2),
                2,
            )
            pygame.draw.line(
                surf,
                HANDLE_COLOR,
                (x, self.y + BRACKET_H // 2),
                (x + sign * BRACKET_W, self.y + BRACKET_H // 2),
                2,
            )

        tri_top = self.y - BAR_THICKNESS // 2 - 2
        pygame.draw.polygon(
            surf,
            HANDLE_COLOR,
            [
                (self.pos_x, tri_top),
                (self.pos_x - TRI_W // 2, tri_top - TRI_H),
                (self.pos_x + TRI_W // 2, tri_top - TRI_H),
            ],
        )

        fh = self.font.get_height()
        pos_y = tri_top - TRI_H - 4 - fh
        txts = [
            (self.min_v, self.min_x, self.y - BRACKET_H // 2 - 4 - fh),
            (self.max_v, self.max_x, self.y - BRACKET_H // 2 - 4 - fh),
            (self.pos_v, self.pos_x, pos_y),
        ]
        for v, x, y in txts:
            s = self.font.render(str(v), True, TEXT_COLOR)
            surf.blit(s, (x - s.get_width() // 2, y))

        self._draw_button(surf, self.min_btn, "set min")
        self._draw_button(surf, self.max_btn, "set max")

    def values(self) -> RangeValues:
        return RangeValues(self.min_v, self.pos_v, self.max_v)


class TeleopIKBridge:
    """Lightweight background thread that mirrors leader IK commands to the follower."""

    def __init__(
        self,
        follower: Robot,
        bus: MotorsBus,
        leader: Robot | None,
        rate_hz: float,
    ) -> None:
        self._follower = follower
        self._bus = bus
        self._leader = leader
        self._rate_hz = max(1.0, rate_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ik_solver = self._build_ik_solver()
        self._latest_pose: TaskSpacePose | None = None
        self._pose_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot_pose(self) -> TaskSpacePose | None:
        with self._pose_lock:
            return self._latest_pose

    def _build_ik_solver(self) -> Any:
        candidates = (
            getattr(self._follower, "ik_solver", None),
            getattr(self._follower, "cartesian_controller", None),
        )
        for builder_name in ("build_ik_solver", "build_inverse_kinematics", "build_cartesian_controller"):
            builder = getattr(self._follower, builder_name, None)
            if callable(builder):
                try:
                    solver = builder()
                except Exception as err:  # pragma: no cover - depends on hardware
                    logging.warning("Failed to call %s(): %s", builder_name, err)
                    continue
                if solver is not None:
                    return solver
        for candidate in candidates:
            if candidate is not None:
                return candidate
        logging.warning("No IK solver/controller exposed by %s; teleop disabled.", type(self._follower).__name__)
        return None

    def start(self) -> None:
        if self._leader is None:
            logging.info("No teleop leader configured; skipping IK bridge start.")
            return
        if self._ik_solver is None:
            logging.info("IK solver unavailable; teleop bridge will not run.")
            return
        self._thread = threading.Thread(target=self._loop, name="so101-ik-teleop", daemon=True)
        self._thread.start()
        logging.info("Started IK teleop bridge at %.1f Hz", self._rate_hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        while not self._stop.is_set():
            pose = self._read_leader_pose()
            if pose is None:
                time.sleep(period)
                continue

            target = self._solve_ik(pose)
            if not target:
                time.sleep(period)
                continue

            action_dict = self._coerce_to_motor_map(target)
            if not action_dict:
                time.sleep(period)
                continue

            try:
                self._bus.sync_write("Goal_Position", action_dict, normalize=False)
            except Exception as err:  # pragma: no cover - requires hardware
                logging.error("Failed to push teleop command: %s", err)
                time.sleep(period)
                continue

            with self._pose_lock:
                self._latest_pose = pose
            time.sleep(period)

    def _read_leader_pose(self) -> TaskSpacePose | None:
        leader = self._leader
        if leader is None:
            return None

        call_candidates = (
            "get_task_space_pose",
            "get_end_effector_pose",
            "get_pose",
            "read_pose",
            "read_task_pose",
        )
        for name in call_candidates:
            fn = getattr(leader, name, None)
            if callable(fn):
                try:
                    raw = fn()
                except Exception:  # pragma: no cover - hardware-specific
                    continue
                pose = self._normalize_pose(raw)
                if pose is not None:
                    return pose

        state = getattr(leader, "state", None)
        pose = self._normalize_pose(state)
        if pose is not None:
            return pose
        return None

    def _normalize_pose(self, raw: Any) -> TaskSpacePose | None:
        if isinstance(raw, TaskSpacePose):
            return raw

        if isinstance(raw, Mapping):
            position = raw.get("position") or raw.get("pos") or raw.get("xyz")
            orientation = raw.get("orientation_rpy") or raw.get("rpy") or raw.get("orientation")
            if position is None:
                return None
            pos_tuple = tuple(float(v) for v in position[:3])  # type: ignore[index]
            ori_tuple: tuple[float, float, float] | None = None
            if orientation is not None:
                ori_tuple = tuple(float(v) for v in orientation[:3])  # type: ignore[index]
            return TaskSpacePose(pos_tuple, ori_tuple)

        if isinstance(raw, Sequence) and len(raw) >= 6:
            pos_tuple = tuple(float(v) for v in raw[:3])  # type: ignore[index]
            ori_tuple = tuple(float(v) for v in raw[3:6])  # type: ignore[index]
            return TaskSpacePose(pos_tuple, ori_tuple)

        return None

    def _solve_ik(self, pose: TaskSpacePose) -> Any:
        solver = self._ik_solver
        if solver is None:
            return None

        payload = pose.as_dict()
        for callable_candidate in (
            getattr(solver, "solve", None),
            getattr(self._follower, "solve_ik", None),
            getattr(self._follower, "apply_task_space_pose", None),
        ):
            if callable(callable_candidate):
                try:
                    return callable_candidate(payload)
                except TypeError:
                    try:
                        return callable_candidate(**payload)
                    except Exception:  # pragma: no cover - solver specific
                        continue
                except Exception:
                    logging.debug("IK solver call failed", exc_info=True)
                    continue

        if callable(solver):
            try:
                return solver(payload)
            except TypeError:
                try:
                    return solver(**payload)
                except Exception:
                    return None
            except Exception:
                logging.debug("IK callable raised", exc_info=True)
                return None

        return None

    def _coerce_to_motor_map(self, target: Any) -> dict[str, int] | None:
        if isinstance(target, Mapping):
            try:
                return {str(k): int(round(v)) for k, v in target.items()}
            except Exception:
                logging.debug("Failed to coerce IK dict result", exc_info=True)
                return None

        if isinstance(target, Sequence):
            try:
                names = list(self._bus.motors.keys())
                return {
                    name: int(round(val))
                    for name, val in zip(names, target, strict=False)
                }
            except Exception:
                logging.debug("Failed to coerce IK sequence result", exc_info=True)
                return None

        return None


class SO101CalibrationGUI:
    def __init__(
        self,
        bus: MotorsBus,
        teleop_bridge: TeleopIKBridge | None = None,
        groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.bus = bus
        self.teleop_bridge = teleop_bridge
        self.groups = groups or {"so101_full_arm": list(bus.motors.keys())}
        self.group_names = list(self.groups)
        self.current_group = self.group_names[0]

        self.bus_lock = threading.Lock()

        self.calibration = self._read_calibration_map()
        if not self.calibration:
            raise RuntimeError("Robot returned an empty calibration map; aborting GUI start.")
        self.res_table = bus.model_resolution_table
        self.present_cache = {
            m: self._read_present_position(m)
            for motors in self.groups.values()
            for m in motors
        }

        pygame.init()
        self.font = pygame.font.Font(None, FONT_SIZE)

        label_pad = max(self.font.size(m)[0] for motors in self.groups.values() for m in motors)
        self.label_pad = label_pad
        width = (
            40
            + label_pad
            + BAR_LEN
            + 6
            + BTN_W
            + 10
            + SAVE_W
            + 10
            + LOAD_W
            + 10
            + SNAP_W
            + 10
            + MOVE_W
            + 10
            + CYCLE_W
            + 10
            + STOP_W
        )
        buttons_row_top = SAVE_H + 10 + 10
        self.controls_bottom = buttons_row_top + SAVE_H
        self.base_y = self.controls_bottom + TOP_GAP

        motors = self.groups[self.current_group]
        height = self.base_y + PADDING_Y * len(motors) + 40

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SO-101 Calibration + IK Teleop")
        margin_left = 10
        spacing = 10
        current_x = margin_left
        self.save_btn = pygame.Rect(current_x, buttons_row_top, SAVE_W, SAVE_H)
        current_x += SAVE_W + spacing
        self.load_btn = pygame.Rect(current_x, buttons_row_top, LOAD_W, SAVE_H)
        current_x += LOAD_W + spacing
        self.snapshot_btn = pygame.Rect(current_x, buttons_row_top, SNAP_W, SAVE_H)
        current_x += SNAP_W + spacing
        self.move_pose_btn = pygame.Rect(current_x, buttons_row_top, MOVE_W, SAVE_H)
        current_x += MOVE_W + spacing
        self.cycle_btn = pygame.Rect(current_x, buttons_row_top, CYCLE_W, SAVE_H)
        current_x += CYCLE_W + spacing
        self.stop_cycle_btn = pygame.Rect(current_x, buttons_row_top, STOP_W, SAVE_H)
        self.dd_btn = pygame.Rect(width // 2 - DD_W // 2, 10, DD_W, DD_H)
        self.dd_open = False
        snapshot_root = Path(os.environ.get(SNAPSHOT_DIR_ENV, str(DEFAULT_SNAPSHOT_DIR)))
        snapshot_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = snapshot_root
        self.pose_menu_open = False
        self.pose_menu_entries: list[Path] = []
        self.pose_menu_scroll = 0
        self.pending_pose_path: Path | None = None
        self.cycle_menu_open = False
        self.cycle_selection: set[Path] = set()
        self.cycle_thread: threading.Thread | None = None
        self.cycle_stop_event = threading.Event()
        self.cycle_stop_event.clear()
        self.prompt_active = False
        self.prompt_buffer = ""
        self.prompt_kind: str | None = None
        self.prompt_label = ""
        self.prompt_instructions = ""
        self.prompt_placeholder = ""
        self.status_message = ""
        self.status_timestamp = 0.0
        self.default_pose_duration = 8.0
        self.motion_thread: threading.Thread | None = None
        self.motion_active = False

        self.clock = pygame.time.Clock()
        self._build_sliders()
        self._adjust_height()

    def _adjust_height(self) -> None:
        motors = self.groups[self.current_group]
        new_h = self.base_y + PADDING_Y * len(motors) + 40
        if new_h != self.screen.get_height():
            w = self.screen.get_width()
            self.screen = pygame.display.set_mode((w, new_h))

    def _build_sliders(self) -> None:
        self.sliders: list[RangeSlider] = []
        motors = self.groups[self.current_group]
        for i, m in enumerate(motors):
            resolution = self.res_table[self.bus.motors[m].model] - 1
            self.sliders.append(
                RangeSlider(
                    motor=m,
                    idx=i,
                    res=resolution,
                    calibration=self.calibration[m],
                    present=self.present_cache[m],
                    label_pad=self.label_pad,
                    base_y=self.base_y,
                )
            )

    def _draw_dropdown(self) -> None:
        hover = self.dd_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, DD_COLOR_HL if hover else DD_COLOR, self.dd_btn, border_radius=6)

        txt = self.font.render(self.current_group, True, TEXT_COLOR)
        self.screen.blit(
            txt, (self.dd_btn.centerx - txt.get_width() // 2, self.dd_btn.centery - txt.get_height() // 2)
        )

        tri_w, tri_h = 12, 6
        cx = self.dd_btn.right - 14
        cy = self.dd_btn.centery + 1
        pygame.draw.polygon(
            self.screen,
            TEXT_COLOR,
            [(cx - tri_w // 2, cy - tri_h // 2), (cx + tri_w // 2, cy - tri_h // 2), (cx, cy + tri_h // 2)],
        )

        if not self.dd_open:
            return

        for i, name in enumerate(self.group_names):
            item_rect = pygame.Rect(self.dd_btn.left, self.dd_btn.bottom + i * DD_H, DD_W, DD_H)
            clr = DD_COLOR_HL if item_rect.collidepoint(pygame.mouse.get_pos()) else DD_COLOR
            pygame.draw.rect(self.screen, clr, item_rect)
            t = self.font.render(name, True, TEXT_COLOR)
            self.screen.blit(t, (item_rect.centerx - t.get_width() // 2, item_rect.centery - t.get_height() // 2))

    def _handle_dropdown_event(self, e: pygame.event.Event) -> bool:
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.dd_btn.collidepoint(e.pos):
                self.dd_open = not self.dd_open
                return True
            if self.dd_open:
                for i, name in enumerate(self.group_names):
                    item_rect = pygame.Rect(self.dd_btn.left, self.dd_btn.bottom + i * DD_H, DD_W, DD_H)
                    if item_rect.collidepoint(e.pos):
                        if name != self.current_group:
                            self.current_group = name
                            self._build_sliders()
                            self._adjust_height()
                        self.dd_open = False
                        return True
                self.dd_open = False
        return False

    def _save_current(self) -> None:
        for s in self.sliders:
            self.calibration[s.motor].range_min = s.min_v
            self.calibration[s.motor].range_max = s.max_v

        self._write_calibration_map()

    def _load_current(self) -> None:
        self.calibration = self._read_calibration_map()
        for s in self.sliders:
            s.min_v = self.calibration[s.motor].range_min
            s.max_v = self.calibration[s.motor].range_max
            s.min_x = s._pos_from_val(s.min_v)
            s.max_x = s._pos_from_val(s.max_v)

    def _set_status(self, text: str) -> None:
        self.status_message = text
        self.status_timestamp = time.time()

    def _status_visible(self) -> bool:
        if not self.status_message:
            return False
        if time.time() - self.status_timestamp > STATUS_TIMEOUT_S:
            self.status_message = ""
            return False
        return True

    def _draw_status_bar(self) -> None:
        if not self._status_visible():
            return
        rect = pygame.Rect(0, self.screen.get_height() - 28, self.screen.get_width(), 24)
        pygame.draw.rect(self.screen, STATUS_BG, rect)
        txt = self.font.render(self.status_message, True, TEXT_COLOR)
        self.screen.blit(txt, (rect.x + 12, rect.y + rect.height // 2 - txt.get_height() // 2))

    def _start_snapshot_prompt(self) -> None:
        self._start_prompt(
            kind="save_pose",
            label="Save pose as:",
            instructions="Enter to save, Esc to cancel",
            placeholder="<name>",
            initial_text="",
        )
        self._set_status("Type a filename, Enter to save, Esc to cancel.")

    def _start_pose_duration_prompt(self, snapshot_path: Path) -> None:
        self.pending_pose_path = snapshot_path
        self._start_prompt(
            kind="pose_duration",
            label=f"Move to '{snapshot_path.stem}' over (seconds):",
            instructions="Enter duration > 0, Esc to cancel",
            placeholder="<seconds>",
            initial_text=f"{self.default_pose_duration:.1f}",
        )

    def _start_prompt(
        self,
        *,
        kind: str,
        label: str,
        instructions: str,
        placeholder: str,
        initial_text: str,
    ) -> None:
        self.pose_menu_open = False
        self.prompt_active = True
        self.prompt_kind = kind
        self.prompt_label = label
        self.prompt_instructions = instructions
        self.prompt_placeholder = placeholder
        self.prompt_buffer = initial_text

    def _handle_prompt_event(self, e: pygame.event.Event) -> bool:
        if not self.prompt_active:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                self._submit_prompt()
            elif e.key == pygame.K_ESCAPE:
                self._set_status("Input cancelled.")
                self._close_prompt()
            elif e.key == pygame.K_BACKSPACE:
                self.prompt_buffer = self.prompt_buffer[:-1]
            else:
                if e.unicode and self._is_valid_prompt_char(e.unicode):
                    self.prompt_buffer += e.unicode
            return True
        if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            return True
        return False

    def _draw_prompt(self) -> None:
        if not self.prompt_active:
            return
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        rect = pygame.Rect(0, 0, min(420, self.screen.get_width() - 40), 110)
        rect.center = self.screen.get_rect().center
        pygame.draw.rect(self.screen, PROMPT_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, PROMPT_BORDER, rect, width=2, border_radius=8)
        label = self.prompt_label or "Input"
        label_txt = self.font.render(label, True, TEXT_COLOR)
        self.screen.blit(label_txt, (rect.centerx - label_txt.get_width() // 2, rect.y + 16))
        value = self.prompt_buffer or self.prompt_placeholder or ""
        value_color = TEXT_COLOR if self.prompt_buffer else POSE_MENU_TEXT_DIM
        value_txt = self.font.render(value, True, value_color)
        self.screen.blit(value_txt, (rect.centerx - value_txt.get_width() // 2, rect.centery - value_txt.get_height() // 2))
        instr = self.prompt_instructions or ""
        if instr:
            instr_txt = self.font.render(instr, True, TEXT_COLOR)
            self.screen.blit(
                instr_txt,
                (rect.centerx - instr_txt.get_width() // 2, rect.bottom - instr_txt.get_height() - 16),
            )

    def _is_valid_prompt_char(self, char: str) -> bool:
        if self.prompt_kind == "pose_duration":
            return char.isdigit() or char in {".", ",", "-"}
        if char.isalnum():
            return True
        return char in {"_", "-", " "}

    def _close_prompt(self) -> None:
        self.prompt_active = False
        self.prompt_buffer = ""
        self.prompt_kind = None
        self.prompt_label = ""
        self.prompt_instructions = ""
        self.prompt_placeholder = ""

    def _submit_prompt(self) -> None:
        if self.prompt_kind == "pose_duration":
            self._finalize_duration_prompt()
            return
        if self.prompt_kind == "save_pose":
            self._finalize_snapshot_prompt()
            return

    def _finalize_duration_prompt(self) -> None:
        raw = self.prompt_buffer.strip()
        raw = raw.replace(",", ".")
        try:
            duration = float(raw)
        except ValueError:
            self._set_status("Enter a numeric duration in seconds.")
            return
        if duration <= 0:
            self._set_status("Duration must be > 0 seconds.")
            return
        target = self.pending_pose_path
        if target is None:
            self._set_status("No snapshot selected.")
            self._close_prompt()
            return
        self.default_pose_duration = duration
        self._close_prompt()
        self._start_pose_motion(target, duration)

    def _finalize_snapshot_prompt(self) -> None:
        name = self._normalize_name(self.prompt_buffer)
        if not name:
            self._set_status("Please enter at least one letter or number.")
            return
        try:
            path = self._save_named_snapshot(name)
        except Exception as err:  # pragma: no cover - file I/O at runtime.
            logging.exception("Failed to save snapshot '%s'", name)
            self._set_status(f"Failed to save snapshot: {err}")
        else:
            self._set_status(f"Snapshot saved to {path.name}.")
        finally:
            self._close_prompt()

    @staticmethod
    def _normalize_name(raw: str) -> str:
        cleaned = raw.strip().replace(" ", "_")
        filtered = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"})
        return filtered[:80]

    def _save_named_snapshot(self, base_name: str) -> Path:
        filename = f"{base_name}.json"
        path = self.snapshot_dir / filename
        if path.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.snapshot_dir / f"{base_name}-{timestamp}.json"
        payload = self._build_snapshot_payload(base_name)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def _build_snapshot_payload(self, name: str) -> dict[str, Any]:
        motors_payload: dict[str, dict[str, Any]] = {}
        for motor, calibration in self.calibration.items():
            motors_payload[motor] = self._calibration_to_dict(calibration)
        for slider in self.sliders:
            entry = motors_payload.setdefault(slider.motor, {})
            entry["range_min"] = int(slider.min_v)
            entry["range_max"] = int(slider.max_v)
            entry["saved_position"] = int(slider.pos_v)
            entry["present_position"] = int(self.present_cache.get(slider.motor, slider.pos_v))
        return {
            "name": name,
            "group": self.current_group,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "motors": motors_payload,
        }

    @staticmethod
    def _calibration_to_dict(calibration: MotorCalibration) -> dict[str, Any]:
        if is_dataclass(calibration):
            data = asdict(calibration)
        elif isinstance(calibration, Mapping):
            data = dict(calibration)
        else:
            data = {}
            for key in ("range_min", "range_max"):
                if hasattr(calibration, key):
                    data[key] = getattr(calibration, key)
        if "range_min" in data:
            data["range_min"] = int(data["range_min"])
        if "range_max" in data:
            data["range_max"] = int(data["range_max"])
        return data

    def _read_calibration_map(self) -> dict[str, MotorCalibration]:
        with self.bus_lock:
            return self.bus.read_calibration()

    def _write_calibration_map(self) -> None:
        with self.bus_lock:
            with self.bus.torque_disabled():
                self.bus.write_calibration(self.calibration)

    def _read_present_position(self, motor: str) -> int:
        with self.bus_lock:
            return self.bus.read("Present_Position", motor, normalize=False)

    def _write_goal_position(self, motor: str, value: int) -> None:
        with self.bus_lock:
            self.bus.write("Goal_Position", motor, value, normalize=False)

    def _sync_read_present_positions(self) -> dict[str, int]:
        with self.bus_lock:
            return self.bus.sync_read("Present_Position", normalize=False)

    def _sync_write_goal_positions(self, action: Mapping[str, int]) -> None:
        with self.bus_lock:
            self.bus.sync_write("Goal_Position", action, normalize=False)

    def _toggle_pose_menu(self) -> None:
        if self.pose_menu_open:
            self.pose_menu_open = False
            return
        self._refresh_pose_menu_entries()
        self.pose_menu_open = True

    def _refresh_pose_menu_entries(self) -> None:
        try:
            entries = sorted(
                (path for path in self.snapshot_dir.glob("*.json") if path.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception as err:  # pragma: no cover - filesystem specific
            logging.error("Failed to enumerate snapshots in %s: %s", self.snapshot_dir, err)
            entries = []
        self.pose_menu_entries = entries
        self.pose_menu_scroll = 0

    def _handle_pose_menu_event(self, e: pygame.event.Event) -> bool:
        if not self.pose_menu_open:
            return False
        panel_rect, items = self._pose_menu_layout()
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.pose_menu_open = False
            return True
        if e.type == pygame.MOUSEBUTTONDOWN:
            if panel_rect.collidepoint(e.pos):
                if not items:
                    return True
                for rect, path, _ in items:
                    if rect.collidepoint(e.pos):
                        self.pose_menu_open = False
                        self._start_pose_duration_prompt(path)
                        return True
                return True
            self.pose_menu_open = False
            return True
        if e.type == pygame.MOUSEWHEEL:
            total = max(0, len(self.pose_menu_entries) - POSE_MENU_MAX_VISIBLE)
            if total == 0:
                return True
            self.pose_menu_scroll = max(0, min(self.pose_menu_scroll - e.y, total))
            return True
        return False

    def _pose_menu_layout(self) -> tuple[pygame.Rect, list[tuple[pygame.Rect, Path, str]]]:
        width = min(POSE_MENU_W, self.screen.get_width() - 40)
        visible = self.pose_menu_entries[self.pose_menu_scroll : self.pose_menu_scroll + POSE_MENU_MAX_VISIBLE]
        item_rows = max(len(visible), 1)
        height = item_rows * POSE_MENU_ITEM_H + 20
        panel = pygame.Rect(0, 0, width, height)
        min_cx = panel.width // 2 + 10
        max_cx = self.screen.get_width() - panel.width // 2 - 10
        centerx = min(max(self.move_pose_btn.centerx, min_cx), max_cx)
        panel.centerx = centerx
        panel.top = self.move_pose_btn.bottom + 8
        items: list[tuple[pygame.Rect, Path, str]] = []
        start_y = panel.y + 10
        for idx, path in enumerate(visible):
            item_rect = pygame.Rect(panel.x + 10, start_y + idx * POSE_MENU_ITEM_H, panel.width - 20, POSE_MENU_ITEM_H - 6)
            items.append((item_rect, path, path.stem))
        return panel, items

    def _draw_pose_menu(self) -> None:
        if not self.pose_menu_open:
            return
        panel_rect, items = self._pose_menu_layout()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, POSE_MENU_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, POSE_MENU_BORDER, panel_rect, width=2, border_radius=8)
        if not items:
            msg = "No snapshots found"
            txt = self.font.render(msg, True, TEXT_COLOR)
            self.screen.blit(
                txt,
                (panel_rect.centerx - txt.get_width() // 2, panel_rect.centery - txt.get_height() // 2),
            )
            return
        mouse = pygame.mouse.get_pos()
        for rect, path, label in items:
            clr = POSE_MENU_ITEM_HL if rect.collidepoint(mouse) else POSE_MENU_ITEM
            pygame.draw.rect(self.screen, clr, rect, border_radius=4)
            timestamp = ""
            try:
                ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
                timestamp = ts
            except Exception:
                timestamp = ""
            text = f"{label}"
            txt = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(txt, (rect.x + 8, rect.centery - txt.get_height() // 2))
            if timestamp:
                ts_txt = self.font.render(timestamp, True, POSE_MENU_TEXT_DIM)
                self.screen.blit(ts_txt, (rect.right - ts_txt.get_width() - 8, rect.centery - ts_txt.get_height() // 2))
            if len(self.pose_menu_entries) > POSE_MENU_MAX_VISIBLE:
                info = f"{self.pose_menu_scroll + 1}-{min(len(self.pose_menu_entries), self.pose_menu_scroll + POSE_MENU_MAX_VISIBLE)} / {len(self.pose_menu_entries)}"
                info_txt = self.font.render(info, True, POSE_MENU_TEXT_DIM)
                self.screen.blit(info_txt, (panel_rect.centerx - info_txt.get_width() // 2, panel_rect.bottom - info_txt.get_height() - 4))

    def _toggle_cycle_menu(self) -> None:
        if self.motion_active:
            self._set_status("Stop current motion before starting a cycle.")
            return
        self.pose_menu_open = False
        self.prompt_active = False
        if self.cycle_menu_open:
            self.cycle_menu_open = False
            self.cycle_selection.clear()
            return
        self._refresh_pose_menu_entries()
        self.cycle_selection.clear()
        self.cycle_menu_open = True
        self._set_status("Select poses to cycle and press Enter to start.")

    def _handle_cycle_menu_event(self, e: pygame.event.Event) -> bool:
        if not self.cycle_menu_open:
            return False
        panel_rect, items = self._cycle_menu_layout()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.cycle_menu_open = False
                self.cycle_selection.clear()
                return True
            if e.key == pygame.K_RETURN:
                if not self.cycle_selection:
                    self._set_status("Select at least one pose to cycle.")
                    return True
                selected = sorted(self.cycle_selection, key=lambda p: p.stem.lower())
                self.cycle_menu_open = False
                self.cycle_selection.clear()
                self._start_pose_cycle(selected)
                return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if panel_rect.collidepoint(e.pos):
                for rect, path, _ in items:
                    if rect.collidepoint(e.pos):
                        if path in self.cycle_selection:
                            self.cycle_selection.remove(path)
                        else:
                            self.cycle_selection.add(path)
                        return True
                return True
            self.cycle_menu_open = False
            self.cycle_selection.clear()
            return True
        return True

    def _cycle_menu_layout(self) -> tuple[pygame.Rect, list[tuple[pygame.Rect, Path, str]]]:
        width = min(POSE_MENU_W, self.screen.get_width() - 40)
        visible = self.pose_menu_entries[self.pose_menu_scroll : self.pose_menu_scroll + POSE_MENU_MAX_VISIBLE]
        item_rows = max(len(visible), 1)
        height = item_rows * POSE_MENU_ITEM_H + 40
        panel = pygame.Rect(0, 0, width, height)
        min_cx = panel.width // 2 + 10
        max_cx = self.screen.get_width() - panel.width // 2 - 10
        centerx = min(max(self.cycle_btn.centerx, min_cx), max_cx)
        panel.centerx = centerx
        panel.top = self.cycle_btn.bottom + 8
        items: list[tuple[pygame.Rect, Path, str]] = []
        start_y = panel.y + 10
        for idx, path in enumerate(visible):
            item_rect = pygame.Rect(panel.x + 10, start_y + idx * POSE_MENU_ITEM_H, panel.width - 20, POSE_MENU_ITEM_H - 6)
            items.append((item_rect, path, path.stem))
        return panel, items

    def _draw_cycle_menu(self) -> None:
        if not self.cycle_menu_open:
            return
        panel_rect, items = self._cycle_menu_layout()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, POSE_MENU_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, POSE_MENU_BORDER, panel_rect, width=2, border_radius=8)
        if not items:
            msg = "No snapshots found"
            txt = self.font.render(msg, True, TEXT_COLOR)
            self.screen.blit(
                txt,
                (panel_rect.centerx - txt.get_width() // 2, panel_rect.centery - txt.get_height() // 2),
            )
            return
        mouse = pygame.mouse.get_pos()
        for rect, path, label in items:
            selected = path in self.cycle_selection
            clr = POSE_MENU_ITEM_HL if rect.collidepoint(mouse) else POSE_MENU_ITEM
            pygame.draw.rect(self.screen, clr, rect, border_radius=4)
            checkbox = "[x]" if selected else "[ ]"
            text = f"{checkbox} {label}"
            txt = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(txt, (rect.x + 8, rect.centery - txt.get_height() // 2))
        instructions = "Enter to start, Esc to cancel"
        instr_txt = self.font.render(instructions, True, TEXT_COLOR)
        self.screen.blit(instr_txt, (panel_rect.centerx - instr_txt.get_width() // 2, panel_rect.bottom - instr_txt.get_height() - 8))

    def _start_pose_cycle(self, paths: list[Path]) -> None:
        if self.motion_active:
            self._set_status("Already busy with another motion.")
            return
        targets = self._load_cycle_targets(paths)
        if not targets:
            self._set_status("No valid poses selected for cycle.")
            return
        self.motion_active = True
        self.cycle_stop_event.clear()
        self.cycle_thread = threading.Thread(
            target=self._execute_pose_cycle,
            args=(targets,),
            daemon=True,
        )
        self.cycle_thread.start()

    def _stop_pose_cycle(self) -> None:
        if self.cycle_thread and self.cycle_thread.is_alive():
            self.cycle_stop_event.set()
            self.cycle_thread.join(timeout=3.0)
            self.cycle_thread = None
            self.motion_active = False
            self._set_status("Pose cycle stopped.")

    def _load_cycle_targets(self, paths: list[Path]) -> list[tuple[str, dict[str, int]]]:
        targets: list[tuple[str, dict[str, int]]] = []
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception as err:  # pragma: no cover - best effort
                logging.warning("Failed to read snapshot %s: %s", path, err)
                continue
            payload = self._build_pose_targets(raw)
            if payload:
                targets.append((path.stem, payload))
        return targets

    def _execute_pose_cycle(self, pose_targets: list[tuple[str, dict[str, int]]]) -> None:
        try:
            if not pose_targets:
                self._set_status("No targets to cycle.")
                return
            self._set_status("Starting pose cycle...")
            while not self.cycle_stop_event.is_set():
                try:
                    home_positions = self._sync_read_present_positions()
                except Exception as err:
                    logging.exception("Failed to read home positions during cycle")
                    self._set_status(f"Cycle aborted: {err}")
                    break
                if not home_positions:
                    self._set_status("Cycle aborted: unable to read home positions.")
                    break
                sequence = [("home", home_positions)]
                sequence.extend(pose_targets)
                sequence.append(("home", home_positions))
                current_positions = {name: float(value) for name, value in home_positions.items()}
                for label, target in sequence[1:]:
                    if self.cycle_stop_event.is_set():
                        break
                    self._set_status(f"Cycling: moving to {label}")
                    ended = self._execute_cycle_transition(current_positions, target, self.default_pose_duration)
                    current_positions = {name: float(value) for name, value in ended.items()}
                time.sleep(0.1)
        finally:
            self.motion_active = False
            self.cycle_thread = None
            self.cycle_stop_event.clear()
            self._set_status("Pose cycle ended.")

    def _execute_cycle_transition(
        self, start: Mapping[str, float], target: Mapping[str, int], duration: float
    ) -> dict[str, float]:
        names = [name for name in target if name in start]
        if not names:
            return start
        current = np.array([start[name] for name in names], dtype=np.float32)
        dest = np.array([target[name] for name in names], dtype=np.float32)
        num_steps = max(2, int(max(duration, 0.5) * 40))
        step_duration = max(duration / num_steps, 1e-3)
        for pose in np.linspace(current, dest, num_steps, dtype=np.float32):
            if self.cycle_stop_event.is_set():
                break
            action = {
                name: int(round(value))
                for name, value in zip(names, pose.tolist(), strict=False)
            }
            try:
                self._sync_write_goal_positions(action)
            except Exception as err:
                logging.exception("Failed during cycle transition")
                self._set_status(f"Cycle interrupted: {err}")
                return {name: float(dest[idx]) for idx, name in enumerate(names)}
            time.sleep(step_duration)
        return {name: float(dest[idx]) for idx, name in enumerate(names)}

    def _start_pose_motion(self, snapshot_path: Path, duration: float) -> None:
        if self.motion_thread and self.motion_thread.is_alive():
            self._set_status("Robot is already moving; wait for current motion to finish.")
            return
        self.motion_active = True
        self.motion_thread = threading.Thread(
            target=self._execute_pose_motion,
            args=(snapshot_path, duration),
            daemon=True,
        )
        self.motion_thread.start()

    def _execute_pose_motion(self, snapshot_path: Path, duration: float) -> None:
        try:
            try:
                with snapshot_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as err:
                logging.exception("Failed to load snapshot %s", snapshot_path)
                self._set_status(f"Failed to load snapshot: {err}")
                return

            targets = self._build_pose_targets(data)
            if not targets:
                self._set_status("Snapshot missing motor targets.")
                return

            try:
                current_positions = self._sync_read_present_positions()
            except Exception as err:
                logging.exception("Failed to read current joint state before motion")
                self._set_status(f"Failed to read joints: {err}")
                return

            names = [name for name in targets if name in current_positions]
            if not names:
                self._set_status("No overlapping motors between snapshot and robot.")
                return

            current = np.array([current_positions[name] for name in names], dtype=np.float32)
            target = np.array([targets[name] for name in names], dtype=np.float32)
            num_steps = max(2, int(duration * 40))
            step_duration = max(duration / num_steps, 1e-3)
            trajectory = np.linspace(current, target, num_steps, dtype=np.float32)
            self._set_status(f"Moving to '{snapshot_path.stem}' over {duration:.1f}s...")
            for pose in trajectory:
                action = {
                    name: int(round(value))
                    for name, value in zip(names, pose.tolist(), strict=False)
                }
                try:
                    self._sync_write_goal_positions(action)
                except Exception as err:
                    logging.exception("Failed during motion command")
                    self._set_status(f"Motion interrupted: {err}")
                    return
                time.sleep(step_duration)
            self._set_status(f"Reached pose '{snapshot_path.stem}'.")
        finally:
            self.motion_active = False
            self.motion_thread = None

    def _build_pose_targets(self, snapshot_payload: Mapping[str, Any]) -> dict[str, int]:
        motors_payload = snapshot_payload.get("motors", {})
        targets: dict[str, int] = {}
        if not isinstance(motors_payload, Mapping):
            return targets
        for motor in self.bus.motors.keys():
            entry = motors_payload.get(motor)
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("saved_position")
            if value is None:
                value = entry.get("present_position")
            if value is None:
                continue
            try:
                raw_int = int(round(float(value)))
            except Exception:
                continue
            targets[motor] = self._clamp_motor_value(motor, raw_int)
        return targets

    def _clamp_motor_value(self, motor: str, value: int) -> int:
        calibration = self.calibration.get(motor)
        if calibration is not None:
            return max(calibration.range_min, min(calibration.range_max, value))
        motor_info = self.bus.motors.get(motor)
        if motor_info is not None:
            resolution = self.res_table.get(motor_info.model)
            if resolution:
                return max(0, min(resolution - 1, value))
        return value

    def _draw_teleop_panel(self) -> None:
        rect = pygame.Rect(20, 10, self.dd_btn.left - 40, SAVE_H)
        pygame.draw.rect(self.screen, TELEOP_TAG_BG, rect, border_radius=6)

        if self.teleop_bridge and self.teleop_bridge.is_running:
            pose = self.teleop_bridge.snapshot_pose()
            if pose is None:
                text = "Teleop: running (awaiting pose)"
            else:
                px, py, pz = pose.position
                if pose.orientation_rpy is not None:
                    roll, pitch, yaw = pose.orientation_rpy
                    text = f"Teleop: pose=({px:.1f}, {py:.1f}, {pz:.1f}) rpy=({roll:.1f}, {pitch:.1f}, {yaw:.1f})"
                else:
                    text = f"Teleop: pose=({px:.1f}, {py:.1f}, {pz:.1f})"
        elif self.teleop_bridge:
            text = "Teleop: configured but idle"
        else:
            text = "Teleop: disabled"

        t = self.font.render(text, True, TEXT_COLOR)
        self.screen.blit(t, (rect.x + 10, rect.centery - t.get_height() // 2))

    def run(self) -> dict[str, MotorCalibration]:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return self.calibration

                if self.prompt_active and e.type != pygame.QUIT:
                    self._handle_prompt_event(e)
                    continue

                if self.cycle_menu_open and self._handle_cycle_menu_event(e):
                    continue

                if self.pose_menu_open and self._handle_pose_menu_event(e):
                    continue

                if self._handle_dropdown_event(e):
                    continue

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if self.save_btn.collidepoint(e.pos):
                        self._save_current()
                        self._set_status("Saved calibration to robot.")
                    elif self.load_btn.collidepoint(e.pos):
                        self._load_current()
                        self._set_status("Reloaded calibration from robot.")
                    elif self.snapshot_btn.collidepoint(e.pos):
                        self._start_snapshot_prompt()
                    elif self.move_pose_btn.collidepoint(e.pos):
                        self._toggle_pose_menu()
                    elif self.cycle_btn.collidepoint(e.pos):
                        self._toggle_cycle_menu()
                    elif self.stop_cycle_btn.collidepoint(e.pos):
                        self._stop_pose_cycle()

                for s in self.sliders:
                    s.handle_event(e)

            if not self.prompt_active and not self.motion_active:
                for s in self.sliders:
                    if s.drag_pos:
                        self._write_goal_position(s.motor, s.pos_v)

            for s in self.sliders:
                pos = self._read_present_position(s.motor)
                s.set_tick(pos)
                self.present_cache[s.motor] = pos

            self.screen.fill(BG_COLOR)
            self._draw_teleop_panel()

            for s in self.sliders:
                s.draw(self.screen)

            self._draw_dropdown()

            stop_active = bool(self.cycle_thread and self.cycle_thread.is_alive())
            buttons = [
                (self.stop_cycle_btn, "STOP", lambda: (180, 60, 60) if stop_active else (70, 70, 70)),
                (self.cycle_btn, "CYCLE", lambda: BTN_COLOR),
                (self.move_pose_btn, "MOVE POSE", lambda: BTN_COLOR),
                (self.snapshot_btn, "SAVE POSE", lambda: BTN_COLOR),
                (self.load_btn, "LOAD", lambda: BTN_COLOR),
                (self.save_btn, "SAVE", lambda: BTN_COLOR),
            ]
            mouse_pos = pygame.mouse.get_pos()
            for rect, text, base_color_fn in buttons:
                base = base_color_fn()
                if rect == self.stop_cycle_btn:
                    clr = (210, 90, 90) if stop_active and rect.collidepoint(mouse_pos) else base
                else:
                    clr = BTN_COLOR_HL if rect.collidepoint(mouse_pos) else base
                pygame.draw.rect(self.screen, clr, rect, border_radius=6)
                t = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

            self._draw_pose_menu()
            self._draw_cycle_menu()
            self._draw_status_bar()
            self._draw_prompt()

            pygame.display.flip()
            self.clock.tick(FPS)


def _default_robot_config() -> so101_follower.SO101FollowerConfig:
    return so101_follower.SO101FollowerConfig(
        port=DEFAULT_ROBOT_PORT,
        id=DEFAULT_ROBOT_ID,
    )


def _default_teleop_config() -> TeleoperatorConfig | None:
    if so101_leader is None:
        return None
    return so101_leader.SO101LeaderConfig(
        port=DEFAULT_TELEOP_PORT,
        id=DEFAULT_TELEOP_ID,
    )


@dataclass
class CalibrationGUIConfig:
    robot: so101_follower.SO101FollowerConfig = field(default_factory=_default_robot_config)
    teleop: TeleoperatorConfig | None = field(default_factory=_default_teleop_config)
    calibrate_on_connect: bool = False
    enable_teleop: bool = True
    teleop_rate_hz: float = 30.0
    groups: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        if self.robot is None:
            self.robot = _default_robot_config()

        if self.teleop is None:
            self.teleop = _default_teleop_config()


@parser.wrap()
def main(cfg: CalibrationGUIConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    register_third_party_plugins()

    robot = make_robot_from_config(cfg.robot)
    robot_name = getattr(cfg.robot, "id", "so101_follower")
    logging.info("Connecting to follower %s (calibrate=%s)", robot_name, cfg.calibrate_on_connect)
    robot.connect(calibrate=cfg.calibrate_on_connect)

    try:
        bus: MotorsBus = robot.bus  # type: ignore[assignment]
    except AttributeError as err:
        robot.disconnect()
        raise RuntimeError("Robot instance must expose a Dynamixel bus via `.bus`." ) from err

    teleop_robot: Robot | None = None
    teleop_bridge: TeleopIKBridge | None = None

    try:
        teleop_cfg = cfg.teleop
        if cfg.enable_teleop and teleop_cfg is not None:
            try:
                teleop_robot = make_teleoperator_from_config(teleop_cfg)
            except Exception as err:
                logging.warning("Unable to construct teleop device from config %s: %s", teleop_cfg, err)
                teleop_robot = None
            if teleop_robot is not None:
                teleop_name = getattr(teleop_cfg, "id", "so101_leader")
                logging.info("Connecting to leader %s for teleop", teleop_name)
                teleop_robot.connect(calibrate=False)
                teleop_bridge = TeleopIKBridge(robot, bus, teleop_robot, cfg.teleop_rate_hz)
                teleop_bridge.start()
        elif cfg.enable_teleop:
            logging.warning("Teleop configuration unavailable; teleop disabled.")

        gui = SO101CalibrationGUI(bus, teleop_bridge=teleop_bridge if cfg.enable_teleop else None, groups=cfg.groups)
        gui.run()
    finally:
        if teleop_bridge:
            teleop_bridge.stop()
        if teleop_robot is not None:
            logging.info("Disconnecting teleop leader")
            teleop_robot.disconnect()
        logging.info("Disconnecting follower")
        robot.disconnect()


if __name__ == "__main__":
    main()
