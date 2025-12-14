#!/usr/bin/env python3
"""Keyframe animation GUI for SO-101 follower motors.

This tool reuses the slider-driven UI from the calibration helper but records
snapshot keyframes and interpolates between them so the follower arm can be
animated smoothly.

Usage is similar to ``test-calibration-gui.py`` - pass ``--robot.port``/
``--robot.id`` to target the follower, then use the GUI controls to capture
keyframes, tune their timing, and hit ``PLAY``. Saved animations land in
``outputs/keyframe_animations``.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "pygame is required for test-keyframe-gui.py. Install it with 'pip install pygame'."
    ) from exc

from lerobot.configs import parser
from lerobot.motors import MotorCalibration, MotorsBus
from lerobot.robots import so101_follower
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.import_utils import register_third_party_devices

logger = logging.getLogger(__name__)

BAR_LEN, BAR_THICKNESS = 440, 8
HANDLE_R = 10
BRACKET_W, BRACKET_H = 6, 14
TRI_W, TRI_H = 12, 14
BTN_W, BTN_H = 60, 22
PADDING_Y = 62
FONT_SIZE = 18
FPS = 60
TIMELINE_HEIGHT = 90
SUMMARY_HEIGHT = 70
BUTTON_WIDTH = 130
BUTTON_HEIGHT = 32
BUTTON_SPACING = 12
TIMELINE_KEYFRAME_R = 8
TIMELINE_SEL_R = 12
MIN_DURATION = 0.5
DURATION_STEP = 0.5
PLAYBACK_RATE_HZ = 40.0
STATUS_TIMEOUT_S = 3.0
STATUS_HEIGHT = 28
KEYFRAME_LIST_LINES = 4
LOAD_MENU_MAX = 8
LOAD_MENU_ITEM_H = 28
LOAD_MENU_W = 360
DEFAULT_ANIMATION_DIR = Path("outputs/keyframe_animations")
SELECTOR_SIZE = 18

BG_COLOR = (24, 24, 24)
BAR_RED = (180, 60, 60)
BAR_GREEN = (60, 180, 90)
HANDLE_COLOR = (240, 240, 240)
TEXT_COLOR = (240, 240, 240)
TICK_COLOR = (255, 200, 70)
BTN_COLOR = (80, 80, 80)
BTN_COLOR_HL = (110, 110, 110)
SUMMARY_BG = (32, 32, 32)
SUMMARY_BORDER = (100, 100, 100)
TIMELINE_BG = (40, 40, 40)
TIMELINE_BORDER = (110, 110, 110)
KEYFRAME_COLOR = (90, 160, 240)
KEYFRAME_SELECTED = (255, 170, 60)
PLAYHEAD_COLOR = (240, 240, 240)
STATUS_BG = (28, 28, 28)
STATUS_BORDER = (90, 90, 90)
PROMPT_BG = (30, 30, 30)
PROMPT_BORDER = (150, 150, 150)


def dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(slots=True)
class RangeValues:
    min_v: int
    pos_v: int
    max_v: int


@dataclass
class Keyframe:
    name: str
    time: float
    positions: Mapping[str, int]


class RangeSlider:
    """One slider row for every motor."""

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
        sel_x = 12
        self.selector_rect = pygame.Rect(sel_x, self.y - SELECTOR_SIZE // 2, SELECTOR_SIZE, SELECTOR_SIZE)

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

    def draw(self, surf: pygame.Surface, selected: bool = False) -> None:
        pygame.draw.rect(surf, BG_COLOR, self.selector_rect, border_radius=4)
        border_color = KEYFRAME_SELECTED if selected else BTN_COLOR_HL
        pygame.draw.rect(surf, border_color, self.selector_rect, width=2, border_radius=4)
        dot_color = KEYFRAME_SELECTED if selected else TEXT_COLOR
        pygame.draw.circle(surf, dot_color, self.selector_rect.center, SELECTOR_SIZE // 4)

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

    def selector_hit(self, pos: tuple[int, int]) -> bool:
        return self.selector_rect.collidepoint(pos)


class KeyframeAnimationGUI:
    def __init__(
        self,
        bus: MotorsBus,
        groups: dict[str, list[str]] | None = None,
        duration: float = 6.0,
        playback_rate: float = PLAYBACK_RATE_HZ,
    ) -> None:
        self.bus = bus
        self.bus_lock = threading.Lock()
        self.groups = self._filter_groups(groups)
        self.group_names = list(self.groups)
        self.current_group = self.group_names[0]

        self.calibration = self._read_calibration_map()
        if not self.calibration:
            raise RuntimeError("Robot returned zero calibration entries; cannot start timeline.")
        self.res_table = getattr(bus, "model_resolution_table", {})

        motors = self.groups[self.current_group]
        self.present_cache = {}
        for motor in motors:
            try:
                value = self._read_present_position(motor)
            except Exception:
                value = 0
            self.present_cache[motor] = value

        pygame.init()
        self.font = pygame.font.Font(None, FONT_SIZE)
        label_pad = max(self.font.size(m)[0] for m in motors)
        self.label_pad = label_pad

        slider_width = 40 + label_pad + BAR_LEN + 40
        width = max(slider_width, 960)
        timeline_width = width - 40
        timeline_top = 20
        self.timeline_rect = pygame.Rect(20, timeline_top, timeline_width, TIMELINE_HEIGHT)
        summary_top = self.timeline_rect.bottom + 6
        self.summary_rect = pygame.Rect(20, summary_top, timeline_width, SUMMARY_HEIGHT)
        buttons_top = self.summary_rect.bottom + 10
        self.controls_bottom = buttons_top + BUTTON_HEIGHT
        self.base_y = self.controls_bottom + 20
        height = self.base_y + len(motors) * PADDING_Y + STATUS_HEIGHT + 20

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SO-101 Keyframe Animator")

        self.add_btn = pygame.Rect(20, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.remove_btn = pygame.Rect(self.add_btn.right + BUTTON_SPACING, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.play_btn = pygame.Rect(self.remove_btn.right + BUTTON_SPACING, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.stop_btn = pygame.Rect(self.play_btn.right + BUTTON_SPACING, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.save_btn = pygame.Rect(self.stop_btn.right + BUTTON_SPACING, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.load_btn = pygame.Rect(self.save_btn.right + BUTTON_SPACING, buttons_top, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.duration_minus_btn = pygame.Rect(self.load_btn.right + BUTTON_SPACING, buttons_top, 80, BUTTON_HEIGHT)
        self.duration_plus_btn = pygame.Rect(self.duration_minus_btn.right + 6, buttons_top, 80, BUTTON_HEIGHT)

        self.animation_dir = DEFAULT_ANIMATION_DIR
        self.animation_dir.mkdir(parents=True, exist_ok=True)

        self.load_menu_open = False
        self.load_menu_entries: list[Path] = []

        self.sliders: list[RangeSlider] = []
        self._build_sliders()

        self.keyframes: list[Keyframe] = []
        self.selected_keyframe: Keyframe | None = None
        self.dragging_keyframe: Keyframe | None = None
        self.timeline_duration = max(MIN_DURATION, duration)
        self.playback_rate_hz = max(5.0, playback_rate)
        self.playhead_time = 0.0
        self.playing = False
        self.playback_thread: threading.Thread | None = None
        self.animation_stop_event = threading.Event()
        self.scrub_time = 0.0
        self.scrub_dragging = False
        self.active_motor: str | None = None

        self.status_message = ""
        self.status_timestamp = 0.0

        self.prompt_active = False
        self.prompt_buffer = ""
        self.prompt_label = ""
        self.prompt_placeholder = "animation"
        self.prompt_instructions = ""

        self.clock = pygame.time.Clock()

    def _filter_groups(self, groups: dict[str, list[str]] | None) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        available = list(self.bus.motors.keys())
        if not groups:
            return {"so101_full_arm": available}
        for name, motors in groups.items():
            filtered = [m for m in motors if m in self.bus.motors]
            if filtered:
                cleaned[name] = filtered
        if not cleaned:
            cleaned["so101_full_arm"] = available
        return cleaned

    def _build_sliders(self) -> None:
        motors = self.groups[self.current_group]
        self.sliders = []
        base_y = self.base_y
        for idx, motor in enumerate(motors):
            real_res = self.res_table.get(self.bus.motors[motor].model)
            if real_res is None:
                calibration = self.calibration[motor]
                real_res = max(1, calibration.range_max - calibration.range_min)
            resolution = max(1, real_res - 1)
            slider = RangeSlider(
                motor=motor,
                idx=idx,
                res=resolution,
                calibration=self.calibration[motor],
                present=self.present_cache.get(motor, 0),
                label_pad=self.label_pad,
                base_y=base_y,
            )
            self.sliders.append(slider)

    def _read_calibration_map(self) -> dict[str, MotorCalibration]:
        with self.bus_lock:
            return self.bus.read_calibration()

    def _read_present_position(self, motor: str) -> int:
        with self.bus_lock:
            return self.bus.read("Present_Position", motor, normalize=False)

    def _write_goal_position(self, motor: str, value: int) -> None:
        with self.bus_lock:
            self.bus.write("Goal_Position", motor, value, normalize=False)

    def _sync_write_goal_positions(self, action: Mapping[str, int]) -> None:
        with self.bus_lock:
            self.bus.sync_write("Goal_Position", action, normalize=False)

    def run(self) -> None:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return

                if self.prompt_active:
                    self._handle_prompt_event(e)
                    continue

                if self.load_menu_open and self._handle_load_menu_event(e):
                    continue

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if self.add_btn.collidepoint(e.pos):
                        self._capture_keyframe()
                        continue
                    if self.remove_btn.collidepoint(e.pos):
                        self._remove_selected()
                        continue
                    if self.play_btn.collidepoint(e.pos):
                        self._start_playback()
                        continue
                    if self.stop_btn.collidepoint(e.pos):
                        self._stop_playback()
                        continue
                    if self.save_btn.collidepoint(e.pos):
                        self._begin_save_prompt()
                        continue
                    if self.load_btn.collidepoint(e.pos):
                        self._toggle_load_menu()
                        continue
                    if self.duration_minus_btn.collidepoint(e.pos):
                        self._adjust_duration(-DURATION_STEP)
                        continue
                    if self.duration_plus_btn.collidepoint(e.pos):
                        self._adjust_duration(DURATION_STEP)
                        continue

                for slider in self.sliders:
                    slider.handle_event(e)

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for slider in self.sliders:
                        if slider.selector_hit(e.pos):
                            self._toggle_active_motor(slider.motor)
                            break

                self._handle_timeline_event(e)

            if not self.playing:
                for slider in self.sliders:
                    if slider.drag_pos:
                        self._write_goal_position(slider.motor, slider.pos_v)

            for slider in self.sliders:
                pos = self._safe_read_present(slider.motor)
                slider.set_tick(pos)
                self.present_cache[slider.motor] = pos

            self.screen.fill(BG_COLOR)
            self._draw_timeline()
            self._draw_summary()
            self._draw_controls()
            self._draw_sliders()
            if self.load_menu_open:
                self._draw_load_menu()
            if self.prompt_active:
                self._draw_prompt()
            self._draw_status_bar()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _safe_read_present(self, motor: str) -> int:
        try:
            return self._read_present_position(motor)
        except Exception:
            logger.debug("Failed to read %s" % motor, exc_info=True)
            return self.present_cache.get(motor, 0)

    def _handle_timeline_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.timeline_rect.collidepoint(e.pos):
            clicked = self._keyframe_at_position(e.pos)
            if clicked is not None:
                self.selected_keyframe = clicked
                self.dragging_keyframe = clicked
                self.scrub_time = clicked.time
                return
            self.selected_keyframe = None
            self.scrub_dragging = True
            self._update_scrub_time_from_x(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging_keyframe = None
            self.scrub_dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging_keyframe is not None:
            x = max(self.timeline_rect.left, min(e.pos[0], self.timeline_rect.right))
            rel = (x - self.timeline_rect.left) / max(1, self.timeline_rect.width)
            self.dragging_keyframe.time = rel * self.timeline_duration
            self.scrub_time = self.dragging_keyframe.time
        elif e.type == pygame.MOUSEMOTION and self.scrub_dragging:
            self._update_scrub_time_from_x(e.pos[0])

    def _keyframe_at_position(self, pos: tuple[int, int]) -> Keyframe | None:
        for keyframe in self.keyframes:
            x = self._time_to_x(keyframe.time)
            y = self.timeline_rect.centery
            if dist(pos, (x, y)) <= TIMELINE_SEL_R:
                return keyframe
        return None

    def _time_to_x(self, time_value: float) -> int:
        if self.timeline_duration <= 0:
            return self.timeline_rect.left
        ratio = max(0.0, min(1.0, time_value / self.timeline_duration))
        return int(self.timeline_rect.left + ratio * self.timeline_rect.width)

    def _update_scrub_time_from_x(self, x: int) -> None:
        ratio = (x - self.timeline_rect.left) / max(1, self.timeline_rect.width)
        self.scrub_time = max(0.0, min(1.0, ratio)) * self.timeline_duration

    def _draw_timeline(self) -> None:
        pygame.draw.rect(self.screen, TIMELINE_BG, self.timeline_rect)
        pygame.draw.rect(self.screen, TIMELINE_BORDER, self.timeline_rect, width=2)
        sorted_kfs = self._sorted_keyframes()
        for keyframe in sorted_kfs:
            color = KEYFRAME_SELECTED if keyframe is self.selected_keyframe else KEYFRAME_COLOR
            pos_x = self._time_to_x(keyframe.time)
            pygame.draw.circle(self.screen, color, (pos_x, self.timeline_rect.centery), TIMELINE_KEYFRAME_R)
            label = self.font.render(keyframe.name, True, TEXT_COLOR)
            lbl_x = max(self.timeline_rect.left, min(pos_x - label.get_width() // 2, self.timeline_rect.right - label.get_width()))
            self.screen.blit(label, (lbl_x, self.timeline_rect.bottom - label.get_height() - 4))
        cursor_time = self.playhead_time if self.playing else self.scrub_time
        cursor_color = PLAYHEAD_COLOR if self.playing else TICK_COLOR
        playhead_x = self._time_to_x(cursor_time)
        pygame.draw.line(
            self.screen,
            cursor_color,
            (playhead_x, self.timeline_rect.top),
            (playhead_x, self.timeline_rect.bottom),
            2,
        )

    def _draw_summary(self) -> None:
        pygame.draw.rect(self.screen, SUMMARY_BG, self.summary_rect)
        pygame.draw.rect(self.screen, SUMMARY_BORDER, self.summary_rect, width=2)
        lines = [f"Duration: {self.timeline_duration:.1f}s · Rate: {self.playback_rate_hz:.0f}Hz · Keyframes: {len(self.keyframes)}"]
        lines.append(f"Scrub: {self.scrub_time:.2f}s")
        if self.active_motor:
            lines.append(f"Active servo: {self.active_motor}")
        if self.selected_keyframe:
            lines.append(f"Selected: {self.selected_keyframe.name} @{self.selected_keyframe.time:.2f}s")
        if not self.keyframes:
            lines.append("No keyframes captured yet. Use ADD KEYFRAME to capture the current pose.")
        else:
            sorted_kfs = self._sorted_keyframes()
            for idx, keyframe in enumerate(sorted_kfs[:KEYFRAME_LIST_LINES]):
                lines.append(f"{idx + 1}. {keyframe.name} @{keyframe.time:.2f}s")
        for row, text in enumerate(lines[:KEYFRAME_LIST_LINES + 1]):
            rendered = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(rendered, (self.summary_rect.left + 8, self.summary_rect.top + 6 + row * (self.font.get_height() + 2)))

    def _draw_controls(self) -> None:
        buttons = [
            (self.add_btn, "ADD KEYFRAME"),
            (self.remove_btn, "REMOVE"),
            (self.play_btn, "PLAY"),
            (self.stop_btn, "STOP"),
            (self.save_btn, "SAVE"),
            (self.load_btn, "LOAD"),
            (self.duration_minus_btn, "DUR -"),
            (self.duration_plus_btn, "DUR +"),
        ]
        for rect, label in buttons:
            clr = BTN_COLOR_HL if rect.collidepoint(pygame.mouse.get_pos()) else BTN_COLOR
            pygame.draw.rect(self.screen, clr, rect, border_radius=6)
            text = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

    def _draw_sliders(self) -> None:
        for slider in self.sliders:
            slider.draw(self.screen, slider.motor == self.active_motor)

    def _sorted_keyframes(self) -> list[Keyframe]:
        return sorted(self.keyframes, key=lambda kf: kf.time)

    def _capture_keyframe(self) -> None:
        if not self.sliders:
            self._set_status("No sliders available to capture.")
            return
        slider_values = {slider.motor: int(slider.pos_v) for slider in self.sliders}
        base_name = f"kf-{len(self.keyframes)+1}"
        time_stamp = max(0.0, min(self.timeline_duration, self.scrub_time))
        if self.active_motor is None:
            positions = slider_values
        else:
            baseline = self._pose_at_time(time_stamp)
            if not baseline:
                baseline = slider_values.copy()
            else:
                baseline = dict(baseline)
            for motor, value in slider_values.items():
                baseline.setdefault(motor, value)
            baseline[self.active_motor] = slider_values[self.active_motor]
            positions = baseline
        keyframe = Keyframe(name=base_name, time=time_stamp, positions=positions)
        self.keyframes.append(keyframe)
        self.selected_keyframe = keyframe
        self.scrub_time = time_stamp
        self._set_status(f"Captured {base_name} @ {time_stamp:.2f}s")

    def _toggle_active_motor(self, motor: str) -> None:
        if self.active_motor == motor:
            self.active_motor = None
            self._set_status(f"Deselected servo {motor}")
        else:
            self.active_motor = motor
            self._set_status(f"Selected servo {motor}")

    def _remove_selected(self) -> None:
        if self.selected_keyframe is None:
            self._set_status("No keyframe selected to remove.")
            return
        self.keyframes.remove(self.selected_keyframe)
        self._set_status(f"Removed {self.selected_keyframe.name}")
        self.selected_keyframe = None

    def _start_playback(self) -> None:
        if self.playing:
            self._set_status("Already playing.")
            return
        sorted_kfs = self._sorted_keyframes()
        if len(sorted_kfs) < 2:
            self._set_status("Capture at least two keyframes before playback.")
            return
        self.animation_stop_event.clear()
        self.playing = True
        self.playback_thread = threading.Thread(target=self._run_playback, args=(sorted_kfs,), daemon=True)
        self.playback_thread.start()
        self._set_status("Playback started")

    def _run_playback(self, keyframes: list[Keyframe]) -> None:
        finished_normally = False
        try:
            start = time.monotonic()
            frame_dt = 1.0 / self.playback_rate_hz
            while not self.animation_stop_event.is_set():
                now = time.monotonic()
                elapsed = max(0.0, now - start)
                t = min(elapsed, self.timeline_duration)
                action = self._interpolate_pose(t, keyframes)
                try:
                    self._sync_write_goal_positions(action)
                except Exception as err:
                    logger.exception("Playback write failed")
                    self._set_status(f"Playback interrupted: {err}")
                    break
                self.playhead_time = t
                if elapsed >= self.timeline_duration:
                    finished_normally = True
                    break
                sleep_time = frame_dt - (time.monotonic() - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.playing = False
            self.playhead_time = 0.0
            if finished_normally:
                self._set_status("Playback finished")
            self.animation_stop_event.set()
            self.playback_thread = None

    def _interpolate_pose(self, t: float, keyframes: list[Keyframe]) -> dict[str, int]:
        if not keyframes:
            return {}
        if t <= keyframes[0].time:
            return dict(keyframes[0].positions)
        if t >= keyframes[-1].time:
            return dict(keyframes[-1].positions)
        for left, right in zip(keyframes, keyframes[1:]):
            if left.time <= t <= right.time:
                span = max(right.time - left.time, 1e-3)
                alpha = (t - left.time) / span
                return {
                    motor: int(round(left.positions[motor] + (right.positions[motor] - left.positions[motor]) * alpha))
                    for motor in left.positions
                }
        return dict(keyframes[-1].positions)

    def _pose_at_time(self, time_value: float) -> dict[str, int]:
        return self._interpolate_pose(time_value, self._sorted_keyframes())

    def _stop_playback(self) -> None:
        if not self.playing:
            self._set_status("Playback is not running.")
            return
        self.animation_stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        self.playing = False
        self.playhead_time = 0.0
        self._set_status("Playback stopped")

    def _adjust_duration(self, delta: float) -> None:
        self.timeline_duration = max(MIN_DURATION, self.timeline_duration + delta)
        for keyframe in self.keyframes:
            keyframe.time = min(keyframe.time, self.timeline_duration)
        self.scrub_time = max(0.0, min(self.scrub_time, self.timeline_duration))
        self._set_status(f"Timeline duration set to {self.timeline_duration:.1f}s")

    def _begin_save_prompt(self) -> None:
        self.prompt_active = True
        self.prompt_buffer = ""
        self.prompt_label = "Animation name"
        self.prompt_instructions = "Press Enter to save or Esc to cancel"

    def _handle_prompt_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.prompt_active = False
            elif e.key == pygame.K_RETURN:
                name = self.prompt_buffer.strip() or self.prompt_placeholder
                self.prompt_active = False
                self._save_animation(name)
            elif e.key == pygame.K_BACKSPACE:
                self.prompt_buffer = self.prompt_buffer[:-1]
            elif e.unicode.isprintable():
                self.prompt_buffer += e.unicode

    def _save_animation(self, name: str) -> None:
        if not self.keyframes:
            self._set_status("Nothing to save yet.")
            return
        payload = {
            "name": name,
            "duration": self.timeline_duration,
            "playback_rate_hz": self.playback_rate_hz,
            "keyframes": [
                {
                    "name": k.name,
                    "time": k.time,
                    "positions": dict(k.positions),
                }
                for k in self._sorted_keyframes()
            ],
        }
        safe_name = _normalize_name(name) or datetime.utcnow().strftime("animation-%Y%m%d-%H%M%S")
        path = self.animation_dir / f"{safe_name}.json"
        if path.exists():
            path = self.animation_dir / f"{safe_name}-{int(time.time())}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self._set_status(f"Saved animation to {path.name}")

    def _toggle_load_menu(self) -> None:
        if self.load_menu_open:
            self.load_menu_open = False
            return
        self._refresh_load_entries()
        self.load_menu_open = True

    def _refresh_load_entries(self) -> None:
        try:
            entries = sorted(
                [path for path in self.animation_dir.glob("*.json") if path.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception as err:
            logger.warning("Failed to enumerate animations: %s", err)
            entries = []
        self.load_menu_entries = entries[:LOAD_MENU_MAX]

    def _draw_load_menu(self) -> None:
        if not self.load_menu_entries:
            return
        menu_x = self.load_btn.left
        menu_y = self.load_btn.bottom + 6
        menu_h = LOAD_MENU_ITEM_H * len(self.load_menu_entries)
        menu_rect = pygame.Rect(menu_x, menu_y, LOAD_MENU_W, menu_h)
        pygame.draw.rect(self.screen, SUMMARY_BG, menu_rect)
        pygame.draw.rect(self.screen, SUMMARY_BORDER, menu_rect, width=2)
        for idx, path in enumerate(self.load_menu_entries):
            item_rect = pygame.Rect(menu_x + 4, menu_y + idx * LOAD_MENU_ITEM_H, LOAD_MENU_W - 8, LOAD_MENU_ITEM_H)
            hover = item_rect.collidepoint(pygame.mouse.get_pos())
            bg = BTN_COLOR_HL if hover else BG_COLOR
            pygame.draw.rect(self.screen, bg, item_rect, border_radius=6)
            text = self.font.render(path.stem, True, TEXT_COLOR)
            self.screen.blit(text, (item_rect.left + 4, item_rect.centery - text.get_height() // 2))

    def _handle_load_menu_event(self, e: pygame.event.Event) -> bool:
        if e.type != pygame.MOUSEBUTTONDOWN or e.button != 1:
            return False
        if not self.load_menu_entries:
            self.load_menu_open = False
            return False
        menu_x = self.load_btn.left
        menu_y = self.load_btn.bottom + 6
        menu_rect = pygame.Rect(menu_x, menu_y, LOAD_MENU_W, LOAD_MENU_ITEM_H * len(self.load_menu_entries))
        if not menu_rect.collidepoint(e.pos):
            self.load_menu_open = False
            return False
        for idx, path in enumerate(self.load_menu_entries):
            item_rect = pygame.Rect(menu_x + 4, menu_y + idx * LOAD_MENU_ITEM_H, LOAD_MENU_W - 8, LOAD_MENU_ITEM_H)
            if item_rect.collidepoint(e.pos):
                self._load_animation(path)
                self.load_menu_open = False
                return True
        return False

    def _load_animation(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as err:
            self._set_status(f"Failed to load animation: {err}")
            return
        keyframes = []
        for item in payload.get("keyframes", []):
            if not isinstance(item, Mapping):
                continue
            positions = {motor: int(value) for motor, value in item.get("positions", {}).items()}
            keyframes.append(Keyframe(name=item.get("name", "unnamed"), time=float(item.get("time", 0.0)), positions=positions))
        if not keyframes:
            self._set_status("Animation file does not contain keyframes.")
            return
        self.keyframes = keyframes
        self.timeline_duration = max(MIN_DURATION, float(payload.get("duration", self.timeline_duration)))
        self.playback_rate_hz = float(payload.get("playback_rate_hz", self.playback_rate_hz))
        self.selected_keyframe = None
        self.scrub_time = 0.0
        self.active_motor = None
        self._set_status(f"Loaded {path.name}")

    def _draw_prompt(self) -> None:
        prompt_w = 420
        prompt_h = 140
        prompt_x = (self.screen.get_width() - prompt_w) // 2
        prompt_y = (self.screen.get_height() - prompt_h) // 2
        rect = pygame.Rect(prompt_x, prompt_y, prompt_w, prompt_h)
        pygame.draw.rect(self.screen, PROMPT_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, PROMPT_BORDER, rect, width=2, border_radius=8)
        label = self.font.render(self.prompt_label, True, TEXT_COLOR)
        self.screen.blit(label, (rect.left + 14, rect.top + 12))
        instructions = self.font.render(self.prompt_instructions, True, TEXT_COLOR)
        self.screen.blit(instructions, (rect.left + 14, rect.top + 36))
        text = self.prompt_buffer or self.prompt_placeholder
        input_surf = self.font.render(text, True, TEXT_COLOR)
        self.screen.blit(input_surf, (rect.left + 14, rect.top + 68))

    def _set_status(self, msg: str) -> None:
        self.status_message = msg
        self.status_timestamp = time.time()

    def _draw_status_bar(self) -> None:
        height = self.screen.get_height()
        rect = pygame.Rect(0, height - STATUS_HEIGHT, self.screen.get_width(), STATUS_HEIGHT)
        pygame.draw.rect(self.screen, STATUS_BG, rect)
        pygame.draw.rect(self.screen, STATUS_BORDER, rect, width=1)
        if time.time() - self.status_timestamp > STATUS_TIMEOUT_S:
            self.status_message = ""
        if self.status_message:
            text = self.font.render(self.status_message, True, TEXT_COLOR)
            self.screen.blit(text, (10, rect.top + (STATUS_HEIGHT - text.get_height()) // 2))


def _normalize_name(raw: str) -> str:
    cleaned = raw.strip().replace(" ", "_")
    filtered = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"})
    return filtered[:80]

def _default_robot_config() -> so101_follower.SO101FollowerConfig:
    return so101_follower.SO101FollowerConfig(port="/dev/ttyACM0", id="my_awesome_follower_arm")


@dataclass
class KeyframeAnimationGUIConfig:
    robot: so101_follower.SO101FollowerConfig = field(default_factory=_default_robot_config)
    groups: dict[str, list[str]] | None = None
    duration: float = 6.0
    playback_rate_hz: float = PLAYBACK_RATE_HZ


@parser.wrap()
def main(cfg: KeyframeAnimationGUIConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    register_third_party_devices()

    robot = make_robot_from_config(cfg.robot)
    logging.info("Connecting to follower %s", getattr(cfg.robot, "id", "so101_follower"))
    robot.connect(calibrate=False)

    try:
        bus: MotorsBus = robot.bus  # type: ignore[assignment]
        gui = KeyframeAnimationGUI(bus=bus, groups=cfg.groups, duration=cfg.duration, playback_rate=cfg.playback_rate_hz)
        gui.run()
    finally:
        logging.info("Disconnecting follower")
        robot.disconnect()


if __name__ == "__main__":
    main()
