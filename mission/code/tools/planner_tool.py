"""Smolagents helper that turns a high-level request into a concise small-step plan."""

from __future__ import annotations

import re
import textwrap
from typing import Any

from smolagents import Tool

MAX_STEPS_DEFAULT = 4
MAX_STEPS_LIMIT = 6
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TaskPlannerTool(Tool):
    """Generate a short, actionable plan so the agent can stay focused on small steps."""

    name = "break_down_task"
    output_type = "string"
    description = (
        "Summarize a task prompt into a handful of concrete, sequential steps the robot "
        "can execute with the available tools."
    )
    inputs = {
        "task": {
            "type": "string",
            "description": "Natural language description of the problem to solve.",
            "default": "",
            "nullable": False,
        },
        "focus_step": {
            "type": "string",
            "description": "Optional step that should be expanded into micro-steps (defaults to the overall task).",
            "default": "",
            "nullable": True,
        },
        "max_steps": {
            "type": "number",
            "description": "Maximum number of steps to include (default 4, max 6).",
            "default": MAX_STEPS_DEFAULT,
            "nullable": True,
        },
    }
    outputs = {
        "plan": {
            "type": "string",
            "description": "Indexed list of the computed sub-steps (one per line).",
        },
        "next_step": {
            "type": "string",
            "description": "First action from the plan that should happen next.",
        },
        "step_count": {
            "type": "number",
            "description": "Number of steps returned for this plan.",
        },
        "expanded_text": {
            "type": "string",
            "description": "The text (task or focus step) that was used to build this breakdown.",
        },
    }

    def forward(
        self,
        task: str | None = None,
        focus_step: str | None = None,
        max_steps: float | None = None,
    ) -> dict[str, Any]:
        target = (focus_step or task or "").strip()
        limit = self._normalize_step_limit(max_steps)
        if not target:
            return {
                "plan": "No task provided; describe what needs to be done before planning.",
                "next_step": "Provide a task description.",
                "step_count": 0,
            }

        steps = self._build_steps(target, limit)
        plan_text = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
        return {
            "plan": plan_text,
            "next_step": steps[0],
            "step_count": len(steps),
            "expanded_text": target,
        }

    @staticmethod
    def _normalize_step_limit(value: float | None) -> int:
        if value is None:
            return MAX_STEPS_DEFAULT
        try:
            requested = int(value)
        except (TypeError, ValueError):
            return MAX_STEPS_DEFAULT
        requested = max(1, requested)
        return min(MAX_STEPS_LIMIT, requested)

    def _build_steps(self, task: str, limit: int) -> list[str]:
        fragments = self._extract_fragments(task)
        steps: list[str] = []
        for fragment in fragments:
            if len(steps) >= limit:
                break
            normalized = self._normalize_fragment(fragment)
            if normalized:
                steps.append(normalized)
        if not steps:
            steps = [self._normalize_fragment(task)]
        return steps[:limit]

    @staticmethod
    def _extract_fragments(task: str) -> list[str]:
        candidates = [frag.strip() for frag in SENTENCE_SPLIT.split(task) if frag.strip()]
        if len(candidates) > 1:
            return candidates
        lines = [line.strip() for line in task.splitlines() if line.strip()]
        if lines:
            return lines
        return [task.strip()]

    @staticmethod
    def _normalize_fragment(fragment: str) -> str:
        snippet = " ".join(fragment.split())
        if not snippet:
            return ""
        snippet = textwrap.shorten(snippet, width=88, placeholder="...")
        if snippet[-1] not in ".!?":
            snippet += "."
        if len(snippet) == 1:
            return snippet.upper()
        return snippet[0].upper() + snippet[1:]
