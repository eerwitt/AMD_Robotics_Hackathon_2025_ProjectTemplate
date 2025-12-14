"""Smolagents tool wrapper around the ACT inference pipeline."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from lerobot.configs.policies import PreTrainedConfig
from smolagents import Tool

from inference.inferenceact import StandaloneACTDemoConfig, run as run_act

logger = logging.getLogger(__name__)

POLICY_LIBRARY = {
    "default": {
        "repo": "ichbinblau/so101_act_stack2cubes",
        "description": "Stack small cubes in a tight tower using the stack2cubes data",
        "when": "When the workspace shows individual cubes on the standard bench under even lighting",
        "toon": (
            "TOON\n"
            "policy[<=1]{name,description,when}\n"
            "  default,Stack small cubes in a tight tower using the stack2cubes data,When the workspace shows individual cubes on the standard bench under even lighting\n"
            "END TOON"
        ),
    },
    "kapla_first_level": {
        "repo": "LeTeamAMDHackhaton/KaplaTile0",
        "description": "A model used to stack the first level of wooden blocks on a purple background with a mat below",
        "when": "When you need to lay the base layer of Kapla-style blocks on a purple mat setup",
        "toon": (
            "TOON\n"
            "policy[<=1]{name,description,when}\n"
            "  kapla_first_level,A model used to stack the first level of wooden blocks on a purple background with a mat below,When you need to lay the base layer of Kapla-style blocks on a purple mat setup\n"
            "END TOON"
        ),
    },
}
class ACTPolicyTool(Tool):
    """Base helper that runs a fixed ACT policy."""

    output_type = "string"
    inputs = {
        "duration": {
            "type": "number",
            "description": "How long to run the inference loop (seconds).",
            "default": 10.0,
            "nullable": True,
        },
        "task": {
            "type": "string",
            "description": "Optional natural language task prompt for the policy.",
            "default": "",
            "nullable": True,
        },
    }
    outputs = {
        "finished": {
            "type": "boolean",
            "description": "True when the ACT loop completed without throwing.",
        },
        "details": {
            "type": "string",
            "description": "Summary of the run or error message on failure.",
        },
        "policy_selected": {
            "type": "string",
            "description": "Policy key that powered this run.",
        },
        "policy_description_toon": {
            "type": "string",
            "description": "TOON-formatted summary of what the selected policy is for.",
        },
    }

    policy_key: ClassVar[str]

    def forward(
        self,
        duration: float | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        cfg_kwargs: dict[str, Any] = {}
        if duration is not None and duration > 0:
            cfg_kwargs["duration"] = duration
        if task:
            cfg_kwargs["task"] = task

        policy_entry = POLICY_LIBRARY[self.policy_key]
        cfg_kwargs["policy"] = PreTrainedConfig.from_pretrained(policy_entry["repo"])
        cfg_kwargs["policy"].pretrained_path = policy_entry["repo"]
        cfg = StandaloneACTDemoConfig(**cfg_kwargs)

        try:
            run_act(cfg)
        except Exception as exc:  # pragma: no cover - rely on runtime behavior
            logger.exception("ACT inference failed")
            return {
                "finished": False,
                "details": f"ACT inference failed: {exc}",
            }

        summary = (
            f"ACT run finished task='{cfg.task}' "
            f"policy='{cfg.policy.pretrained_path if cfg.policy else 'unknown'}' "
            f"dataset='{cfg.dataset.repo_id}' duration={cfg.duration:.1f}s"
        )
        return {
            "finished": True,
            "details": summary,
            "policy_selected": self.policy_key,
            "policy_description_toon": policy_entry["toon"],
        }


class PerformActionDefault(ACTPolicyTool):
    name = "perform_action_default"
    policy_key = "default"
    description = (
        f"{POLICY_LIBRARY['default']['description']}. {POLICY_LIBRARY['default']['when']}."
    )


class PerformActionKaplaFirstLevel(ACTPolicyTool):
    name = "perform_action_kapla_first_level"
    policy_key = "kapla_first_level"
    description = (
        f"{POLICY_LIBRARY['kapla_first_level']['description']}. {POLICY_LIBRARY['kapla_first_level']['when']}."
    )


ALL_POLICY_TOOLS = [PerformActionDefault, PerformActionKaplaFirstLevel]
