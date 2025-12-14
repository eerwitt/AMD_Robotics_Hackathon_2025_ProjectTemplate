"""CLI entrypoint wiring the single-agent tool flow."""

from __future__ import annotations

import argparse
import logging
from textwrap import dedent
from typing import Sequence

from agents import (
    MODEL_ID,
    build_all_tools_agent,
    build_transformers_model,
)

def _build_agent(stream_outputs: bool, instructions: str):
    """Build the single agent that exposes every tool."""

    model = build_transformers_model()
    return build_all_tools_agent(
        model,
        instructions=instructions,
        stream_outputs=stream_outputs,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the agent against a natural language task prompt."
    )
    parser.add_argument(
        "task",
        help="Instruction for the agent to solve using its toolset.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum number of agent/tool iterations to run.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming intermediate tool/agent outputs.",
    )
    return parser.parse_args(argv)


def _render_instructions() -> str:
    return dedent(
        """
        You orchestrate the SO-101 robot exclusively via the registered tools. Never write custom drivers or
        attempt to reach hardware directly, always call a tool.

        You're the CodeAgent coordinating a single toolkit agent:
          1. `all_tools_agent` can search the web, inspect the scene via SmolVLM, run ACT policies, and move the follower arm.

        Available tools are exposed through the `all_tools_agent` helper. Provide that tool with a clear `task` prompt (and optional `additional_args`) so it can choose between:
          * `capture_scene_inspection` – grab all camera feeds, emit TOON context, and return per-view descriptions.
          * `perform_action_default` / `perform_action_kapla_first_level` – execute the respective ACT policy.
          * `SmolPoseMoverTool` – move the follower arm to saved poses.
          * `WebSearchTool`, `visit_webpage` – research the scene or tasks online.
          * `curious` – wiggle the follower servo whenever the agent is confused or hits an unexpected error.

        To gather scene context, call `all_tools_agent(task="Inspect the cameras and summarize the scene in TOON")` instead of invoking `capture_scene_inspection()` directly; direct function calls and new helper definitions are not permitted.

        Ground rules:
        1. Reference TOON objects (label, normalized point_x/point_y, description) in your plans.
        2. Communicate with the robot solely through the provided tools.
        3. NOTE: You are a reasoning model. You may perform internal thinking, but your final output MUST be valid Python code using the tools provided.
        4. Do not emit JSON tool calls directly; always write Python that calls `all_tools_agent(task=..., additional_args={...})`.
        """
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    task_prompt = args.task
    instructions = _render_instructions()
    agent = _build_agent(stream_outputs=not args.no_stream, instructions=instructions)
    logging.info(f"Starting Agent with model: {MODEL_ID}")
    
    result = agent.run(
        task_prompt,
        max_steps=args.max_iterations,
    )
    if result is not None:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
