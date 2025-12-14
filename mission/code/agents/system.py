"""Helpers for building the smolagent multi-agent orchestration."""

from __future__ import annotations

import logging
import gc
import torch
from transformers import BitsAndBytesConfig
from smolagents import CodeAgent, ToolCallingAgent, TransformersModel, WebSearchTool

from agents.web_tools import visit_webpage
from tools.act_tool import ALL_POLICY_TOOLS
from tools.curious_tool import CuriousTool
from tools.planner_tool import TaskPlannerTool
from tools.pose_mover_tool import SmolPoseMoverTool
from tools.smolvlm_inspect_tool import SmolVLMInspectTool
from inference.swappable_model import SwappableTransformersModel

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def build_transformers_model() -> SwappableTransformersModel:
    """Return a quantized SwappableTransformersModel."""

    # 4-bit config is mandatory for 14B model on 16GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Use the custom Swappable class
    model = SwappableTransformersModel(
        model_id=MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        # CRITICAL: DeepSeek R1 needs high token limits to "think" before answering.
        # 512 is too low; the model will cut off. 2048-4096 is recommended.
        max_new_tokens=4096, 
    )
    
    _ensure_pad_token(model)
    return model


def build_web_search_agent(model: TransformersModel) -> ToolCallingAgent:
    """Wrap web search + webpage visit helpers as a dedicated agent."""
    return ToolCallingAgent(
        tools=[WebSearchTool(), visit_webpage],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Search the web and summarize findings with visit_webpage.",
    )


def build_scene_agent(model: SwappableTransformersModel) -> ToolCallingAgent:
    """Combine SmolVLM inspection into a scene understanding agent."""

    # IMPORTANT: We pass the model to the tool here.
    # This allows the tool to say "Model.offload()" before it loads the heavy Vision model.
    # Ensure your SmolVLMInspectTool's __init__ accepts 'agent_model'.
    inspect_tool = SmolVLMInspectTool(agent_model=model)
    
    return ToolCallingAgent(
        tools=[inspect_tool],
        model=model,
        max_steps=6,
        name="scene_inspection_agent",
        description="Gather TOON-style scene context from the SO-101 cameras.",
    )


def build_robotics_agent(model: TransformersModel) -> ToolCallingAgent:
    """Group ACT policies with pose movers into a single robotics agent."""
    return ToolCallingAgent(
        tools=[tool_class() for tool_class in ALL_POLICY_TOOLS] + [SmolPoseMoverTool()],
        model=model,
        max_steps=4,
        name="robotics_agent",
        description="Run policy-specific ACT loops and move the SO-101 follower arm to saved poses.",
    )


def build_all_tools_agent(
    model: TransformersModel,
    *,
    instructions: str | None = None,
    stream_outputs: bool = True,
) -> ToolCallingAgent:
    """Debug-focused agent exposing every tool the CLI can use."""
    inspect_tool = SmolVLMInspectTool(agent_model=model)
    tools = [WebSearchTool(), visit_webpage, inspect_tool] + [
        tool_class() for tool_class in ALL_POLICY_TOOLS
    ] + [CuriousTool(), SmolPoseMoverTool()]
    return ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=16,
        name="all_tools_agent",
        description="Single agent that can search, inspect, and execute every policy/tool.",
        instructions=instructions,
        stream_outputs=stream_outputs,
    )


def build_manager_agent(
    *,
    model: TransformersModel,
    instructions: str,
    stream_outputs: bool,
    managed_agents: list[ToolCallingAgent],
) -> CodeAgent:
    """Assemble the manager CodeAgent that orchestrates the sub-agents."""
    return CodeAgent(
        tools=[TaskPlannerTool()],
        model=model,
        stream_outputs=stream_outputs,
        instructions=instructions,
        managed_agents=managed_agents,
        additional_authorized_imports=["os", "subprocess", "sys", "time", "numpy", "pandas"],
        name="manager_agent",
        description="Plans complex tasks, delegates work, and synthesizes multi-agent answers.",
    )


def _ensure_pad_token(model: TransformersModel) -> None:
    """Fix missing pad token for Qwen/Llama based models."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return
    
    # If pad token is valid, exit
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        return

    # Alias Pad to EOS (Safe fix for 4-bit models)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if hasattr(model.model, "config"):
        model.model.config.pad_token_id = tokenizer.eos_token_id
        
    logging.info("Pad token aliased to EOS (%s)", tokenizer.pad_token_id)
