"""Reusable web browsing tools for the multi-agent workflow."""

from __future__ import annotations

import re

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Download a webpage and return its Markdown representation for downstream agents.

    Args:
        url: The URL to fetch.

    Returns:
        Markdownified content or an error message.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text or "").strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as exc:
        return f"Error fetching the webpage: {exc}"
    except Exception as exc:  # pragma: no cover - guard against unexpected encoding issues
        return f"An unexpected error occurred: {exc}"
