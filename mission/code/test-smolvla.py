#!/usr/bin/env python3
"""CLI entrypoint for SmolVLA inference leveraging inference.inferencevla."""

from lerobot.configs import parser

from inference.inferencevla import StandaloneRTCDemoConfig, run


@parser.wrap()
def main(cfg: StandaloneRTCDemoConfig) -> None:
    """Parse CLI flags and launch SmolVLA inference."""
    run(cfg)


if __name__ == "__main__":
    main()
