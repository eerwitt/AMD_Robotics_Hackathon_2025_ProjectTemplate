#!/usr/bin/env python3
"""CLI entrypoint for ACT inference leveraging inference.inferenceact."""

from lerobot.configs import parser

from inference.inferenceact import StandaloneACTDemoConfig, run


@parser.wrap()
def main(cfg: StandaloneACTDemoConfig) -> None:
    """Parse CLI flags and launch ACT inference."""
    run(cfg)


if __name__ == "__main__":
    main()
