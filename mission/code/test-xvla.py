#!/usr/bin/env python3
"""CLI entrypoint for XVLA inference leveraging inference.inferencevla."""

from lerobot.configs import parser

import inference.inferencevla as inference_vla
from inference.inferencevla import StandaloneRTCDemoConfig, run

_DEFAULT_XVLA_POLICY = "lerobot/xvla-base"
inference_vla.DEFAULT_POLICY_PATH = _DEFAULT_XVLA_POLICY


@parser.wrap()
def main(cfg: StandaloneRTCDemoConfig) -> None:
    """Parse CLI flags and launch XVLA inference."""

    run(cfg)


if __name__ == "__main__":
    main()
