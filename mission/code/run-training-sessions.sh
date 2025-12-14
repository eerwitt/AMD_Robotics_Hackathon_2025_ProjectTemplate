#!/usr/bin/env bash
# Helper script for collecting baseline demonstrations for SmolVLA fine-tuning.
#
# Usage:
#   ./run-training-sessions.sh [dataset_repo]
#
# The script iterates through a curated set of language prompts and records
# one episode per prompt using lerobot's teleoperation stack. Adjust the
# PROMPTS array or any CLI flag below to match your setup before running.

set -euo pipefail

DATASET_REPO=${1:-"erik/so101_finetune_baseline"}
EPISODE_TIME=45
EPISODES_PER_PROMPT=1

PROMPTS=(
  "move forward to grab headphones"
  "pick up the can and lift it"
  "place the can on the platform"
  "move left around the obstacle"
  "open the gripper near the target"
)

CAMERAS='{"top": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}, "side": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, "front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}'

for prompt in "${PROMPTS[@]}"; do
  echo "\n=== Recording prompt: '$prompt' ==="
  lerobot-record \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes="$EPISODES_PER_PROMPT" \
    --dataset.episode_time_s="$EPISODE_TIME" \
    --dataset.single_task="$prompt" \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

done

echo "\nAll baseline prompts recorded into $DATASET_REPO"
