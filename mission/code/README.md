# amd-hackathon-2025

Agentic workflow for using the so101 with act or vla policies.

## General Notes

sudo chmod 666 /dev/ttyACM*
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm

pip install gsplat --index-url=https://pypi.amd.com/simple
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm6.4


https://github.com/ByteDance-Seed/Depth-Anything-3.git

ffplay /dev/video0 # Webcam
ffplay /dev/video2 # Claw
ffplay /dev/video4 # Mobile stick
ffplay /dev/video6 # Top down


lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, claw: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true


lerobot-record   --dataset.repo_id=erik/so101_test_dataset   --dataset.push_to_hub=false   --robot.type=so101_follower   --robot.port=/dev/ttyACM1   --robot.id=my_awesome_follower_arm   --robot.cameras="{top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM0   --teleop.id=my_awesome_leader_arm --dataset.single_task="3d scan can" --dataset.num_episodes=1 --dataset.episode_time_s=10 --display_data=true

# Run the vla
lerobot-record   --robot.type=so101_follower   --robot.port=/dev/ttyACM1   --robot.id=my_awesome_follower_arm   --robot.cameras="{camera1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"   --policy.path=lerobot/smolvla_base   --dataset.single_task="move to the left"   --dataset.repo_id=erik/eval_so101_smolvla   --dataset.num_episodes=1   --dataset.episode_time_s=30   --display_data=true

## Single-agent orchestration

The CLI now boots a single **toolkit agent** (`all_tools_agent`) via `CodeAgent`:

1. `all_tools_agent` – a `ToolCallingAgent` exposing `WebSearchTool`, `visit_webpage`, `SmolVLMInspectTool`, each of the `perform_action_*` ACT tools, `curious`, and `SmolPoseMoverTool` so a single worker can search, sense, plan, and act.

Legacy builders (`web_search_agent`, `scene_inspection_agent`, `robotics_agent`) are retained for experimentation, but every run now speaks directly to `all_tools_agent`. Always call that helper (e.g., `all_tools_agent(task="Inspect the cameras and summarize the scene in TOON")`); direct function or module calls such as `capture_scene_inspection()` are blocked by guardrails.

`visit_webpage` uses `requests` + `markdownify` to turn HTML into Markdown so the web agent can stay focused on relevant content; these packages have been added to `requirements.txt`.

## ACT policy choices

Each policy now lives behind its own tool (`perform_action_default` and `perform_action_kapla_first_level`). They accept the usual `duration`/`task` inputs and return `finished`, `details`, `policy_selected`, and `policy_description_toon`. Use the corresponding TOON summaries when deciding which version of the stack or Kapla demo you need next.

```
default:
TOON
policy[<=1]{name,description,when}
  default,Stack small cubes in a tight tower using the stack2cubes data,When the workspace shows individual cubes on the standard bench under even lighting
END TOON
```

```
kapla_first_level:
TOON
policy[<=1]{name,description,when}
  kapla_first_level,A model used to stack the first level of wooden blocks on a purple background with a mat below,When you need to lay the base layer of Kapla-style blocks on a purple mat setup
END TOON
```

Add more entries to `tools/act_tool.py` when you train or want to describe additional policies.

## SmolVLM inspect context

When you need a textual sense of what each SO-101 camera is seeing, call `capture_scene_inspection` before executing other tools. It mirrors the logic from `test-smolvlm-inspect.py`, captures the top/side/front feeds, runs SmolVLM descriptions, and returns both a TOON-style `context` string (for use by the agent on the next iterations) and a verbose `descriptions_json` payload. The CLI no longer performs any capture on startup; request this tool explicitly whenever fresh scene context is required.

## SmolVLM segmentation inference

To highlight what SmolVLM thinks is the object of interest in a single image, call the helper in `inference/smolvlm_segmentation.py`. It loads `HuggingFaceTB/SmolVLM-Instruct`, asks the model to return normalized coordinates in JSON, then overlays a point and caption on top of the source image.

```bash
python -m inference.smolvlm_segmentation \
    path/to/frame.png \
    --prompt "Highlight the red block the gripper should grab" \
    --output outputs/smolvlm/red-block.png
```

The resulting file includes the highlighted point plus the model's short reasoning so you can visualize what the VLM is focusing on.

If you want to capture frames directly from the SO-101 camera layout (top/side/front indices 6/4/2) and annotate each feed, use:

```bash
python test-smolvlm-seg.py --prompt "Highlight the target object"
```

The script waits 3 seconds, snaps all three cameras, runs SmolVLM segmentation per view, and drops annotated frames into `outputs/smolvlm/cameras/`. SmolVLM returns objects using the [TOON](https://github.com/toon-format/toon) table format (`objects[<=3]{label,point_x,point_y,description}`) so downstream tools can parse multiple candidates deterministically.

For a purely descriptive pass without segmentation, run:

```bash
python test-smolvlm-inspect.py --prompt "Describe what the arm is doing and any obstacles"
```

This captures the same camera feeds, but instead of annotating images it prints the VLM’s textual description for each view.

## Wiggle helper

If you want to exercise the `curious` tool manually, run:

```bash
python test-wiggle.py
```

The script centers the follower servo, executes a three-loop wiggle (±200 steps), and re-centers it so you can verify the actuator is responsive before invoking additional policies.

```py
from PIL import Image
import os

# 1. Create a directory to store them
output_dir = "debug_frames"
os.makedirs(output_dir, exist_ok=True)

# 2. Loop and save
print(f"Saving {len(all_frames)} frames to '{output_dir}/'...")

for i, frame_arr in enumerate(all_frames):
    # frame_arr is a numpy array (RGB)
    img = Image.fromarray(frame_arr)

    # Save as PNG (lossless) or JPG
    filename = os.path.join(output_dir, f"frame_{i:04d}.png")
    img.save(filename)

print("Done!")
```

```py
import torch
import torchvision

def extract_frames_torch(video_path, target_fps=1.0):
    # read_video returns (video_frames, audio_frames, metadata)
    # video_frames is [T, H, W, C] in 0-255 range
    vframes, _, info = torchvision.io.read_video(video_path, pts_unit='sec')

    orig_fps = info['video_fps']

    # Calculate step size to achieve target FPS
    step = int(orig_fps / target_fps)
    if step < 1: step = 1

    # Slice the tensor to get the frames
    selected_frames = vframes[::step]

    # Convert to list of numpy arrays if you need exactly what your old function returned
    return [f.numpy() for f in selected_frames]
```

```py
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
model = model.to(device)
```

```py
import numpy as np
video_paths = [
    "/home/erik/.cache/huggingface/lerobot/erik/so101_test_dataset/videos/observation.images.front/chunk-000/file-000.mp4",
    "/home/erik/.cache/huggingface/lerobot/erik/so101_test_dataset/videos/observation.images.side/chunk-000/file-000.mp4",
    "/home/erik/.cache/huggingface/lerobot/erik/so101_test_dataset/videos/observation.images.top/chunk-000/file-000.mp4"
]
all_frames = []

for video_path in video_paths:
    frames = extract_frames_torch(video_path, target_fps=2.0)  # Extract 2 frames per second  
    all_frames.extend(frames)

num_frames = len(all_frames)  
resolution = 256  # Reduced from 504  
intrinsics = np.array([  
    [[250, 0, 128], [0, 250, 128], [0, 0, 1]]  # Adjusted for 256x256  
] * num_frames, dtype=np.float32) 
```

```py
prediction = model.inference(
    image=all_frames,
    infer_gs=True,
    export_dir="./output",
    intrinsics=intrinsics,
    export_format="gs_ply-gs_video",  # Export GLB for Blender + depth visualizations  
    process_res=256,
    process_res_method="lower_bound_resize",  # Ensure max dimension is 256  
    align_to_input_ext_scale=True  
)

print(f"GLB file saved to: ./output/scene.glb")
print(f"Depth shape: {prediction.depth.shape}")
print(f"Import scene.glb into Blender to view the 3D reconstruction")
```
