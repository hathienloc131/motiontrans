# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MotionTrans is a robotics framework for human-to-robot motion transfer. It enables training manipulation policies (Diffusion Policy and Pi0-VLA) by co-training on both human VR demonstration data and robot teleoperation data. The core idea is that human hand motions captured via Meta Quest can be retargeted to robot actions, allowing policies to learn from human demonstrations.

## Environment Setup

```bash
conda create -n dexmimic python=3.10
conda activate dexmimic
pip install -r requirements.txt
pip install torch torchvision peft open3d viser
pip install huggingface-hub==0.21.4 pin==3.3.1 numpy==1.24.4
```

ZED SDK (for ZED2 camera) must be installed separately from the [official docs](https://www.stereolabs.com/docs/app-development/python/install).

## Key Commands

### Data Processing
```bash
# Convert raw human VR data → zarr format
bash scripts_data/zarr_human_data_conversion_batch.sh

# Convert raw robot teleoperation data → zarr format
bash scripts_data/zarr_robot_data_conversion_batch.sh

# Visualize processed data
bash scripts_data/zarr_human_data_conversion_vis.sh
bash scripts_data/data_visualization.sh
```

### Training
```bash
# Multi-task human-robot co-training (zero-shot setting)
bash scripts/dp_base_cotraining.sh

# Few-shot finetuning on robot demonstrations
bash scripts/dp_base_finetune_5demo.sh
bash scripts/dp_base_finetune_20demo.sh
```

Training uses `accelerate launch` with `bf16` mixed precision. Config is managed via Hydra; the config name `train_diffusion_unet_timm_hra_workspace` points to `diffusion_policy/config/`. Checkpoints and a `.yaml` task list are saved to `checkpoints/`.

### Inference / Deployment
```bash
# Diffusion Policy on real robot
bash scripts/dp_infer.sh

# Pi0-VLA (requires separate policy server)
bash scripts/pi0_infer.sh

# Replay processed human data on robot
bash scripts/replay.sh
```

### Robot Teleoperation
```bash
bash scripts/teleop_unimanual_frankainspire.sh
```

## Architecture

### Data Flow
```
Raw human VR data (Meta Quest)          Raw robot teleoperation data
        ↓                                         ↓
scripts_data/entry/zarr_human_data_*    scripts_data/entry/zarr_robot_data_*
        ↓                                         ↓
data/zarr_data/zarr_data_human/         data/zarr_data/zarr_data_robot/
        ↓                                         ↓
            HRACoTrainDataset (co-training)
                        ↓
              dp_train.py (Hydra + accelerate)
                        ↓
            diffusion_policy/workspace/train_diffusion_unet_hra_image_workspace.py
```

### Key Modules

- **`diffusion_policy/`** — Core ML framework (Hydra-configured). Key subdirs:
  - `config/` — Hydra YAML configs; `train_diffusion_unet_timm_hra_workspace.yaml` is the main co-training config
  - `dataset/hra_cotrain_dataset.py` — Co-training dataset: interleaves robot samples (embodiment=1, weight=alpha) with human samples (embodiment=0, weight=1-alpha); alpha=0.5 by default
  - `dataset/hra_dataset.py` — Base HRA (Human-Robot Action) dataset backed by zarr replay buffers
  - `policy/diffusion_unet_hra_timm_policy.py` — Main policy: UNet diffusion with TIMM vision encoder (DINOv2 ViT-B/14)
  - `workspace/train_diffusion_unet_hra_image_workspace.py` — Training loop, handles multi-task + co-training
  - `model/` — Vision encoders, diffusion UNet, embodiment adversarial network

- **`real/`** — Real robot interface:
  - `controller_robot_system.py` — Coordinates cameras, robot arm (Franka), and gripper (Inspire Hand)
  - `robot_franka.py`, `robot_inspire_hand.py` — Hardware drivers
  - `camera_zed.py`, `camera_realsense.py`, `camera_orbbec.py` — Camera drivers
  - `config/franka_inspire_atv_cam_unimanual.yaml` — Robot system config (IPs, calibration, latencies)

- **`human_data/`** — Human VR data collection:
  - `quest_recorder.py` — Records Meta Quest hand tracking data
  - `hand_retargeting.py` — Retargets human hand poses to robot gripper actions
  - `data_teleop_server.py` — Server for VR teleoperation

- **`common/`** — Shared utilities:
  - `replay_buffer.py` — Zarr-backed replay buffer (the core data structure)
  - `pose_util.py`, `pose_repr_util.py` — SE(3) pose math and representation conversion
  - `normalize_util.py` — Action/observation normalization
  - `real_inference_util.py` — Temporal ensemble inference helpers
  - `hra_sampler.py` — Sequence sampling for HRA datasets

- **`dp_train.py`** / **`dp_infer_real.py`** / **`dp_finetune.py`** — Top-level entry points

### Robot System Config
Hardware configuration lives in `real/config/*.yaml` (JSON format despite `.yaml` extension). Each config specifies cameras (ZED, RealSense, Orbbec), robot arm (Franka), gripper (Inspire Hand), calibration matrices, and latency parameters.

### Data Format
All processed data is stored as zarr arrays under `data/zarr_data/`. Each task gets its own zarr group containing observation sequences (images, low-dim states) and action sequences. The `replay_buffer.py` provides the unified interface.

### Inference Parameters
Critical inference parameters in `scripts/dp_infer.sh`:
- `robot_action_horizon` / `robot_steps_per_inference` — control action chunking for the arm
- `gripper_action_horizon` / `gripper_steps_per_inference` — control action chunking for the gripper
- `control_freq_downsample` — model inference frequency relative to control frequency (20Hz)
- `ensemble_steps` / `ensemble_weights_exp_k` — temporal ensemble settings
