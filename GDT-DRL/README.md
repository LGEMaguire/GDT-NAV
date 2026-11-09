# GDT-DRL

Reference trainer, simulator, evaluator, and Husky deployment for the graph-first TD3 policy.

LiDAR beams, one robot node, and up to three human nodes share one graph (`graph_utils.py`). A two-layer GAT with a residual around the projected features feeds a TD3 actor and twin critics. Human nodes carry a short-horizon position from a Gaussian-process / constant-velocity blend. The actor update uses a detached GAT embedding. Held-out critic loss can freeze GAT updates (`--val-split`, patience 1000 checks) while TD3 keeps training.

LiDAR nodes are stored as 4 values. The last value is fixed at 0 so existing checkpoints still load. Robot nodes are 4 values and human nodes are 6.

## Setup

```bash
pip install -r requirements.txt
```

`torch==2.3.0` in this file is the CPU/PyPI pin. The machine this was trained on used `torch==2.3.0+cu118` with `torch-geometric==2.5.3`. Install the CUDA wheel that matches the GPU, then install `torch-geometric` against that build.

Simulation can run without ROS. `pygame_env_rosv5.py` reads Husky velocity limits from `husky_control` and `husky_description` when those packages are on the ROS path, and falls back if they are not.

Checkpoints are not in this repository. Training writes `models/TD3_gat_humans_{gat,actor,critic}.pth` and the target networks. Deployment looks in its own directory for `*_gat.pth` and `*_actor.pth`.

## Train

```bash
python gat_td3_train_pygamev5.py --steps 5000000 --eval-every 100000 --max-ep-len 500 --batch 128 --humans 3
```

Add `--render` for the pygame window. Logs `qvalues.log` and `gat_validation.log` are written in the working directory.

## Evaluate the reference environment

```bash
python test_gat_td3_v5.py --model-dir models --model-name TD3_gat_humans --episodes 100 --seed 42
```

This is the reference workspace (5 obstacles, 2 pedestrians). It is not the ORCA circle-crossing benchmark or the collision-course stress test. Those result files are in [results](results/).

## Husky

On the robot, forward the front 180 degree scan and odometry:

```bash
python lidarprocess.py
```

On the machine with the ZED 2i:

```bash
python cameraprocess.py
```

That node needs the ZED SDK (`pyzed`) and publishes `/person_detections`.

Then run the policy. It subscribes to `/husky/scan_raw`, `/husky/odom_raw`, and `/person_detections`, and publishes `/cmd_vel`:

```bash
python deploy_v5.py --goal-x X --goal-y Y
```

Place `TD3_gat_humans_gat.pth` and `TD3_gat_humans_actor.pth` next to the script, or pass `--model-dir`. ROS (`rospy`, `tf`, sensor and geometry messages) comes from the ROS install, not from `requirements.txt`.
