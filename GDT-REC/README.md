# GDT-REC

Digital-twin view and stuck recovery for a Husky already running the GDT-DRL policy.

`graph_utils.py` and `lidar_front.py` are copies of the GDT-DRL modules so this folder can be launched on its own.

## Setup

```bash
pip install -r requirements.txt
```

Use the same PyTorch build as GDT-DRL (`torch==2.3.0+cu118` and `torch-geometric==2.5.3` on the training machine). ROS (`rospy` and the message packages) is installed separately.

Weights stay outside the repo. Put `TD3_gat_humans_gat.pth` and `TD3_gat_humans_actor.pth` in this directory, or pass `--model-dir`.

## Live twin

`dt_v5.py` subscribes to `/husky/scan_raw`, `/husky/odom_raw`, and `/person_detections` and draws the robot-centred graph.

```bash
python dt_v5.py --model-dir . --model-name TD3_gat_humans
```

Optional `--goal-x` and `--goal-y` set the initial goal marker.

## Recovery deployment

`deploy_v5_recovery.py` runs the same policy, and on a stuck detection it snapshots the scan, retrains a clone in `stuck_recovery.py`, then yaws to the logged heading and blends the logged actions with the live policy up to the first waypoint.

```bash
python deploy_v5_recovery.py --goal-x X --goal-y Y
```

`--visualize-recovery` opens the recovery window. `--no-recovery` is policy only. `--enable-planning` turns on continuous lookahead.

## Off-robot laptop

`bridge.py` relays `/husky/scan_raw`, `/husky/odom_raw`, and `/person_detections` from the Husky rosbridge websocket onto the laptop ROS master. The Husky address is set in `bridge.py` (`ws://192.168.131.224:9090`). Change that URL before using it on another network.

```bash
python bridge.py
```
