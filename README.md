# GDT-NAV

Two stacks that share the same graph state.

- [GDT-DRL](GDT-DRL/) is the paper method: a typed LiDAR, robot, and human graph, a residual Graph Attention Network, and TD3. It trains in the Husky digital twin and deploys zero-shot on a Clearpath Husky A200 with a VLP-16 and a ZED 2i.
- [GDT-REC](GDT-REC/) is the digital-twin recovery layer used at deployment: live graph visualisation, stuck detection, local re-simulation, and a short recovery retrain.

Weights are not in this repository. Paper tables that were produced on another machine are listed in [GDT-DRL/results/MISSING.md](GDT-DRL/results/MISSING.md).

## Digital-twin recovery demo

[![Watch the demo](https://img.youtube.com/vi/KmYLyrCZc0Y/hqdefault.jpg)](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)

**Video:** [Watch demo](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)
