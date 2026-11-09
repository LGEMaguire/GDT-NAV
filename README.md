## GDT-NAV

GDT-NAV is for human-aware navigation of mobile robots in human-populated environments. The robot pursues a goal, avoids static obstacles and responds to pedestrian motion, in simulation and in zero-shot deployment on a physical robot. Against other navigation frameworks it achieves higher observed success and lower observed collision rates in dense crowds and in a collision-course stress test, with strong zero-shot transfer.

The framework is a graph-first digital twin. At each control cycle, LiDAR beams, the robot state and detected humans form distinct node types in a single heterogeneous graph. Short-horizon future-position features augment the human nodes so the policy can reason proactively rather than purely reactively. A graph attention network processes that graph, and the resulting embedding drives a Twin Delayed Deep Deterministic Policy Gradient (TD3) controller.



<img width="520" height="520" alt="GDT-DRL_trained_crowdsim" src="https://github.com/user-attachments/assets/b1d004f3-b83b-418c-a477-48e869fc057f" />

## DT Recovery Demo

[![Watch the demo](https://img.youtube.com/vi/KmYLyrCZc0Y/hqdefault.jpg)](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)

**Video:** [Watch demo](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)

## Deployment Demo

[![Watch the deployment](https://img.youtube.com/vi/6QW1z5_KnqM/hqdefault.jpg)](https://youtube.com/shorts/6QW1z5_KnqM?feature=share)

**Video:** [Watch deployment](https://youtube.com/shorts/6QW1z5_KnqM?feature=share)
