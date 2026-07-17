
## GDT-DRL

Human-aware navigation with a **graph-first** state: LiDAR beams, the robot, and detected humans are distinct nodes in one heterogeneous graph, encoded by a Graph Attention Network (GAT) and controlled with TD3. Short-horizon motion cues are embedded in human nodes for proactive avoidance, and a value-grounded early-stopping rule freezes the GAT once a held-out value loss saturates so the policy can refine on a stable representation.

In simulation, GDT-DRL improves success and sharply reduces collisions versus recent graph-based crowd baselines (HEIGHT, HGAT-DRL) in cooperative and non-cooperative settings, with ablations on LiDAR resolution and trajectory features. The same graph is reconstructed on a Clearpath Husky A200 (Velodyne + ZED) for zero-shot transfer. At deployment the graph acts as a **digital-twin interface**: typed attention and behavioural-mode labels for operators, plus graph-mediated stuck recovery (local re-simulation, fine-tune, action replay) and light online adaptation.

<img width="458" height="360" alt="showcasev3" src="https://github.com/user-attachments/assets/eec62e9c-9878-4f2d-bb0c-cba729795bf2" />


<img width="1080" height="1080" alt="GDT-DRL_trained_crowdsim" src="https://github.com/user-attachments/assets/b1d004f3-b83b-418c-a477-48e869fc057f" />




## DT Recovery

[![Watch the demo](https://img.youtube.com/vi/KmYLyrCZc0Y/hqdefault.jpg)](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)

**Video:** [Watch demo](https://youtube.com/shorts/KmYLyrCZc0Y?feature=share)

