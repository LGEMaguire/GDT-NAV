# Results still to add

These files are not on this machine. Copy the logs or tables from the machine that ran the experiments into this folder. Do not add the sweep, ablation, ORCA, HEIGHT, or HGAT-DRL scripts.

The 50-episode reference logs in the training tree are not these tables, so they are not included.

- Section V, LiDAR resolution sweep (Table II, Fig. 4). One run each for L in {6, 10, 16, 20, 26, 32, 40}, 5e6 steps, final-window success and collision, and the evaluation curves.
- Section VI-A, horizon and blend screen (Table III). GP / constant-velocity grid, e95, e99, CVaR95, P(e > emax), I95, and the bootstrap intervals. The operating point used by the trainer is T = 0.3 s, lambda_CV = 0.9, embedded waypoint index 1.
- Section VI-B, future-position ablation (Table IV). Five seeds (0-4), 100 episodes each, for the 6-feature model, the direction cue (T = 0.01 s), and the 4-feature model, plus the paired tests.
- Section VII-A, ORCA circle crossing (Tables V-VI, Fig. 5). HEIGHT, HGAT-DRL, GDT-DRL zero-shot, and GDT-DRL trained. 100 paired episodes, seed 42, with McNemar intervals.
- Section VII-B, collision-course stress test (Tables VII-VIII, Fig. 6). Same comparators, goal threshold 0.5 m, 100 paired episodes, with McNemar intervals.
- Section VIII, Husky deployment (Table IX). Five trials each for static, crossing, and adversarial pedestrians: success, collision, timeout, navigation time, minimum LiDAR range, and closest human. The local `logs/deploy_runs/deploy_20260905_*` and `deploy_20260922_*` folders are a different set and are not this table.
