#!/usr/bin/env python3

"""
GAT-TD3 Real Husky Deployment Script with Stuck Recovery and Continuous Planning
Runs trained GAT-TD3 navigation agent on Clearpath Husky A200
Integrates stuck detection, environment snapshot, quick retraining, and recovery deployment

Features:
- Normal GAT-TD3 navigation
- Automatic stuck detection (velocity, progress, oscillation)
- LiDAR-based environment snapshot
- Quick model retraining for recovery
- Yaw to the logged set-off heading, then blend logged actions with live
  DRL (humans included) up to the first waypoint, then the original policy
- Light weight update to original model after successful recovery
- Continuous lookahead planning (runs ahead of robot)
- Quality-gated continuous learning (micro-updates during success)
- Path visualization for explainability (publishes planned path)
"""

import os
import csv
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Float32MultiArray, Float32, String
from visualization_msgs.msg import Marker, MarkerArray
import tf
import math
import time
import threading
import json
from collections import deque

from graph_utils import create_edge_index
from lidar_front import BEAM_ANGLES, sample_sim_beams
from stuck_recovery import (
    StuckRecoveryController,
    StuckDetector,
    EnvironmentSnapshotBuilder,
    LIDAR_BEAMS, MAX_HUMANS, LIDAR_RANGE, LOGGED_ACTION_BLEND
)

# ==================== CONFIGURATION ====================
CONTROL_RATE = 10  # Hz
GOAL_REACHED_DIST = 0.75  # meters
COLLISION_DIST = 0.5  # matches pygame_env_rosv5.py and the Sept deploy logs
TURN_W_THRESH = 0.05  # rad/s, same "turning" bar as deploy_v5
FRONT_BEAM_SLICE = slice(8, 13)  # central 5 of 20 beams
# Same observation scaling as deploy_v5.py / pygame_env_rosv5.py
GOAL_NORM = 20.0
CAM_MAX = 15.0
GAT_PRED_HORIZON = float(np.linspace(0.1, 2.0, 10)[1])

# Husky velocity limits
MAX_LINEAR_VEL = 1.0  # m/s
MAX_ANGULAR_VEL = 2.0  # rad/s

# Recovery configuration
RECOVERY_ENABLED = True
RECOVERY_MAX_EPISODES = 50
RECOVERY_LIGHT_UPDATE_BLEND = 0.03  # 3% of recovery weights

# Continuous planning configuration
CONTINUOUS_PLANNING_ENABLED = True
PLANNING_INTERVAL = 0.5  # seconds between plans
PLANNING_HORIZON = 15  # steps to simulate ahead (keep small for speed)

# Continuous learning configuration
CONTINUOUS_LEARNING_ENABLED = True
MICRO_UPDATE_BLEND = 0.005  # 0.5% micro updates

# Visualization configuration
PUBLISH_PLANNED_PATH = True
PATH_PUBLISH_RATE = 2.0  # Hz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _human_csv_fields():
    fields = []
    for i in range(MAX_HUMANS):
        fields.extend([
            f'h{i}_x_rel', f'h{i}_y_rel', f'h{i}_speed_norm',
            f'h{i}_conf', f'h{i}_pred_x', f'h{i}_pred_y', f'h{i}_dist',
        ])
    return fields


STEP_CSV_FIELDS = [
    'step', 't',
    'x', 'y', 'theta',
    'goal_dist', 'goal_bearing',
    'dx_ego_norm', 'dy_ego_norm', 'v_odom', 'w_odom',
    'a0_tanh', 'a1_tanh', 'v_norm', 'w_norm', 'v_cmd', 'w_cmd',
    'min_lidar', 'forward_clearance', 'lidar_std',
    'collision', 'emergency_stop',
    'n_humans', 'closest_human',
    'mode', 'action_frame', 'recovery_id',
    'wp_dist', 'heading_err', 'replay_idx',
    'logged_v', 'logged_w', 'drl_a0', 'drl_a1',
    'estop_suppressed', 'n_humans_seen',
] + _human_csv_fields() + [f'lidar_{i}' for i in range(LIDAR_BEAMS)]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


class RecoveryRunLogger:
    """Same step/summary logs as deploy_v5, plus recovery artifacts for the paper."""

    def __init__(self, gat_path, actor_path, goal_x, goal_y):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'logs', 'deploy_runs')
        self.run_dir = os.path.join(log_dir, f'deploy_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, 'steps.csv')
        self.summary_json_path = os.path.join(self.run_dir, 'summary.json')
        self.summary_txt_path = os.path.join(self.run_dir, 'summary.txt')
        self.events_path = os.path.join(self.run_dir, 'events.jsonl')
        self.header = {
            'domain': 'real',
            'script': 'deploy_v5_recovery',
            'timestamp': timestamp,
            'gat_model': os.path.basename(gat_path),
            'actor_model': os.path.basename(actor_path),
            'control_rate_hz': CONTROL_RATE,
            'goal_reached_dist': GOAL_REACHED_DIST,
            'collision_dist': COLLISION_DIST,
            'v_max': MAX_LINEAR_VEL,
            'w_max': MAX_ANGULAR_VEL,
            'goal_norm': GOAL_NORM,
            'cam_max': CAM_MAX,
            'gat_pred_horizon': GAT_PRED_HORIZON,
            'logged_action_blend': float(LOGGED_ACTION_BLEND),
            'goal_x': float(goal_x),
            'goal_y': float(goal_y),
        }
        self.steps = []
        self.events = []
        self.attempts = []
        self.start_time = None
        self.start_x = None
        self.start_y = None
        self.start_theta = None
        self.start_goal_dist = None
        self.path_length = 0.0
        self.prev_x = None
        self.prev_y = None
        self.e_stop_steps = 0
        self.collision_steps = 0
        self.stale_skips = 0
        self.finalized = False
        self._t0 = time.time()
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=STEP_CSV_FIELDS, extrasaction='ignore', restval=''
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()
        self._events_file = open(self.events_path, 'w')
        self.event('run_start', goal_x=float(goal_x), goal_y=float(goal_y),
                   gat_model=self.header['gat_model'], actor_model=self.header['actor_model'])

    def event(self, name, **payload):
        record = {'t': time.time() - self._t0, 'event': name, **payload}
        self.events.append(record)
        self._events_file.write(json.dumps(record, default=_json_default) + '\n')
        self._events_file.flush()

    def mark_start(self, x, y, theta, goal_dist):
        self.start_time = time.time()
        self.start_x = float(x)
        self.start_y = float(y)
        self.start_theta = float(theta)
        self.start_goal_dist = float(goal_dist)
        self.prev_x = float(x)
        self.prev_y = float(y)
        self.event('control_start', x=self.start_x, y=self.start_y,
                   theta=self.start_theta, goal_dist=self.start_goal_dist)

    def note_stale_skip(self):
        self.stale_skips += 1

    def log_step(self, row):
        if self.prev_x is not None:
            self.path_length += math.hypot(row['x'] - self.prev_x, row['y'] - self.prev_y)
        self.prev_x = row['x']
        self.prev_y = row['y']
        if row.get('emergency_stop'):
            self.e_stop_steps += 1
        if row.get('collision'):
            self.collision_steps += 1
        self.steps.append(row)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save_attempt(self, attempt, wall_s):
        """Write one stuck-recovery retrain: episodes, action log, snapshot, trajectory."""
        idx = len(self.attempts) + 1
        tag = f'recovery_{idx:03d}'
        record = dict(attempt)
        record['train_wall_s'] = float(wall_s)
        record['index'] = idx
        self.attempts.append({
            'index': idx,
            'success': bool(record.get('success')),
            'train_wall_s': float(wall_s),
            'n_episodes': len(record.get('episodes') or []),
            'n_actions': len(record.get('action_log') or []),
        })
        with open(os.path.join(self.run_dir, f'{tag}.json'), 'w') as f:
            json.dump(record, f, indent=2, default=_json_default)
        episodes = record.get('episodes') or []
        if episodes:
            ep_path = os.path.join(self.run_dir, f'{tag}_episodes.csv')
            with open(ep_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
                writer.writeheader()
                writer.writerows(episodes)
        actions = record.get('action_log') or []
        if actions:
            act_path = os.path.join(self.run_dir, f'{tag}_actions.csv')
            flat = []
            for a in actions:
                flat.append({
                    'step': a['step'],
                    'segment': a.get('segment', ''),
                    'x': a['x'],
                    'y': a['y'],
                    'theta': a['theta'],
                    'v': a['action'][0],
                    'w': a['action'][1],
                })
            with open(act_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['step', 'segment', 'x', 'y', 'theta', 'v', 'w']
                )
                writer.writeheader()
                writer.writerows(flat)
        traj = record.get('trajectory') or []
        if traj:
            with open(os.path.join(self.run_dir, f'{tag}_trajectory.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                writer.writerows(traj)
        self.event('recovery_train_done', index=idx, success=bool(record.get('success')),
                   train_wall_s=float(wall_s), n_episodes=len(episodes),
                   n_actions=len(actions))

    def _aggregate(self, outcome, final_x, final_y, final_theta, final_goal_dist):
        elapsed = (time.time() - self.start_time) if self.start_time else 0.0
        n = len(self.steps)
        base = {
            **self.header,
            'outcome': outcome,
            'n_steps': n,
            'elapsed_s': elapsed,
            'stale_sensor_skips': self.stale_skips,
            'recovery_attempts': len(self.attempts),
            'recovery_trains_succeeded': sum(1 for a in self.attempts if a['success']),
            'run_dir': self.run_dir,
        }
        if n == 0:
            base['note'] = 'no control steps recorded'
            return base

        min_lidars = np.array([s['min_lidar'] for s in self.steps], dtype=np.float64)
        v_cmds = np.array([s['v_cmd'] for s in self.steps], dtype=np.float64)
        w_cmds = np.array([s['w_cmd'] for s in self.steps], dtype=np.float64)
        v_odoms = np.array([s['v_odom'] for s in self.steps], dtype=np.float64)
        closest_h = np.array([s['closest_human'] for s in self.steps], dtype=np.float64)
        n_humans = np.array([s['n_humans'] for s in self.steps], dtype=np.float64)
        modes = [s.get('mode', 'NORMAL') for s in self.steps]
        dt = 1.0 / CONTROL_RATE
        dv = np.diff(v_cmds) if n > 1 else np.array([0.0])
        dw = np.diff(w_cmds) if n > 1 else np.array([0.0])
        finite_human = closest_h[np.isfinite(closest_h)]
        start_goal = self.start_goal_dist if self.start_goal_dist else 0.0
        efficiency = (start_goal / self.path_length) if self.path_length > 1e-6 else 0.0
        mode_counts = {m: modes.count(m) for m in ('NORMAL', 'ALIGN', 'BLEND', 'GOTO', 'RECOVERY')}
        mode_counts['BLEND'] += mode_counts['GOTO']

        base.update({
            'time_to_goal_s': elapsed if outcome == 'SUCCESS' else None,
            'start_x': self.start_x,
            'start_y': self.start_y,
            'start_theta': self.start_theta,
            'start_goal_dist': start_goal,
            'final_x': float(final_x),
            'final_y': float(final_y),
            'final_theta': float(final_theta),
            'final_goal_dist': float(final_goal_dist),
            'path_length_m': self.path_length,
            'path_efficiency': efficiency,
            'mean_v_cmd': float(np.mean(v_cmds)),
            'max_v_cmd': float(np.max(v_cmds)),
            'mean_w_cmd': float(np.mean(np.abs(w_cmds))),
            'max_w_cmd': float(np.max(np.abs(w_cmds))),
            'mean_v_odom': float(np.mean(v_odoms)),
            'std_dv_cmd': float(np.std(dv)),
            'std_dw_cmd': float(np.std(dw)),
            'frac_turning': float(np.mean(np.abs(w_cmds) > TURN_W_THRESH)),
            'mean_clearance': float(np.mean(min_lidars)),
            'min_clearance': float(np.min(min_lidars)),
            'p5_clearance': float(np.percentile(min_lidars, 5)),
            'time_clearance_lt_1_0_s': float(np.mean(min_lidars < 1.0) * n * dt),
            'time_clearance_lt_0_7_s': float(np.mean(min_lidars < 0.7) * n * dt),
            'time_clearance_lt_0_5_s': float(np.mean(min_lidars < COLLISION_DIST) * n * dt),
            'collision_steps': self.collision_steps,
            'e_stop_steps': self.e_stop_steps,
            'mean_n_humans': float(np.mean(n_humans)),
            'max_n_humans': int(np.max(n_humans)),
            'closest_human_m': float(np.min(finite_human)) if len(finite_human) else None,
            'time_human_lt_2_0_s': float(np.mean(closest_h < 2.0) * n * dt) if len(finite_human) else 0.0,
            'steps_normal': mode_counts['NORMAL'],
            'steps_align': mode_counts['ALIGN'],
            'steps_blend': mode_counts['BLEND'],
            'steps_recovery_policy': mode_counts['RECOVERY'],
            'time_normal_s': mode_counts['NORMAL'] * dt,
            'time_align_s': mode_counts['ALIGN'] * dt,
            'time_blend_s': mode_counts['BLEND'] * dt,
            'time_recovery_policy_s': mode_counts['RECOVERY'] * dt,
            'recovery_attempts_detail': self.attempts,
        })
        return base

    def _format_txt(self, summary):
        lines = [
            'GAT-TD3 real-Husky recovery run summary (sim2real)',
            '=' * 52,
            f"Outcome:            {summary['outcome']}",
            f"Steps:              {summary['n_steps']}",
            f"Elapsed:            {summary['elapsed_s']:.2f} s",
            f"Start:              ({summary.get('start_x', 0):.2f}, {summary.get('start_y', 0):.2f})",
            f"Goal:               ({summary['goal_x']:.2f}, {summary['goal_y']:.2f})",
            f"Start-goal dist:    {summary.get('start_goal_dist', 0):.2f} m",
            f"Final goal dist:    {summary.get('final_goal_dist', 0):.2f} m",
            f"Path length:        {summary.get('path_length_m', 0):.2f} m",
            f"Path efficiency:    {summary.get('path_efficiency', 0):.3f}  (start-goal / path)",
            f"Mean / min / p5 clearance: "
            f"{summary.get('mean_clearance', 0):.2f} / "
            f"{summary.get('min_clearance', 0):.2f} / "
            f"{summary.get('p5_clearance', 0):.2f} m",
            f"Time clearance <1.0/0.7/0.5 m: "
            f"{summary.get('time_clearance_lt_1_0_s', 0):.2f} / "
            f"{summary.get('time_clearance_lt_0_7_s', 0):.2f} / "
            f"{summary.get('time_clearance_lt_0_5_s', 0):.2f} s",
            f"Collision steps:    {summary.get('collision_steps', 0)}",
            f"E-stop steps:       {summary.get('e_stop_steps', 0)}",
            f"Mean / max v_cmd:   {summary.get('mean_v_cmd', 0):.3f} / {summary.get('max_v_cmd', 0):.3f} m/s",
            f"Mean |w_cmd| / max: {summary.get('mean_w_cmd', 0):.3f} / {summary.get('max_w_cmd', 0):.3f} rad/s",
            f"std(Δv), std(Δw):   {summary.get('std_dv_cmd', 0):.4f}, {summary.get('std_dw_cmd', 0):.4f}",
            f"Fraction turning:   {summary.get('frac_turning', 0):.3f}",
            f"Humans mean / max:  {summary.get('mean_n_humans', 0):.2f} / {summary.get('max_n_humans', 0)}",
            f"Closest human:      {summary.get('closest_human_m')}",
            f"Time human <2 m:    {summary.get('time_human_lt_2_0_s', 0):.2f} s",
            f"Stale sensor skips: {summary.get('stale_sensor_skips', 0)}",
            f"Recovery trains:    {summary.get('recovery_trains_succeeded', 0)}/"
            f"{summary.get('recovery_attempts', 0)} succeeded",
            f"Steps N/A/B/R:      {summary.get('steps_normal', 0)} / "
            f"{summary.get('steps_align', 0)} / {summary.get('steps_blend', 0)} / "
            f"{summary.get('steps_recovery_policy', 0)}",
            f"Models:             {summary.get('gat_model')} / {summary.get('actor_model')}",
            f"Step log:           {self.csv_path}",
            f"Events:             {self.events_path}",
            f"Summary JSON:       {self.summary_json_path}",
        ]
        return '\n'.join(lines) + '\n'

    def finalize(self, outcome, final_x, final_y, final_theta, final_goal_dist):
        if self.finalized:
            return
        self.finalized = True
        self.event('run_end', outcome=outcome, x=float(final_x), y=float(final_y),
                   theta=float(final_theta), goal_dist=float(final_goal_dist))
        summary = self._aggregate(outcome, final_x, final_y, final_theta, final_goal_dist)
        for handle in (self._csv_file, self._events_file):
            try:
                handle.close()
            except Exception:
                pass

        def _safe(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        summary = {k: _safe(v) for k, v in summary.items()}
        with open(self.summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        txt = self._format_txt(summary)
        with open(self.summary_txt_path, 'w') as f:
            f.write(txt)
        print('\n' + txt)
        try:
            rospy.loginfo(f"Run logs written to {self.run_dir}")
        except Exception:
            pass
        return summary


# ==================== NEURAL NETWORK MODELS ====================
class GATNetwork(nn.Module):
    """Graph Attention Network for heterogeneous graph"""
    
    def __init__(self, lidar_channels=4, robot_channels=4, human_channels=6,
                 out_channels=16, num_heads=4, lidar_nodes=20, robot_nodes=1, human_nodes=3):
        super().__init__()
        self.L = lidar_nodes
        self.R = robot_nodes
        self.H = human_nodes
        
        self.lidar_proj = nn.Linear(lidar_channels, out_channels)
        self.robot_proj = nn.Linear(robot_channels, out_channels)
        self.human_proj = nn.Linear(human_channels, out_channels)
        
        self.g1 = GATConv(out_channels, out_channels, heads=num_heads, concat=True)
        self.g2 = GATConv(out_channels * num_heads, out_channels, heads=num_heads, concat=False)
    
    def forward(self, lidar_nodes, robot_nodes, human_nodes, edge_index):
        lidar_proj = self.lidar_proj(lidar_nodes)
        robot_proj = self.robot_proj(robot_nodes)
        human_proj = self.human_proj(human_nodes)
        
        x = torch.cat([lidar_proj, robot_proj, human_proj], dim=0)
        x_init = x.clone()
        
        x = F.elu(self.g1(x, edge_index))
        x = self.g2(x, edge_index)
        x = x + x_init
        
        return x


class Actor(nn.Module):
    """Policy network"""
    
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, action_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))


# ==================== MODEL AUTO-DISCOVERY ====================
def find_model_files(search_dir=None):
    """Find GAT and Actor model files"""
    if search_dir is None:
        search_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Searching for model files in: {search_dir}")
    
    gat_pattern = os.path.join(search_dir, "*_gat.pth")
    actor_pattern = os.path.join(search_dir, "*_actor.pth")
    
    gat_files = glob.glob(gat_pattern)
    actor_files = glob.glob(actor_pattern)
    
    print(f"Found {len(gat_files)} GAT model file(s)")
    print(f"Found {len(actor_files)} Actor model file(s)")
    
    if not gat_files or not actor_files:
        all_pth = glob.glob(os.path.join(search_dir, "*.pth"))
        if all_pth:
            print(f"Available .pth files: {[os.path.basename(f) for f in all_pth]}")
        return None, None
    
    gat_path = max(gat_files, key=os.path.getmtime)
    actor_path = max(actor_files, key=os.path.getmtime)
    
    print(f"Selected GAT model: {os.path.basename(gat_path)}")
    print(f"Selected Actor model: {os.path.basename(actor_path)}")
    
    return gat_path, actor_path


# ==================== TRAJECTORY PREDICTOR ====================
class SimpleTrajectoryPredictor:
    """Lightweight trajectory prediction for humans"""
    
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.person_histories = {}
    
    def update(self, tracking_id, x, y):
        current_time = time.time()
        if tracking_id not in self.person_histories:
            self.person_histories[tracking_id] = deque(maxlen=self.history_size)
        self.person_histories[tracking_id].append([x, y, current_time])
    
    def predict(self, tracking_id, dt=1.0):
        if tracking_id not in self.person_histories:
            return 0.0, 0.0, 0.0
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            if len(history) == 1:
                return history[-1][0], history[-1][1], 0.5
            return 0.0, 0.0, 0.0
        
        recent = list(history)[-5:]
        vx = (recent[-1][0] - recent[0][0]) / max(0.1, recent[-1][2] - recent[0][2])
        vy = (recent[-1][1] - recent[0][1]) / max(0.1, recent[-1][2] - recent[0][2])
        
        pred_x = recent[-1][0] + vx * dt
        pred_y = recent[-1][1] + vy * dt
        confidence = min(1.0, len(history) / self.history_size)
        
        return pred_x, pred_y, confidence
    
    def get_speed(self, tracking_id):
        if tracking_id not in self.person_histories:
            return 0.0
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            return 0.0
        
        recent = list(history)[-5:]
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.01:
            return 0.0
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return math.sqrt(dx**2 + dy**2) / dt
    
    def get_velocity(self, tracking_id):
        """Get velocity components (vx, vy) for a tracked person"""
        if tracking_id not in self.person_histories:
            return 0.0, 0.0
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            return 0.0, 0.0
        
        recent = list(history)[-5:]
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.01:
            return 0.0, 0.0
        
        vx = (recent[-1][0] - recent[0][0]) / dt
        vy = (recent[-1][1] - recent[0][1]) / dt
        return vx, vy


# ==================== HUSKY CONTROLLER WITH RECOVERY ====================
class HuskyGATControllerWithRecovery:
    """
    Main controller with integrated stuck recovery, continuous planning, and visualization
    
    Features:
    - Normal GAT-TD3 navigation
    - Stuck detection and recovery
    - Continuous lookahead planning (runs ahead of robot)
    - Quality-gated continuous learning
    - Path visualization for explainability
    """
    
    def __init__(self, gat_path, actor_path, goal_x, goal_y, 
                 enable_recovery=True,
                 enable_continuous_planning=True,
                 enable_continuous_learning=True,
                 planning_interval=PLANNING_INTERVAL,
                 planning_horizon=PLANNING_HORIZON,
                 publish_planned_path=True,
                 path_publish_rate=PATH_PUBLISH_RATE,
                 visualize_recovery=False):
        rospy.init_node('husky_gat_controller_recovery', anonymous=True)
        self.gat_path = gat_path
        self.actor_path = actor_path
        self.run_logger = None
        self._action_frame = 'tanh'
        self._facing_goal = False
        self._face_goal_t0 = 0.0
        
        # Goal position
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_reached = False
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vx = 0.0
        self.robot_wz = 0.0
        
        # Sensor data
        self.lidar_ranges = None
        self.lidar_angle_min = -math.pi / 2.0
        self.lidar_angle_increment = math.pi / LIDAR_BEAMS
        self.raw_scan_ranges = None  # Full 360° scan
        self.detected_persons = []
        
        # Data synchronization
        self.last_lidar_time = None
        self.last_odom_time = None
        self.last_camera_time = None
        
        # Trajectory predictor
        self.trajectory_predictor = SimpleTrajectoryPredictor()
        
        # Feature flags
        self.enable_recovery = enable_recovery
        self.enable_continuous_planning = enable_continuous_planning
        self.enable_continuous_learning = enable_continuous_learning
        self.visualize_recovery = visualize_recovery  # Pygame window during recovery
        
        # Recovery system (now also handles planning and learning)
        self.recovery_controller = None
        if enable_recovery or enable_continuous_planning or enable_continuous_learning:
            self.recovery_controller = StuckRecoveryController(
                gat_path, actor_path,
                light_update_blend=RECOVERY_LIGHT_UPDATE_BLEND,
                enable_continuous_planning=enable_continuous_planning,
                enable_continuous_learning=enable_continuous_learning,
                planning_interval=planning_interval,
                planning_horizon=planning_horizon,
                visualize_recovery=visualize_recovery
            )
            self.recovery_controller.initialize()
        
        # Standard models (when not using recovery controller)
        self.gat = None
        self.actor = None
        self.edge_index = None
        
        if not (enable_recovery or enable_continuous_planning or enable_continuous_learning):
            self._load_standard_models(gat_path, actor_path)
        
        # ROS subscribers
        rospy.Subscriber('/husky/scan_raw', LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.raw_scan_callback, queue_size=1)
        rospy.Subscriber('/husky/odom_raw', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/person_detections', Float32MultiArray, self.camera_callback, queue_size=1)
        
        # ROS publishers - Standard
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)
        
        # ROS publishers - Visualization (new)
        self.planned_path_pub = rospy.Publisher('/gat_planned_path', Path, queue_size=1)
        self.path_markers_pub = rospy.Publisher('/gat_path_markers', MarkerArray, queue_size=1)
        self.stats_pub = rospy.Publisher('/gat_navigation_stats', String, queue_size=1)
        self.quality_pub = rospy.Publisher('/gat_quality_score', Float32, queue_size=1)
        
        # Path publishing configuration
        self.publish_planned_path = publish_planned_path
        self.path_publish_rate = path_publish_rate
        
        # Path publishing state
        self._last_path_publish_time = 0.0
        self._path_publish_interval = 1.0 / max(0.1, self.path_publish_rate)
        
        # Reward tracking for continuous learning
        self._last_reward = 0.0
        self._last_action = None
        
        # Safety
        self.emergency_stop = False
        self.min_obstacle_distance = float('inf')
        
        rospy.loginfo("Husky GAT-TD3 Controller with Recovery initialized")
        rospy.loginfo(f"Goal: ({goal_x:.2f}, {goal_y:.2f})")
        rospy.loginfo(f"Recovery: {'ENABLED' if enable_recovery else 'DISABLED'}")
        rospy.loginfo(f"Continuous Planning: {'ENABLED' if enable_continuous_planning else 'DISABLED'}")
        rospy.loginfo(f"Continuous Learning: {'ENABLED' if enable_continuous_learning else 'DISABLED'}")
        rospy.loginfo(f"Device: {device}")
    
    def _load_standard_models(self, gat_path, actor_path):
        """Load models for non-recovery mode"""
        try:
            self.gat = GATNetwork(
                lidar_channels=4, robot_channels=4, human_channels=6,
                out_channels=16, num_heads=4,
                lidar_nodes=LIDAR_BEAMS, robot_nodes=1, human_nodes=MAX_HUMANS
            ).to(device)
            
            state_dim = 16 * (LIDAR_BEAMS + 1 + MAX_HUMANS)
            self.actor = Actor(state_dim, action_dim=2).to(device)
            
            self.gat.load_state_dict(torch.load(gat_path, map_location=device))
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            
            self.gat.eval()
            self.actor.eval()
            
            self.edge_index = create_edge_index(LIDAR_BEAMS, 1, MAX_HUMANS).to(device)
            
            rospy.loginfo("Standard models loaded successfully")
        except Exception as e:
            rospy.logfatal(f"Failed to load models: {e}")
            raise
    
    def publish_goal(self):
        """Publish goal for visualization"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "odom"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = self.goal_x
        goal_msg.pose.position.y = self.goal_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        rospy.loginfo(f"Goal published: ({self.goal_x:.2f}, {self.goal_y:.2f})")
    
    def publish_planned_path_viz(self):
        """Publish the planned path for visualization"""
        if not self.publish_planned_path or self.recovery_controller is None:
            return
        
        current_time = time.time()
        if current_time - self._last_path_publish_time < self._path_publish_interval:
            return
        
        self._last_path_publish_time = current_time
        
        # Get planned path from recovery controller
        path_data = self.recovery_controller.get_planned_path_for_visualization()
        
        if not path_data['valid'] or not path_data['path']:
            return
        
        path_points = path_data['path']
        
        # Create Path message
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = rospy.Time.now()
        
        for x, y in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.planned_path_pub.publish(path_msg)
        
        # Also publish markers for better visualization
        self._publish_path_markers(path_points, path_data)
    
    def _publish_path_markers(self, path_points, path_data):
        """Publish path visualization markers"""
        marker_array = MarkerArray()
        
        # Line strip for path
        path_marker = Marker()
        path_marker.header.frame_id = "odom"
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "planned_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.08  # Line width
        
        # Color based on path quality (green = good, yellow = okay, red = poor)
        stats = path_data.get('planning_stats', {})
        success_rate = stats.get('successful_plans', 0) / max(1, stats.get('total_plans', 1))
        
        if success_rate > 0.7:
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.2
        elif success_rate > 0.4:
            path_marker.color.r = 1.0
            path_marker.color.g = 0.8
            path_marker.color.b = 0.0
        else:
            path_marker.color.r = 1.0
            path_marker.color.g = 0.3
            path_marker.color.b = 0.0
        path_marker.color.a = 0.9
        
        for x, y in path_points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.1
            path_marker.points.append(p)
        
        path_marker.pose.orientation.w = 1.0
        marker_array.markers.append(path_marker)
        
        # Add waypoint spheres at intervals
        if len(path_points) > 5:
            for i in range(0, len(path_points), max(1, len(path_points) // 10)):
                wp_marker = Marker()
                wp_marker.header.frame_id = "odom"
                wp_marker.header.stamp = rospy.Time.now()
                wp_marker.ns = "planned_waypoints"
                wp_marker.id = i + 100
                wp_marker.type = Marker.SPHERE
                wp_marker.action = Marker.ADD
                wp_marker.pose.position.x = path_points[i][0]
                wp_marker.pose.position.y = path_points[i][1]
                wp_marker.pose.position.z = 0.15
                wp_marker.pose.orientation.w = 1.0
                wp_marker.scale.x = 0.12
                wp_marker.scale.y = 0.12
                wp_marker.scale.z = 0.12
                wp_marker.color.r = 0.3
                wp_marker.color.g = 0.7
                wp_marker.color.b = 1.0
                wp_marker.color.a = 0.8
                marker_array.markers.append(wp_marker)
        
        # Add endpoint marker
        if len(path_points) > 0:
            end_marker = Marker()
            end_marker.header.frame_id = "odom"
            end_marker.header.stamp = rospy.Time.now()
            end_marker.ns = "path_endpoint"
            end_marker.id = 200
            end_marker.type = Marker.CYLINDER
            end_marker.action = Marker.ADD
            end_marker.pose.position.x = path_points[-1][0]
            end_marker.pose.position.y = path_points[-1][1]
            end_marker.pose.position.z = 0.1
            end_marker.pose.orientation.w = 1.0
            end_marker.scale.x = 0.2
            end_marker.scale.y = 0.2
            end_marker.scale.z = 0.05
            end_marker.color.r = 1.0
            end_marker.color.g = 1.0
            end_marker.color.b = 0.0
            end_marker.color.a = 0.9
            marker_array.markers.append(end_marker)
        
        self.path_markers_pub.publish(marker_array)
    
    def publish_navigation_stats(self):
        """Publish navigation statistics for monitoring"""
        if self.recovery_controller is None:
            return
        
        stats = self.recovery_controller.get_stats()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats = convert_numpy(stats)
        
        # Publish as JSON string for flexibility
        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self.stats_pub.publish(stats_msg)
        
        # Publish quality score separately for easy monitoring
        if 'continuous_learning' in stats:
            quality = stats['continuous_learning'].get('quality_score', 0.0)
            self.quality_pub.publish(Float32(quality))
    
    def compute_reward(self, prev_dist, curr_dist, collision=False):
        """Compute reward for continuous learning"""
        reward = 0.0
        
        # Progress reward
        progress = prev_dist - curr_dist
        if progress > 0:
            reward += 2.0 * progress  # Reward for getting closer
        else:
            reward += 0.5 * progress  # Smaller penalty for getting farther
        
        # Goal reached reward
        if curr_dist < GOAL_REACHED_DIST:
            reward += 100.0
        
        # Collision penalty
        if collision or self.emergency_stop:
            reward -= 50.0
        
        # Small step penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def raw_scan_callback(self, msg):
        """Process full 360° LiDAR scan"""
        self.raw_scan_ranges = np.array(msg.ranges)
    
    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_angle_min = msg.angle_min
        if len(msg.ranges) > 1 and msg.angle_increment != 0.0:
            self.lidar_angle_increment = msg.angle_increment
        self.last_lidar_time = rospy.Time.now()
        
        valid_ranges = self.lidar_ranges[~np.isnan(self.lidar_ranges)]
        valid_ranges = valid_ranges[~np.isinf(valid_ranges)]
        if len(valid_ranges) > 0:
            self.min_obstacle_distance = np.min(valid_ranges)
            if self.min_obstacle_distance < 0.5:
                self.emergency_stop = True
                rospy.logwarn(f"Emergency stop! Obstacle at {self.min_obstacle_distance:.2f}m")
    
    def odom_callback(self, msg):
        """Process odometry data"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_wz = msg.twist.twist.angular.z
        
        self.last_odom_time = rospy.Time.now()
        
        # Check goal (only log once)
        dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 +
                                (self.goal_y - self.robot_y)**2)
        if dist_to_goal < GOAL_REACHED_DIST and not self.goal_reached:
            self.goal_reached = True
            rospy.loginfo("Goal reached!")
    
    def camera_callback(self, msg):
        """Process person detection data"""
        try:
            data = list(msg.data)
            if len(data) == 0:
                self.detected_persons = []
                return
            
            num_persons = int(data[0])
            self.detected_persons = []
            
            for i in range(num_persons):
                idx = 1 + i * 4
                if idx + 3 < len(data):
                    cam_x = data[idx]
                    cam_z = data[idx + 1]
                    conf = data[idx + 2]
                    tracking_id = int(data[idx + 3])
                    
                    robot_x_rel = cam_z
                    robot_y_rel = -cam_x
                    
                    person_x_odom = self.robot_x + robot_x_rel * math.cos(self.robot_theta) - robot_y_rel * math.sin(self.robot_theta)
                    person_y_odom = self.robot_y + robot_x_rel * math.sin(self.robot_theta) + robot_y_rel * math.cos(self.robot_theta)
                    
                    self.trajectory_predictor.update(tracking_id, person_x_odom, person_y_odom)
                    
                    # Get velocity for this person (for recovery simulation)
                    vx, vy = self.trajectory_predictor.get_velocity(tracking_id)
                    
                    self.detected_persons.append({
                        'x_rel': robot_x_rel,
                        'y_rel': robot_y_rel,
                        'x_odom': person_x_odom,
                        'y_odom': person_y_odom,
                        'vx': vx,  # Velocity for recovery simulation
                        'vy': vy,
                        'confidence': conf,
                        'tracking_id': tracking_id
                    })
            
            self.last_camera_time = rospy.Time.now()
        except Exception as e:
            rospy.logwarn(f"Error processing camera data: {e}")
    
    def downsample_lidar(self, ranges):
        """20 sim beams over front 180°, nearest real ray in that direction."""
        if ranges is None or len(ranges) == 0:
            return np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float32)
        return sample_sim_beams(
            ranges, self.lidar_angle_min, self.lidar_angle_increment,
            max_range=LIDAR_RANGE,
        )
    
    def build_observation(self):
        """Build observation in training format"""
        lidar_distances = self.downsample_lidar(self.lidar_ranges)
        angles = BEAM_ANGLES
        
        lidar_nodes = []
        for dist, angle in zip(lidar_distances, angles):
            lidar_nodes.append([
                dist / LIDAR_RANGE,
                np.cos(angle),
                np.sin(angle),
                0.0
            ])
        lidar_nodes = np.array(lidar_nodes, dtype=np.float32)
        
        # Robot node
        dx_world = self.goal_x - self.robot_x
        dy_world = self.goal_y - self.robot_y
        dx_ego = dx_world * math.cos(self.robot_theta) + dy_world * math.sin(self.robot_theta)
        dy_ego = -dx_world * math.sin(self.robot_theta) + dy_world * math.cos(self.robot_theta)
        
        robot_node = np.array([[
            dx_ego / GOAL_NORM,
            dy_ego / GOAL_NORM,
            self.robot_vx,
            self.robot_wz
        ]], dtype=np.float32)
        
        # Human nodes: [x_rel/15, y_rel/15, speed_norm, confidence, pred_x/15, pred_y/15]
        human_nodes = []
        for person in self.detected_persons[:MAX_HUMANS]:
            pred_x, pred_y, pred_conf = self.trajectory_predictor.predict(
                person['tracking_id'], dt=GAT_PRED_HORIZON
            )
            
            pred_dx = pred_x - self.robot_x
            pred_dy = pred_y - self.robot_y
            pred_x_ego = pred_dx * math.cos(self.robot_theta) + pred_dy * math.sin(self.robot_theta)
            pred_y_ego = -pred_dx * math.sin(self.robot_theta) + pred_dy * math.cos(self.robot_theta)
            
            speed = self.trajectory_predictor.get_speed(person['tracking_id']) / 1.5
            
            human_nodes.append([
                person['x_rel'] / CAM_MAX,
                person['y_rel'] / CAM_MAX,
                min(speed, 1.0),
                person['confidence'],
                pred_x_ego / CAM_MAX,
                pred_y_ego / CAM_MAX
            ])
        
        while len(human_nodes) < MAX_HUMANS:
            human_nodes.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        human_nodes = np.array(human_nodes, dtype=np.float32)
        
        # Flatten
        flat_state = np.concatenate([
            lidar_nodes.flatten(),
            robot_node.flatten(),
            human_nodes.flatten()
        ])
        
        return flat_state, lidar_nodes, robot_node, human_nodes
    
    def get_action(self, lidar_nodes, robot_node, human_nodes):
        """Get action from policy or action replay"""
        if self.enable_recovery and self.recovery_controller is not None:
            self.recovery_controller.blend_debug = None
            # Yaw in place to the logged heading from when the twin set off.
            if self.recovery_controller.is_positioning_active():
                pos_action, pos_done = self.recovery_controller.get_positioning_action(
                    self.robot_x, self.robot_y, self.robot_theta,
                    self.goal_x, self.goal_y,
                    self.min_obstacle_distance
                )
                
                if not pos_done and pos_action is not None:
                    self._action_frame = 'env'
                    return pos_action
                # Heading matched: the blend may now be active this same tick
            
            # Blend logged actions with the live policy, which includes humans.
            if self.recovery_controller.is_replay_active():
                replay_action, replay_done = self.recovery_controller.get_replay_action(
                    self.robot_x, self.robot_y, self.robot_theta
                )
                
                if not replay_done and replay_action is not None:
                    # Blended command is already env format: [0-1 linear, -1 to 1 angular]
                    self._action_frame = 'env'
                    action = np.array([
                        replay_action[0],
                        replay_action[1]
                    ], dtype=np.float32)
                    return action
                # Waypoint reached: fall through to the original policy
            
            # Build flat state for recovery controller (policy mode)
            flat_state = np.concatenate([
                lidar_nodes.flatten(),
                robot_node.flatten(),
                human_nodes.flatten()
            ])
            self._action_frame = 'tanh'
            return self.recovery_controller.get_action(flat_state)
        else:
            # Standard inference
            self._action_frame = 'tanh'
            with torch.no_grad():
                lidar_t = torch.FloatTensor(lidar_nodes).to(device)
                robot_t = torch.FloatTensor(robot_node).to(device)
                human_t = torch.FloatTensor(human_nodes).to(device)
                
                embeddings = self.gat(lidar_t, robot_t, human_t, self.edge_index)
                state = embeddings.flatten().unsqueeze(0)
                action = self.actor(state).cpu().numpy()[0]
            
            return action
    
    def execute_action(self, action):
        """Same velocity map as deploy_v5 for policy actions.

        Policy tanh is [-1, 1]. Training does a_in = [(a[0]+1)/2, a[1]]
        before scaling by v_max / w_max, so tanh -1 stops, 0 is half
        speed, and +1 is full speed. Blend and yaw commands are already
        in that env frame ([0, 1] linear, [-1, 1] angular).
        """
        if getattr(self, '_action_frame', 'tanh') == 'env':
            v_norm = float(np.clip(action[0], 0.0, 1.0))
            w_norm = float(np.clip(action[1], -1.0, 1.0))
        else:
            v_norm = (float(action[0]) + 1.0) / 2.0
            w_norm = float(action[1])
        
        linear_vel = float(np.clip(v_norm * MAX_LINEAR_VEL, 0.0, MAX_LINEAR_VEL))
        angular_vel = float(np.clip(w_norm * MAX_ANGULAR_VEL, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))
        
        # Ignore e-stop only while blending through the gap. The logged
        # actions already cleared it in the snapshot. Yaw-align does not.
        in_recovery_phase = (self.enable_recovery and 
                            self.recovery_controller is not None and 
                            self.recovery_controller.is_replay_active())
        
        if self.emergency_stop and not in_recovery_phase and not self._facing_goal:
            linear_vel = 0.0
            angular_vel = 0.0
        
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        return linear_vel, angular_vel, v_norm, w_norm
    
    def check_data_freshness(self):
        """Check sensor data freshness"""
        current_time = rospy.Time.now()
        timeout = rospy.Duration(1.0)
        
        if self.last_lidar_time is None or (current_time - self.last_lidar_time) > timeout:
            rospy.logwarn_throttle(2.0, "LiDAR data stale or missing")
            return False
        
        if self.last_odom_time is None or (current_time - self.last_odom_time) > timeout:
            rospy.logwarn_throttle(2.0, "Odometry data stale or missing")
            return False
        
        return True
    
    def _goal_dist(self):
        return math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)

    def _mode_token(self, in_recovery):
        if self._facing_goal:
            return 'FACE'
        rc = self.recovery_controller
        if in_recovery and rc is not None:
            if rc.is_replay_active():
                return 'GOTO'
            if rc.is_positioning_active():
                return 'ALIGN'
            return 'RECOVERY'
        return 'NORMAL'

    def _heading_err(self):
        rc = self.recovery_controller
        if rc is None or rc.waypoint_theta is None:
            return ''
        err = rc.waypoint_theta - self.robot_theta
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        return float(err)

    def _wp_dist(self):
        rc = self.recovery_controller
        if rc is None or rc.waypoint_x is None:
            return ''
        return math.hypot(rc.waypoint_x - self.robot_x, rc.waypoint_y - self.robot_y)

    def _build_step_row(self, step, t, action, v_norm, w_norm, v_cmd, w_cmd,
                        robot_node, human_nodes, lidar_distances, in_recovery):
        dx_world = self.goal_x - self.robot_x
        dy_world = self.goal_y - self.robot_y
        goal_bearing = math.atan2(
            -dx_world * math.sin(self.robot_theta) + dy_world * math.cos(self.robot_theta),
            dx_world * math.cos(self.robot_theta) + dy_world * math.sin(self.robot_theta)
        )
        min_lidar = float(np.min(lidar_distances)) if len(lidar_distances) else LIDAR_RANGE
        front = np.asarray(lidar_distances)[FRONT_BEAM_SLICE]
        n_seen = len(self.detected_persons)
        n_humans = min(n_seen, MAX_HUMANS)
        mode = self._mode_token(in_recovery)
        debug = None if self.recovery_controller is None else self.recovery_controller.blend_debug
        if debug:
            action_frame = 'env_blend'
        elif mode == 'ALIGN':
            action_frame = 'env_yaw'
        else:
            action_frame = 'tanh'
        estop_suppressed = int(
            self.recovery_controller is not None and self.recovery_controller.is_replay_active()
        )
        closest = float('inf')
        row = {
            'step': step,
            't': t,
            'x': self.robot_x,
            'y': self.robot_y,
            'theta': self.robot_theta,
            'goal_dist': self._goal_dist(),
            'goal_bearing': goal_bearing,
            'dx_ego_norm': float(robot_node[0, 0]),
            'dy_ego_norm': float(robot_node[0, 1]),
            'v_odom': self.robot_vx,
            'w_odom': self.robot_wz,
            'a0_tanh': float(action[0]),
            'a1_tanh': float(action[1]),
            'v_norm': float(v_norm),
            'w_norm': float(w_norm),
            'v_cmd': float(v_cmd),
            'w_cmd': float(w_cmd),
            'min_lidar': min_lidar,
            'forward_clearance': float(np.mean(front)) if len(front) else min_lidar,
            'lidar_std': float(np.std(lidar_distances)) if len(lidar_distances) else 0.0,
            'collision': int(min_lidar < COLLISION_DIST),
            'emergency_stop': int(self.emergency_stop),
            'n_humans': n_humans,
            'n_humans_seen': n_seen,
            'closest_human': closest,
            'mode': mode,
            'action_frame': action_frame,
            'recovery_id': '' if self.run_logger is None else len(self.run_logger.attempts),
            'wp_dist': self._wp_dist(),
            'heading_err': self._heading_err(),
            'replay_idx': '' if not debug else debug.get('replay_idx', ''),
            'logged_v': '' if not debug else debug.get('logged_v', ''),
            'logged_w': '' if not debug else debug.get('logged_w', ''),
            'drl_a0': '' if not debug else debug.get('drl_a0', ''),
            'drl_a1': '' if not debug else debug.get('drl_a1', ''),
            'estop_suppressed': estop_suppressed,
        }
        for i in range(MAX_HUMANS):
            h = human_nodes[i]
            conf = float(h[3])
            dist = math.hypot(float(h[0]) * CAM_MAX, float(h[1]) * CAM_MAX) if conf > 0 else float('inf')
            if dist < closest:
                closest = dist
            row[f'h{i}_x_rel'] = float(h[0])
            row[f'h{i}_y_rel'] = float(h[1])
            row[f'h{i}_speed_norm'] = float(h[2])
            row[f'h{i}_conf'] = conf
            row[f'h{i}_pred_x'] = float(h[4])
            row[f'h{i}_pred_y'] = float(h[5])
            row[f'h{i}_dist'] = dist
        row['closest_human'] = closest
        for i, d in enumerate(lidar_distances):
            row[f'lidar_{i}'] = float(d)
        return row

    def _finish_run(self, outcome):
        try:
            self.stop_robot()
        except Exception:
            pass
        if self.run_logger is not None:
            self.run_logger.finalize(
                outcome, self.robot_x, self.robot_y, self.robot_theta, self._goal_dist()
            )

    def run(self):
        """Main control loop with recovery, continuous planning, and learning"""
        rospy.loginfo("Starting GAT-TD3 navigation with recovery and continuous planning")

        self.run_logger = RecoveryRunLogger(
            self.gat_path, self.actor_path, self.goal_x, self.goal_y
        )
        rospy.loginfo(f"Run logs: {self.run_logger.run_dir}")

        rate = rospy.Rate(CONTROL_RATE)
        outcome = 'CANCELLED'
        step_count = 0
        start_time = time.time()
        in_recovery = False

        try:
            rospy.loginfo("Waiting for sensor data...")
            while not rospy.is_shutdown():
                if self.lidar_ranges is not None and self.last_odom_time is not None:
                    rospy.loginfo("Sensor data received, starting control")
                    break
                rate.sleep()

            if rospy.is_shutdown():
                return

            self.run_logger.mark_start(
                self.robot_x, self.robot_y, self.robot_theta, self._goal_dist()
            )
            start_time = time.time()
        
            # Distance tracking for reward computation
            prev_dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 +
                                          (self.goal_y - self.robot_y)**2)
        
            while not rospy.is_shutdown():
                dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 +
                                        (self.goal_y - self.robot_y)**2)
            
                if self.goal_reached:
                    # Check if we're in recovery mode
                    if self.enable_recovery and self.recovery_controller and self.recovery_controller.recovery_mode:
                        self.run_logger.event(
                            'light_update', blend=RECOVERY_LIGHT_UPDATE_BLEND
                        )
                        # Recovery complete - this applies light update and resets
                        self.recovery_controller.check_recovery_complete(
                            self.robot_x, self.robot_y,
                            self.goal_x, self.goal_y
                        )
                    outcome = 'SUCCESS'
                
                    rospy.loginfo("Goal reached! Stopping robot.")
                    self.stop_robot()
                
                    # Print comprehensive stats
                    if self.recovery_controller:
                        stats = self.recovery_controller.get_stats()
                        rospy.loginfo(f"\n{'='*50}")
                        rospy.loginfo("NAVIGATION COMPLETE - STATISTICS")
                        rospy.loginfo(f"{'='*50}")
                        rospy.loginfo(f"Recovery: {stats['total_recoveries']} attempts, "
                                    f"{stats['successful_recoveries']} successful "
                                    f"({stats['recovery_success_rate']*100:.1f}%)")
                    
                        if 'planning' in stats:
                            p = stats['planning']
                            rospy.loginfo(f"Planning: {p['total_plans']} plans, "
                                        f"{p['successful_plans']} reached goal "
                                        f"(avg time: {p['avg_plan_time']*1000:.1f}ms)")
                    
                        if 'continuous_learning' in stats:
                            cl = stats['continuous_learning']
                            rospy.loginfo(f"Continuous Learning: {cl['total_updates']} updates, "
                                        f"quality={cl['quality_score']:.2f}")
                    
                        # Save continuously updated models
                        if self.enable_continuous_learning:
                            self.recovery_controller.save_updated_models()
                    break
            
                # Check data freshness
                if not self.check_data_freshness():
                    self.run_logger.note_stale_skip()
                    self.stop_robot()
                    rate.sleep()
                    continue
            
                # Build observation
                try:
                    flat_state, lidar_nodes, robot_node, human_nodes = self.build_observation()
                    lidar_downsampled = self.downsample_lidar(self.lidar_ranges)
                
                    # Compute reward for continuous learning
                    reward = self.compute_reward(prev_dist_to_goal, dist_to_goal)
                    prev_dist_to_goal = dist_to_goal
                
                    # Turn to face the goal before the snapshot, so the front scan
                    # and the waypoint chain line up with the route.
                    if self._facing_goal and not in_recovery:
                        err = math.atan2(self.goal_y - self.robot_y,
                                         self.goal_x - self.robot_x) - self.robot_theta
                        while err > math.pi:
                            err -= 2 * math.pi
                        while err < -math.pi:
                            err += 2 * math.pi
                        aligned = abs(err) < 0.26
                        timed_out = (time.time() - self._face_goal_t0) > 15.0
                        if aligned or timed_out:
                            if timed_out and not aligned:
                                rospy.logwarn("Face-goal timed out — starting recovery at the current heading")
                            else:
                                rospy.loginfo("Facing the goal — starting recovery")
                            self.run_logger.event(
                                'face_goal_done',
                                theta=self.robot_theta,
                                heading_err=err,
                                timed_out=timed_out,
                            )
                            self._facing_goal = False
                            self.stop_robot()
                            rospy.sleep(0.3)
                            self.recovery_controller.refresh_snapshot(
                                self.robot_x, self.robot_y, self.robot_theta,
                                self.goal_x, self.goal_y,
                                lidar_downsampled,
                                self.raw_scan_ranges,
                                self.detected_persons,
                            )
                            train_t0 = time.time()
                            success = self.recovery_controller.start_recovery(
                                verbose=True,
                                visualize=self.visualize_recovery
                            )
                            train_wall_s = time.time() - train_t0
                            attempt = self.recovery_controller.last_attempt
                            if attempt is not None:
                                if self.raw_scan_ranges is not None:
                                    attempt = dict(attempt)
                                    attempt['raw_scan_ranges'] = np.asarray(
                                        self.raw_scan_ranges, dtype=float
                                    ).tolist()
                                self.run_logger.save_attempt(attempt, train_wall_s)
                            if success:
                                in_recovery = True
                                rospy.loginfo("Recovery path found — following the waypoint chain")
                            else:
                                rospy.logwarn("Recovery failed — continuing with the original model")
                            rate.sleep()
                            continue

                        self._action_frame = 'env'
                        action = np.array([
                            0.0, float(np.clip(err / 0.8, -1.0, 1.0))
                        ], dtype=np.float32)
                        self._last_action = action
                        linear_vel, angular_vel, v_norm, w_norm = self.execute_action(action)
                        step_count += 1
                        t = time.time() - start_time
                        row = self._build_step_row(
                            step_count, t, action, v_norm, w_norm, linear_vel, angular_vel,
                            robot_node, human_nodes, lidar_downsampled, in_recovery
                        )
                        self.run_logger.log_step(row)
                        if step_count % 10 == 0:
                            rospy.loginfo(
                                f"Step {step_count} [FACE] | heading err {math.degrees(err):.0f}°"
                            )
                        rate.sleep()
                        continue

                    # Update recovery controller with continuous planning and learning
                    if self.recovery_controller and not in_recovery:
                        stuck_detected, status = self.recovery_controller.update(
                            self.robot_x, self.robot_y, self.robot_theta,
                            self.robot_vx, self.robot_wz,
                            self.goal_x, self.goal_y,
                            lidar_downsampled,
                            self.raw_scan_ranges,
                            self.detected_persons,
                            current_action=self._last_action,
                            reward=reward,
                            emergency_stop=self.emergency_stop
                        )
                    
                        if stuck_detected and self.enable_recovery and not self._facing_goal:
                            rospy.logwarn("STUCK DETECTED — turning to face the goal before recovery")
                            self.run_logger.event(
                                'stuck',
                                x=self.robot_x, y=self.robot_y, theta=self.robot_theta,
                                goal_dist=dist_to_goal,
                                n_humans=len(self.detected_persons),
                            )
                            self._facing_goal = True
                            self._face_goal_t0 = time.time()
                            self.stop_robot()
                            rate.sleep()
                            continue
                
                    # Check if recovery is complete
                    if in_recovery and self.recovery_controller:
                        if self.recovery_controller.check_recovery_complete(
                            self.robot_x, self.robot_y,
                            self.goal_x, self.goal_y
                        ):
                            in_recovery = False
                            self.run_logger.event(
                                'light_update', blend=RECOVERY_LIGHT_UPDATE_BLEND
                            )
                            rospy.loginfo("Recovery complete - switched back to original model")
                
                    # Get action
                    action = self.get_action(lidar_nodes, robot_node, human_nodes)
                    self._last_action = action  # Store for continuous learning
                
                    # Execute
                    linear_vel, angular_vel, v_norm, w_norm = self.execute_action(action)

                    step_count += 1
                    t = time.time() - start_time
                    row = self._build_step_row(
                        step_count, t, action, v_norm, w_norm, linear_vel, angular_vel,
                        robot_node, human_nodes, lidar_downsampled, in_recovery
                    )
                    self.run_logger.log_step(row)
                    mode_str = row['mode']
                    if mode_str == 'GOTO' and row['wp_dist'] != '':
                        mode_str = f"GOTO ({row['wp_dist']:.2f}m to WP)"
                
                    # Publish planned path visualization
                    if self.enable_continuous_planning:
                        self.publish_planned_path_viz()
                
                    # Publish stats periodically
                    if step_count % 10 == 0:
                        self.publish_navigation_stats()
                    
                        quality_str = ""
                        if self.recovery_controller and self.enable_continuous_learning:
                            stats = self.recovery_controller.get_stats()
                            if 'continuous_learning' in stats:
                                q = stats['continuous_learning'].get('quality_score', 0)
                                quality_str = f" | Quality: {q:.2f}"
                    
                        rospy.loginfo(f"Step {step_count} [{mode_str}] | Dist: {dist_to_goal:.2f}m | "
                                    f"Vel: ({linear_vel:.2f}, {angular_vel:.2f}) | "
                                    f"Humans: {len(self.detected_persons)} | "
                                    f"MinObs: {self.min_obstacle_distance:.2f}m{quality_str}")
                
                    # Reset emergency stop if cleared
                    if self.min_obstacle_distance > 0.7:
                        self.emergency_stop = False
                
                except Exception as e:
                    rospy.logerr(f"Error in control loop: {e}")
                    import traceback
                    traceback.print_exc()
                    self.run_logger.event('control_error', error=str(e))
                    self.stop_robot()
                    outcome = 'ERROR'
                    break
            
                rate.sleep()

            rospy.loginfo(f"Navigation completed in {time.time() - start_time:.1f} seconds")
        except KeyboardInterrupt:
            outcome = 'CANCELLED'
            raise
        finally:
            self._finish_run(outcome)
    
    def stop_robot(self):
        """Send stop command"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


# ==================== MAIN ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Deploy GAT-TD3 on Husky with Stuck Recovery, Continuous Planning & Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Normal GAT-TD3 navigation
  - Stuck detection and recovery training
  - Continuous lookahead planning (runs ahead of robot)
  - Quality-gated continuous learning (micro-updates)
  - Path visualization for explainability

Examples:
  # Standard usage with recovery (default - recommended)
  python deploy_v5_recovery.py --goal-x 5.0 --goal-y 3.0

  # With recovery visualization window
  python deploy_v5_recovery.py --goal-x 5.0 --goal-y 3.0 --visualize-recovery

  # Disable recovery (policy only)
  python deploy_v5_recovery.py --goal-x 5.0 --goal-y 3.0 --no-recovery

  # Enable continuous planning (experimental - may cause stutter)
  python deploy_v5_recovery.py --goal-x 5.0 --goal-y 3.0 --enable-planning
        """
    )
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Directory containing model files')
    parser.add_argument('--gat-model', type=str, default=None,
                       help='Specific GAT model path')
    parser.add_argument('--actor-model', type=str, default=None,
                       help='Specific Actor model path')
    parser.add_argument('--goal-x', type=float, required=True, help='Goal X coordinate')
    parser.add_argument('--goal-y', type=float, required=True, help='Goal Y coordinate')
    
    # Recovery options
    parser.add_argument('--no-recovery', action='store_true', 
                       help='Disable stuck recovery feature')
    parser.add_argument('--recovery-blend', type=float, default=0.03,
                       help='Light update blend factor (default: 0.03 = 3%%)')
    
    # Continuous planning options (DISABLED by default - causes stutter)
    parser.add_argument('--enable-planning', action='store_true',
                       help='Enable continuous lookahead planning (causes stutter, disabled by default)')
    parser.add_argument('--planning-interval', type=float, default=PLANNING_INTERVAL,
                       help=f'Seconds between planning updates (default: {PLANNING_INTERVAL})')
    parser.add_argument('--planning-horizon', type=int, default=PLANNING_HORIZON,
                       help=f'Steps to simulate ahead (default: {PLANNING_HORIZON})')
    
    # Continuous learning options (DISABLED by default)
    parser.add_argument('--enable-learning', action='store_true',
                       help='Enable continuous quality-gated learning (disabled by default)')
    parser.add_argument('--micro-blend', type=float, default=MICRO_UPDATE_BLEND,
                       help=f'Micro update blend factor (default: {MICRO_UPDATE_BLEND} = {MICRO_UPDATE_BLEND*100}%%)')
    
    # Visualization options
    parser.add_argument('--no-path-vis', action='store_true',
                       help='Disable path visualization publishing')
    parser.add_argument('--path-rate', type=float, default=PATH_PUBLISH_RATE,
                       help=f'Path publish rate in Hz (default: {PATH_PUBLISH_RATE})')
    parser.add_argument('--visualize-recovery', action='store_true',
                       help='Open pygame window showing recovery training environment')
    
    args = parser.parse_args()
    
    ros_initialized = False
    
    try:
        # Find models
        if args.gat_model and args.actor_model:
            gat_path = args.gat_model
            actor_path = args.actor_model
        else:
            search_dir = args.model_dir if args.model_dir else None
            gat_path, actor_path = find_model_files(search_dir)
            
            if gat_path is None or actor_path is None:
                print("ERROR: Could not find model files!")
                return
        
        # Verify files exist
        if not os.path.exists(gat_path):
            print(f"ERROR: GAT model not found: {gat_path}")
            return
        
        if not os.path.exists(actor_path):
            print(f"ERROR: Actor model not found: {actor_path}")
            return
        
        # Determine feature flags
        enable_recovery = not args.no_recovery
        enable_planning = args.enable_planning  # Disabled by default
        enable_learning = args.enable_learning  # Disabled by default
        publish_path = not args.no_path_vis
        path_rate = args.path_rate
        
        # Initialize controller with all configuration
        controller = HuskyGATControllerWithRecovery(
            gat_path, actor_path, 
            args.goal_x, args.goal_y,
            enable_recovery=enable_recovery,
            enable_continuous_planning=enable_planning,
            enable_continuous_learning=enable_learning,
            planning_interval=args.planning_interval,
            planning_horizon=args.planning_horizon,
            publish_planned_path=publish_path,
            path_publish_rate=path_rate,
            visualize_recovery=args.visualize_recovery
        )
        ros_initialized = True
        
        # Publish goal
        controller.publish_goal()
        rospy.sleep(0.5)
        
        # Confirmation
        print(f"\n{'='*70}")
        print(f"DEPLOYMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"Goal Position: ({args.goal_x:.2f}, {args.goal_y:.2f})")
        print(f"Models: {os.path.basename(gat_path)} & {os.path.basename(actor_path)}")
        print(f"\nFeatures:")
        print(f"  Recovery Mode:         {'ENABLED' if enable_recovery else 'DISABLED'}")
        if enable_recovery:
            print(f"    - Blend Factor:      {args.recovery_blend*100:.1f}%")
        print(f"  Continuous Planning:   {'ENABLED' if enable_planning else 'DISABLED'}")
        if enable_planning:
            print(f"    - Interval:          {args.planning_interval}s")
            print(f"    - Horizon:           {args.planning_horizon} steps")
        print(f"  Continuous Learning:   {'ENABLED' if enable_learning else 'DISABLED'}")
        if enable_learning:
            print(f"    - Micro Blend:       {args.micro_blend*100:.2f}%")
        print(f"  Path Visualization:    {'ENABLED' if publish_path else 'DISABLED'}")
        if publish_path:
            print(f"    - Publish Rate:      {path_rate} Hz")
        
        if enable_planning:
            print(f"\nROS Topics for Visualization:")
            print(f"  /gat_planned_path       - nav_msgs/Path")
            print(f"  /gat_path_markers       - visualization_msgs/MarkerArray")
            print(f"  /gat_navigation_stats   - std_msgs/String (JSON)")
            print(f"  /gat_quality_score      - std_msgs/Float32")
        
        print(f"\nPress Ctrl+C at any time to stop.\n")
        
        response = input("Proceed with deployment? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            print(f"\n{'='*70}")
            print(f"STARTING DEPLOYMENT")
            print(f"{'='*70}\n")
            controller.run()
        else:
            print("\nDeployment cancelled.")
            controller.stop_robot()
    
    except rospy.ROSInterruptException:
        print("\nController interrupted")
    except KeyboardInterrupt:
        print("\nController interrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ros_initialized:
            try:
                cmd = Twist()
                pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
                rospy.sleep(0.1)
                pub.publish(cmd)
                print("\nRobot stopped.")
            except:
                pass


if __name__ == '__main__':
    main()
