#!/usr/bin/env python3

"""
GAT-TD3 Real Husky Deployment Script
Runs trained GAT-TD3 navigation agent on Clearpath Husky A200
Integrates LiDAR, ZED camera, odometry with learned DRL controller
Auto-discovers model files in the script directory
"""

import os
import glob
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray
import tf
import math
import time
from collections import deque
from graph_utils import create_edge_index
from lidar_front import BEAM_ANGLES, sample_sim_beams

# ==================== CONFIGURATION ====================
CONTROL_RATE = 10  # Hz (10 Hz = 0.1s timestep)
LIDAR_BEAMS = 20  # Downsample to 20 beams for network
MAX_HUMANS = 3  # Maximum human nodes
LIDAR_MAX_RANGE = 10.0  # meters
GOAL_REACHED_DIST = 0.75  # meters
# Match pygame_env_rosv5.py node-feature normalisation
GOAL_NORM = 20.0   # sim world_size max(world_w, world_h)
CAM_MAX = 15.0     # sim camera range (human x/y/pred)
# Sim GAT waypoint: index 1 of linspace(0.1, 2.0, 10) ≈ 0.311s
GAT_PRED_HORIZON = float(np.linspace(0.1, 2.0, 10)[1])

# Husky velocity limits (loaded from ROS params or defaults)
MAX_LINEAR_VEL = 1.0  # m/s
MAX_ANGULAR_VEL = 2.0  # rad/s
COLLISION_DIST = 0.5  # matches pygame_env_rosv5.py
TURN_W_THRESH = 0.05  # rad/s, matches sim "turning" threshold
FRONT_BEAM_SLICE = slice(8, 13)  # central 5 of 20 beams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STEP_CSV_FIELDS = [
    'step', 't',
    'x', 'y', 'theta',
    'goal_dist', 'goal_bearing',
    'dx_ego_norm', 'dy_ego_norm', 'v_odom', 'w_odom',
    'a0_tanh', 'a1_tanh', 'v_norm', 'w_norm', 'v_cmd', 'w_cmd',
    'min_lidar', 'forward_clearance', 'lidar_std',
    'collision', 'emergency_stop',
    'n_humans', 'closest_human',
]


def _human_fields():
    fields = []
    for i in range(MAX_HUMANS):
        fields.extend([
            f'h{i}_x_rel', f'h{i}_y_rel', f'h{i}_speed_norm',
            f'h{i}_conf', f'h{i}_pred_x', f'h{i}_pred_y', f'h{i}_dist',
        ])
    return fields


def _lidar_fields():
    return [f'lidar_{i}' for i in range(LIDAR_BEAMS)]


STEP_CSV_FIELDS = STEP_CSV_FIELDS + _human_fields() + _lidar_fields()


class DeployRunLogger:
    """Step CSV + episode summary for sim2real comparison. Always flushed on finish."""

    def __init__(self, gat_path, actor_path, goal_x, goal_y, log_dir=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'logs', 'deploy_runs')
        self.run_dir = os.path.join(log_dir, f'deploy_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, 'steps.csv')
        self.summary_json_path = os.path.join(self.run_dir, 'summary.json')
        self.summary_txt_path = os.path.join(self.run_dir, 'summary.txt')

        self.header = {
            'domain': 'real',
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
            'goal_x': float(goal_x),
            'goal_y': float(goal_y),
        }
        self.steps = []
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
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=STEP_CSV_FIELDS,
                                          extrasaction='ignore')
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def mark_start(self, x, y, theta, goal_dist):
        self.start_time = time.time()
        self.start_x = float(x)
        self.start_y = float(y)
        self.start_theta = float(theta)
        self.start_goal_dist = float(goal_dist)
        self.prev_x = float(x)
        self.prev_y = float(y)

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

    def _aggregate(self, outcome, final_x, final_y, final_theta, final_goal_dist):
        elapsed = (time.time() - self.start_time) if self.start_time else 0.0
        n = len(self.steps)
        if n == 0:
            return {
                **self.header,
                'outcome': outcome,
                'n_steps': 0,
                'elapsed_s': elapsed,
                'note': 'no control steps recorded',
            }

        min_lidars = np.array([s['min_lidar'] for s in self.steps], dtype=np.float64)
        v_cmds = np.array([s['v_cmd'] for s in self.steps], dtype=np.float64)
        w_cmds = np.array([s['w_cmd'] for s in self.steps], dtype=np.float64)
        v_odoms = np.array([s['v_odom'] for s in self.steps], dtype=np.float64)
        closest_h = np.array([s['closest_human'] for s in self.steps], dtype=np.float64)
        n_humans = np.array([s['n_humans'] for s in self.steps], dtype=np.float64)
        dt = 1.0 / CONTROL_RATE
        dv = np.diff(v_cmds) if n > 1 else np.array([0.0])
        dw = np.diff(w_cmds) if n > 1 else np.array([0.0])
        turning = np.mean(np.abs(w_cmds) > TURN_W_THRESH)
        finite_human = closest_h[np.isfinite(closest_h)]
        start_goal = self.start_goal_dist if self.start_goal_dist else 0.0
        efficiency = (start_goal / self.path_length) if self.path_length > 1e-6 else 0.0

        return {
            **self.header,
            'outcome': outcome,
            'n_steps': n,
            'elapsed_s': elapsed,
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
            'frac_turning': float(turning),
            'mean_clearance': float(np.mean(min_lidars)),
            'min_clearance': float(np.min(min_lidars)),
            'p5_clearance': float(np.percentile(min_lidars, 5)),
            'time_clearance_lt_1_0_s': float(np.mean(min_lidars < 1.0) * n * dt),
            'time_clearance_lt_0_7_s': float(np.mean(min_lidars < 0.7) * n * dt),
            'time_clearance_lt_0_5_s': float(np.mean(min_lidars < COLLISION_DIST) * n * dt),
            'collision_steps': self.collision_steps,
            'e_stop_steps': self.e_stop_steps,
            'stale_sensor_skips': self.stale_skips,
            'mean_n_humans': float(np.mean(n_humans)),
            'max_n_humans': int(np.max(n_humans)),
            'closest_human_m': float(np.min(finite_human)) if len(finite_human) else None,
            'time_human_lt_2_0_s': float(np.mean(closest_h < 2.0) * n * dt) if len(finite_human) else 0.0,
        }

    def _format_txt(self, summary):
        lines = [
            'GAT-TD3 real-Husky run summary (sim2real)',
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
            f"Models:             {summary.get('gat_model')} / {summary.get('actor_model')}",
            f"Step log:           {self.csv_path}",
            f"Summary JSON:       {self.summary_json_path}",
        ]
        return '\n'.join(lines) + '\n'

    def finalize(self, outcome, final_x, final_y, final_theta, final_goal_dist):
        if self.finalized:
            return
        self.finalized = True
        summary = self._aggregate(outcome, final_x, final_y, final_theta, final_goal_dist)
        try:
            self._csv_file.close()
        except Exception:
            pass
        def _safe(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        summary = {k: _safe(v) for k, v in summary.items()}
        with open(self.summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
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
        
        # Project all node types to same dimension
        self.lidar_proj = nn.Linear(lidar_channels, out_channels)
        self.robot_proj = nn.Linear(robot_channels, out_channels)
        self.human_proj = nn.Linear(human_channels, out_channels)
        
        # GAT layers operate on uniform dimensions
        self.g1 = GATConv(out_channels, out_channels, heads=num_heads, concat=True)
        self.g2 = GATConv(out_channels * num_heads, out_channels, heads=num_heads, concat=False)
    
    def forward(self, lidar_nodes, robot_nodes, human_nodes, edge_index):
        # Project to uniform dimensions
        lidar_proj = self.lidar_proj(lidar_nodes)
        robot_proj = self.robot_proj(robot_nodes)
        human_proj = self.human_proj(human_nodes)
        
        # Concatenate projected features
        x = torch.cat([lidar_proj, robot_proj, human_proj], dim=0)
        x_init = x.clone()
        
        # Apply GAT layers
        x = F.elu(self.g1(x, edge_index))
        x = self.g2(x, edge_index)
        x = x + x_init
        
        return x


class Actor(nn.Module):
    """Policy network mapping GAT embeddings to actions"""
    
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=128):
        super().__init__()
        # CHANGED: l1, l2, l3 instead of fc1, fc2, fc3
        self.l1 = nn.Linear(state_dim, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, action_dim)
    
    def forward(self, x):
        # CHANGED: use l1, l2, l3
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))


# ==================== MODEL AUTO-DISCOVERY ====================
def find_model_files(search_dir=None):
    """
    Automatically find GAT and Actor model files in the directory.
    Looks for files ending with '_gat.pth' and '_actor.pth'
    
    Args:
        search_dir: Directory to search (defaults to script directory)
    
    Returns:
        tuple: (gat_path, actor_path) or (None, None) if not found
    """
    if search_dir is None:
        search_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Searching for model files in: {search_dir}")
    
    # Search for GAT model (with underscore: _gat.pth)
    gat_pattern = os.path.join(search_dir, "*_gat.pth")
    gat_files = glob.glob(gat_pattern)
    
    # Search for Actor model (with underscore: _actor.pth)
    actor_pattern = os.path.join(search_dir, "*_actor.pth")
    actor_files = glob.glob(actor_pattern)
    
    print(f"Found {len(gat_files)} GAT model file(s)")
    print(f"Found {len(actor_files)} Actor model file(s)")
    
    if not gat_files:
        print(f"No GAT model files (*_gat.pth) found in {search_dir}")
        # List all .pth files to help debug
        all_pth = glob.glob(os.path.join(search_dir, "*.pth"))
        if all_pth:
            print(f"Available .pth files: {[os.path.basename(f) for f in all_pth]}")
        return None, None
    
    if not actor_files:
        print(f"No Actor model files (*_actor.pth) found in {search_dir}")
        # List all .pth files to help debug
        all_pth = glob.glob(os.path.join(search_dir, "*.pth"))
        if all_pth:
            print(f"Available .pth files: {[os.path.basename(f) for f in all_pth]}")
        return None, None
    
    # If multiple files found, use the most recently modified
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
        self.person_histories = {}  # {tracking_id: deque([x, y, t])}
    
    def update(self, tracking_id, x, y):
        """Update person position history"""
        current_time = time.time()
        if tracking_id not in self.person_histories:
            self.person_histories[tracking_id] = deque(maxlen=self.history_size)
        self.person_histories[tracking_id].append([x, y, current_time])
    
    def predict(self, tracking_id, dt=1.0):
        """Predict future position using velocity extrapolation"""
        if tracking_id not in self.person_histories:
            return 0.0, 0.0, 0.0  # No prediction available
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            # Not enough history, return current position
            if len(history) == 1:
                return history[-1][0], history[-1][1], 0.5
            return 0.0, 0.0, 0.0
        
        # Calculate velocity from recent history
        recent = list(history)[-5:]  # Use last 5 observations
        vx = (recent[-1][0] - recent[0][0]) / max(0.1, recent[-1][2] - recent[0][2])
        vy = (recent[-1][1] - recent[0][1]) / max(0.1, recent[-1][2] - recent[0][2])
        
        # Predict future position
        pred_x = recent[-1][0] + vx * dt
        pred_y = recent[-1][1] + vy * dt
        confidence = min(1.0, len(history) / self.history_size)
        
        return pred_x, pred_y, confidence
    
    def get_speed(self, tracking_id):
        """Get current estimated speed for a tracked person"""
        if tracking_id not in self.person_histories:
            return 0.0
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate speed from recent history
        recent = list(history)[-5:]  # Use last 5 observations
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.01:
            return 0.0
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        speed = math.sqrt(dx**2 + dy**2) / dt
        
        return speed


# ==================== HUSKY CONTROLLER ====================
class HuskyGATController:
    """Main controller for GAT-TD3 navigation on real Husky"""
    
    def __init__(self, gat_path, actor_path, goal_x, goal_y):
        rospy.init_node('husky_gat_controller', anonymous=True)
        
        # Goal position (in odom frame)
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
        self.detected_persons = []  # List of (x, y, confidence, tracking_id)
        
        # Data synchronization
        self.last_lidar_time = None
        self.last_odom_time = None
        self.last_camera_time = None
        
        # Trajectory predictor
        self.trajectory_predictor = SimpleTrajectoryPredictor()
        
        # Load trained models
        rospy.loginfo(f"Loading models:")
        rospy.loginfo(f"  GAT: {gat_path}")
        rospy.loginfo(f"  Actor: {actor_path}")
        self.load_models(gat_path, actor_path)
        
        # Create fixed edge index
        self.edge_index = create_edge_index(LIDAR_BEAMS, 1, MAX_HUMANS).to(device)
        
        # ROS subscribers
        rospy.Subscriber('/husky/scan_raw', LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber('/husky/odom_raw', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/person_detections', Float32MultiArray, self.camera_callback, queue_size=1)
        
        # ROS publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # ROS publisher for goal (for visualization)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)
        
        # Safety
        self.emergency_stop = False
        self.min_obstacle_distance = float('inf')
        
        self.gat_path = gat_path
        self.actor_path = actor_path
        self.run_logger = None
        
        rospy.loginfo("Husky GAT-TD3 Controller initialized")
        rospy.loginfo(f"Goal: ({goal_x:.2f}, {goal_y:.2f})")
        rospy.loginfo(f"Device: {device}")
    
    def publish_goal(self):
        """Publish goal to ROS topic for visualization"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "odom"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = self.goal_x
        goal_msg.pose.position.y = self.goal_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0  # Neutral orientation
        
        self.goal_pub.publish(goal_msg)
        rospy.loginfo(f"Goal published to /move_base_simple/goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
    
    def load_models(self, gat_path, actor_path):
        """Load trained GAT and Actor networks"""
        try:
            # Initialize networks with CORRECT dimensions
            self.gat = GATNetwork(
                lidar_channels=4, robot_channels=4, human_channels=6,
                out_channels=16, num_heads=4,  # CHANGED: 16 not 128!
                lidar_nodes=LIDAR_BEAMS, robot_nodes=1, human_nodes=MAX_HUMANS
            ).to(device)
            
            # Actor input is: out_channels * total_nodes = 16 * 24 = 384
            state_dim = 16 * (LIDAR_BEAMS + 1 + MAX_HUMANS)  # 16 * 24 = 384
            self.actor = Actor(state_dim, action_dim=2).to(device)
            
            # Load weights
            self.gat.load_state_dict(torch.load(gat_path, map_location=device))
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            
            self.gat.eval()
            self.actor.eval()
            
            rospy.loginfo("Models loaded successfully")
        except Exception as e:
            rospy.logfatal(f"Failed to load models: {e}")
            raise
    
    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_angle_min = msg.angle_min
        if len(msg.ranges) > 1 and msg.angle_increment != 0.0:
            self.lidar_angle_increment = msg.angle_increment
        self.last_lidar_time = rospy.Time.now()
        
        # Update safety check
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
        
        # Extract yaw from quaternion
        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # Velocity
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_wz = msg.twist.twist.angular.z
        
        self.last_odom_time = rospy.Time.now()
        
        # Check if goal reached
        dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 +
                                (self.goal_y - self.robot_y)**2)
        if dist_to_goal < GOAL_REACHED_DIST:
            self.goal_reached = True
            rospy.loginfo("Goal reached!")
    
    def camera_callback(self, msg):
        """Process person detection data from ZED camera"""
        try:
            data = list(msg.data)
            if len(data) == 0:
                self.detected_persons = []
                return
            
            num_persons = int(data[0])
            self.detected_persons = []
            
            # Parse detections: [num_persons, cam_x, cam_z, conf, tracking_id, ...]
            for i in range(num_persons):
                idx = 1 + i * 4
                if idx + 3 < len(data):
                    cam_x = data[idx]  # Left/right
                    cam_z = data[idx + 1]  # Forward distance
                    conf = data[idx + 2]
                    tracking_id = int(data[idx + 3])
                    
                    # Transform to robot frame (cam_z is forward, -cam_x is left)
                    robot_x_rel = cam_z
                    robot_y_rel = -cam_x
                    
                    # Transform to odom frame
                    person_x_odom = self.robot_x + robot_x_rel * math.cos(self.robot_theta) - robot_y_rel * math.sin(self.robot_theta)
                    person_y_odom = self.robot_y + robot_x_rel * math.sin(self.robot_theta) + robot_y_rel * math.cos(self.robot_theta)
                    
                    # Update trajectory predictor
                    self.trajectory_predictor.update(tracking_id, person_x_odom, person_y_odom)
                    
                    self.detected_persons.append({
                        'x_rel': robot_x_rel,
                        'y_rel': robot_y_rel,
                        'x_odom': person_x_odom,
                        'y_odom': person_y_odom,
                        'confidence': conf,
                        'tracking_id': tracking_id
                    })
            
            self.last_camera_time = rospy.Time.now()
        except Exception as e:
            rospy.logwarn(f"Error processing camera data: {e}")
    
    def downsample_lidar(self, ranges):
        """20 sim beams over front 180°, nearest real ray in that direction."""
        if ranges is None or len(ranges) == 0:
            return np.full(LIDAR_BEAMS, LIDAR_MAX_RANGE, dtype=np.float32)
        return sample_sim_beams(
            ranges, self.lidar_angle_min, self.lidar_angle_increment,
            max_range=LIDAR_MAX_RANGE,
        )
    
    def build_observation(self):
        """Build observation in the same format as training environment"""
        lidar_distances = self.downsample_lidar(self.lidar_ranges)
        
        lidar_nodes = []
        for dist, angle in zip(lidar_distances, BEAM_ANGLES):
            lidar_nodes.append([
                dist / LIDAR_MAX_RANGE,
                np.cos(angle),
                np.sin(angle),
                0.0
            ])
        lidar_nodes = np.array(lidar_nodes, dtype=np.float32)
        
        # Robot node: [goal_x_ego/world_max, goal_y_ego/world_max, linear_vel, angular_vel]
        dx_world = self.goal_x - self.robot_x
        dy_world = self.goal_y - self.robot_y
        
        # Transform to ego frame
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
            # GAT future-position feature: 100% CV at sim waypoint (~0.31s)
            pred_x, pred_y, pred_conf = self.trajectory_predictor.predict(
                person['tracking_id'], dt=GAT_PRED_HORIZON
            )
            
            # Transform prediction to robot frame
            pred_dx = pred_x - self.robot_x
            pred_dy = pred_y - self.robot_y
            pred_x_ego = pred_dx * math.cos(self.robot_theta) + pred_dy * math.sin(self.robot_theta)
            pred_y_ego = -pred_dx * math.sin(self.robot_theta) + pred_dy * math.cos(self.robot_theta)
            
            # Get actual speed from trajectory history (normalized by 1.5 m/s as in simulation)
            speed = self.trajectory_predictor.get_speed(person['tracking_id']) / 1.5
            
            human_nodes.append([
                person['x_rel'] / CAM_MAX,
                person['y_rel'] / CAM_MAX,
                min(speed, 1.0),
                person['confidence'],
                pred_x_ego / CAM_MAX,
                pred_y_ego / CAM_MAX
            ])
        
        # Zero-pad if fewer than MAX_HUMANS
        while len(human_nodes) < MAX_HUMANS:
            human_nodes.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        human_nodes = np.array(human_nodes, dtype=np.float32)
        
        return lidar_nodes, robot_node, human_nodes, lidar_distances
    
    def get_action(self, lidar_nodes, robot_node, human_nodes):
        """Get action from trained policy"""
        with torch.no_grad():
            # Convert to torch tensors
            lidar_t = torch.FloatTensor(lidar_nodes).to(device)
            robot_t = torch.FloatTensor(robot_node).to(device)
            human_t = torch.FloatTensor(human_nodes).to(device)
            
            # Get GAT embeddings
            embeddings = self.gat(lidar_t, robot_t, human_t, self.edge_index)
            
            # Flatten embeddings
            state = embeddings.flatten().unsqueeze(0)
            
            # Get action from actor
            action = self.actor(state).cpu().numpy()[0]
            
        return action
    
    def execute_action(self, action):
        """Same map as training, then SI units for husky_velocity_controller.

        Actor tanh is [-1, 1]. Training does a_in = [(a[0]+1)/2, a[1]] before
        the env scales by v_max / w_max, so:
          tanh -1 → stop,  0 → 0.5*vmax,  +1 → vmax
        That is already [0, vmax] m/s — no reverse, legal cmd_vel.
        Clip is only a safety net.
        """
        v_norm = (float(action[0]) + 1.0) / 2.0  # [-1,1] → [0,1]
        w_norm = float(action[1])
        linear_vel = np.clip(v_norm * MAX_LINEAR_VEL, 0.0, MAX_LINEAR_VEL)
        angular_vel = np.clip(w_norm * MAX_ANGULAR_VEL, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        
        # Emergency stop override
        if self.emergency_stop:
            linear_vel = 0.0
            angular_vel = 0.0
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        return linear_vel, angular_vel, v_norm, w_norm
    
    def check_data_freshness(self):
        """Check if sensor data is recent enough"""
        current_time = rospy.Time.now()
        timeout = rospy.Duration(1.0)  # 1 second timeout
        
        if self.last_lidar_time is None or (current_time - self.last_lidar_time) > timeout:
            rospy.logwarn_throttle(2.0, "LiDAR data stale or missing")
            return False
        
        if self.last_odom_time is None or (current_time - self.last_odom_time) > timeout:
            rospy.logwarn_throttle(2.0, "Odometry data stale or missing")
            return False
        
        return True
    
    def _goal_dist(self):
        return math.sqrt((self.goal_x - self.robot_x)**2 +
                         (self.goal_y - self.robot_y)**2)
    
    def _build_step_row(self, step, t, action, v_norm, w_norm, v_cmd, w_cmd,
                        lidar_nodes, robot_node, human_nodes, lidar_distances):
        goal_dist = self._goal_dist()
        dx_world = self.goal_x - self.robot_x
        dy_world = self.goal_y - self.robot_y
        goal_bearing = math.atan2(
            -dx_world * math.sin(self.robot_theta) + dy_world * math.cos(self.robot_theta),
            dx_world * math.cos(self.robot_theta) + dy_world * math.sin(self.robot_theta)
        )
        min_lidar = float(np.min(lidar_distances)) if len(lidar_distances) else LIDAR_MAX_RANGE
        front = lidar_distances[FRONT_BEAM_SLICE]
        n_humans = int(sum(1 for p in self.detected_persons[:MAX_HUMANS]))
        closest = float('inf')
        row = {
            'step': step,
            't': t,
            'x': self.robot_x,
            'y': self.robot_y,
            'theta': self.robot_theta,
            'goal_dist': goal_dist,
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
            'forward_clearance': float(np.mean(front)),
            'lidar_std': float(np.std(lidar_distances)),
            'collision': int(min_lidar < COLLISION_DIST),
            'emergency_stop': int(self.emergency_stop),
            'n_humans': n_humans,
            'closest_human': closest,
        }
        for i in range(MAX_HUMANS):
            h = human_nodes[i]
            hx = float(h[0]) * CAM_MAX
            hy = float(h[1]) * CAM_MAX
            conf = float(h[3])
            dist = math.hypot(hx, hy) if conf > 0 else float('inf')
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
                outcome,
                self.robot_x, self.robot_y, self.robot_theta,
                self._goal_dist()
            )
    
    def run(self):
        """Main control loop. Always writes run logs on success or cancel."""
        rospy.loginfo("Starting GAT-TD3 navigation control loop")
        
        self.run_logger = DeployRunLogger(
            self.gat_path, self.actor_path, self.goal_x, self.goal_y
        )
        rospy.loginfo(f"Run logs: {self.run_logger.run_dir}")
        
        rate = rospy.Rate(CONTROL_RATE)
        outcome = 'CANCELLED'
        
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
            
            step_count = 0
            start_time = time.time()
            
            while not rospy.is_shutdown():
                if self.goal_reached:
                    rospy.loginfo("Goal reached! Stopping robot.")
                    outcome = 'SUCCESS'
                    break
                
                if not self.check_data_freshness():
                    self.run_logger.note_stale_skip()
                    self.stop_robot()
                    rate.sleep()
                    continue
                
                try:
                    lidar_nodes, robot_node, human_nodes, lidar_distances = self.build_observation()
                    action = self.get_action(lidar_nodes, robot_node, human_nodes)
                    linear_vel, angular_vel, v_norm, w_norm = self.execute_action(action)
                    
                    step_count += 1
                    t = time.time() - start_time
                    row = self._build_step_row(
                        step_count, t, action, v_norm, w_norm, linear_vel, angular_vel,
                        lidar_nodes, robot_node, human_nodes, lidar_distances
                    )
                    self.run_logger.log_step(row)
                    
                    if step_count % 10 == 0:
                        rospy.loginfo(
                            f"Step {step_count} | Dist to goal: {row['goal_dist']:.2f}m | "
                            f"Vel: ({linear_vel:.2f}, {angular_vel:.2f}) | "
                            f"Humans detected: {len(self.detected_persons)} | "
                            f"Min obstacle: {self.min_obstacle_distance:.2f}m"
                        )
                    
                    if self.min_obstacle_distance > 0.7:
                        self.emergency_stop = False
                
                except Exception as e:
                    rospy.logerr(f"Error in control loop: {e}")
                    self.stop_robot()
                    outcome = 'ERROR'
                    break
                
                rate.sleep()
            
            if outcome != 'SUCCESS' and outcome != 'ERROR':
                if self.goal_reached:
                    outcome = 'SUCCESS'
                elif self.run_logger.e_stop_steps > 0 and not self.goal_reached:
                    # Interrupted or shutdown while/after e-stop
                    outcome = 'CANCELLED'
                else:
                    outcome = 'CANCELLED'
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
    parser = argparse.ArgumentParser(description='Deploy GAT-TD3 on Husky A200')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Directory containing model files (default: script directory)')
    parser.add_argument('--gat-model', type=str, default=None,
                       help='Specific GAT model path (overrides auto-discovery)')
    parser.add_argument('--actor-model', type=str, default=None,
                       help='Specific Actor model path (overrides auto-discovery)')
    parser.add_argument('--goal-x', type=float, required=True, help='Goal X coordinate (odom frame)')
    parser.add_argument('--goal-y', type=float, required=True, help='Goal Y coordinate (odom frame)')
    args = parser.parse_args()
    
    # Initialize flag to track if ROS was initialized
    ros_initialized = False
    controller = None
    
    try:
        # Determine model paths
        if args.gat_model and args.actor_model:
            # Explicit paths provided
            gat_path = args.gat_model
            actor_path = args.actor_model
            print("Using explicitly specified model paths")
        else:
            # Auto-discover models
            search_dir = args.model_dir if args.model_dir else None
            gat_path, actor_path = find_model_files(search_dir)
            
            if gat_path is None or actor_path is None:
                print("ERROR: Could not find model files!")
                print("Please specify paths with --gat-model and --actor-model")
                print("Or place *_gat.pth and *_actor.pth files in the script directory")
                print("\nNote: Model files should be named like 'TD3_gat_humans_gat.pth' and 'TD3_gat_humans_actor.pth'")
                return
        
        # Verify files exist
        if not os.path.exists(gat_path):
            print(f"ERROR: GAT model not found: {gat_path}")
            return
        
        if not os.path.exists(actor_path):
            print(f"ERROR: Actor model not found: {actor_path}")
            return
        
        # Initialize controller (ROS gets initialized inside HuskyGATController.__init__)
        controller = HuskyGATController(gat_path, actor_path, args.goal_x, args.goal_y)
        ros_initialized = True  # Mark that ROS was initialized successfully
        
        # Publish goal for visualization
        print(f"\n{'='*60}")
        print(f"GOAL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Goal Position: ({args.goal_x:.2f}, {args.goal_y:.2f}) in odom frame")
        print(f"Publishing goal to /move_base_simple/goal for visualization...")
        controller.publish_goal()
        
        # Give time for goal to propagate
        rospy.sleep(0.5)
        
        # Confirmation prompt
        print(f"\n{'='*60}")
        print(f"DEPLOYMENT CONFIRMATION")
        print(f"{'='*60}")
        print(f"Ready to deploy GAT-TD3 navigation controller")
        print(f"Target Goal: X={args.goal_x:.2f}m, Y={args.goal_y:.2f}m")
        print(f"Models loaded: {os.path.basename(gat_path)} & {os.path.basename(actor_path)}")
        print(f"\nThe robot will begin autonomous navigation when confirmed.")
        print(f"Press Ctrl+C at any time to stop.\n")
        
        response = input("Proceed with deployment? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            print(f"\n{'='*60}")
            print(f"STARTING DEPLOYMENT")
            print(f"{'='*60}\n")
            controller.run()
        else:
            print("\nDeployment cancelled by user.")
            controller.stop_robot()
    
    except rospy.ROSInterruptException:
        print("\nController interrupted by user")
        if controller is not None and controller.run_logger is not None:
            controller._finish_run('CANCELLED')
    except KeyboardInterrupt:
        print("\nController interrupted by user")
        if controller is not None and controller.run_logger is not None:
            controller._finish_run('CANCELLED')
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        if controller is not None and controller.run_logger is not None:
            controller._finish_run('ERROR')
    finally:
        # Only try to publish stop command if ROS was initialized
        if ros_initialized:
            try:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
                rospy.sleep(0.1)  # Give publisher time to connect
                pub.publish(cmd)
                print("\nRobot stopped.")
            except:
                pass  # Silently fail if ROS is already shut down


if __name__ == '__main__':
    main()
