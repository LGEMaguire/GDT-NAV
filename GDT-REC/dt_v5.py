#!/usr/bin/env python3

"""
Digital Twin Visualization for Real Husky A200
Pygame visualization that mirrors the simulation environment
Subscribes to ROS topics and visualizes real-time navigation with GAT attention
Includes coordinate axes for goal placement reference
Robot-centered view with rotation (robot always faces up)
Features: Attention-weighted edges, embedding interpretation, semantic classification
"""

import math
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray, Float32, String
import tf
from collections import deque
import time
import os
import json
from lidar_front import BEAM_ANGLES, sample_sim_beams

# Pygame rendering
try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed")
    exit(1)

# PyTorch for GAT inference
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from graph_utils import create_edge_index
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not available - GAT features disabled")
    TORCH_AVAILABLE = False

device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

# ==================== CONSTANTS ====================
LIDAR_RANGE = 10.0  # meters
LIDAR_BEAMS = 20  # Downsample to match network
CONTROL_RATE = 10  # Hz
MAX_HUMANS = 3
# Sim GAT waypoint: index 1 of linspace(0.1, 2.0, 10) ≈ 0.311s
GAT_PRED_HORIZON = float(np.linspace(0.1, 2.0, 10)[1])

# Husky physical specifications
HUSKY_LENGTH = 0.9874
HUSKY_WIDTH = 0.5709
HUSKY_WHEELBASE = 0.555
HUSKY_TRACK_WIDTH = 0.555
HUSKY_WHEEL_RADIUS = 0.1651

# Colors (matching simulation)
COLOR_BACKGROUND = (30, 30, 30)
COLOR_BOUNDARY = (100, 100, 100)
COLOR_LIDAR_NODE = (211, 211, 211)
COLOR_LIDAR_BEAM = (80, 80, 80)
COLOR_ROBOT_BODY = (200, 200, 200)
COLOR_ROBOT_OUTLINE = (150, 150, 150)
COLOR_ROBOT_WHEELS = (50, 50, 50)
COLOR_ROBOT_HEADING = (255, 255, 0)
COLOR_HUMAN_CIRCLE = (255, 220, 180)  # Pale peach/skin tone
COLOR_HUMAN_TRAIL = (200, 170, 140)   # Muted trail color
COLOR_GOAL = (90, 255, 90)
COLOR_GOAL_OUTLINE = (60, 200, 60)
COLOR_PREDICTION = (255, 255, 0)
COLOR_NO_GOAL = (100, 100, 100)
COLOR_X_AXIS = (255, 50, 50)  # Red
COLOR_Y_AXIS = (50, 255, 50)  # Green
COLOR_PLANNED_PATH = (0, 255, 180)  # Cyan-green for planned path
COLOR_PLANNED_PATH_WAYPOINT = (100, 200, 255)  # Light blue for waypoints
COLOR_QUALITY_GOOD = (100, 255, 100)  # Green for good quality
COLOR_QUALITY_MEDIUM = (255, 200, 100)  # Orange for medium quality
COLOR_QUALITY_LOW = (255, 100, 100)  # Red for low quality

# Attention edge colors (matching simulation)
ATTENTION_COLORS = {
    'lidar_to_lidar': (255, 215, 0),      # Yellow
    'robot_to_lidar': (0, 255, 255),      # Cyan
    'robot_to_human': (255, 100, 0),      # Red-Orange
    'human_to_lidar': (0, 255, 0),        # Green
    'human_to_human': (255, 0, 255)       # Magenta
}


# ==================== NEURAL NETWORK MODELS ====================
if TORCH_AVAILABLE:
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
        
        def forward(self, lidar_nodes, robot_nodes, human_nodes, edge_index, return_attention=False):
            lidar_proj = self.lidar_proj(lidar_nodes)
            robot_proj = self.robot_proj(robot_nodes)
            human_proj = self.human_proj(human_nodes)
            
            x = torch.cat([lidar_proj, robot_proj, human_proj], dim=0)
            x_init = x.clone()
            
            if return_attention:
                x, (edge_index_1, alpha_1) = self.g1(x, edge_index, return_attention_weights=True)
                x = F.elu(x)
                x, (edge_index_2, alpha_2) = self.g2(x, edge_index, return_attention_weights=True)
                x = x + x_init
                return x, [(edge_index_1, alpha_1), (edge_index_2, alpha_2)]
            else:
                x = F.elu(self.g1(x, edge_index))
                x = self.g2(x, edge_index)
                x = x + x_init
                return x

    class Actor(nn.Module):
        """Policy network"""
        
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 128)
            self.l3 = nn.Linear(128, action_dim)
        
        def forward(self, s):
            s = F.relu(self.l1(s))
            s = F.relu(self.l2(s))
            return torch.tanh(self.l3(s))


# ==================== EMBEDDING INTERPRETER ====================
class EmbeddingInterpreter:
    """One-line navigation mode, using the fig_cluster_tsne labels."""

    # Same short names and colours as fig_cluster_tsne.pdf
    MODE_COLORS = {
        "Clear Forward": (33, 118, 174),
        "Human Avoidance": (232, 72, 85),
        "Obstacle Avoidance": (247, 179, 43),
        "Narrow Space": (123, 45, 142),
        "Goal Approach": (59, 178, 115),
        "Goal Seeking": (244, 124, 32),
    }

    def classify_state(self, lidar_min, forward_clearance, goal_dist, goal_angle,
                       speed, angular_vel, closest_human):
        """Same priority rules as the t-SNE cluster labels."""
        if lidar_min is None:
            return None, (140, 140, 140)

        if goal_dist < 2.0 and abs(goal_angle) < 0.35:
            label = "Goal Approach"
        elif 0.55 < closest_human < 3.0:
            label = "Human Avoidance"
        elif lidar_min < 2.5 and angular_vel > 0.3 and speed < 0.4:
            label = "Narrow Space"
        elif 1.5 < lidar_min < 3.5 and speed > 0.4:
            label = "Obstacle Avoidance"
        elif forward_clearance > 2.5 and speed > 0.5:
            label = "Clear Forward"
        else:
            label = "Goal Seeking"
        return label, self.MODE_COLORS[label]

    def draw(self, screen, label, color, panel_x, panel_y, font):
        if label is None:
            return
        text = font.render(label, True, (255, 255, 255))
        pad_x, pad_y = 12, 8
        w = text.get_width() + pad_x * 2
        h = text.get_height() + pad_y * 2
        pygame.draw.rect(screen, color, (panel_x, panel_y, w, h))
        screen.blit(text, (panel_x + pad_x, panel_y + pad_y))


# ==================== TRAJECTORY PREDICTOR ====================
class SimpleTrajectoryPredictor:
    """Lightweight trajectory prediction matching deployment script"""
    
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
            return 0.0, 0.0, 0.0
        
        history = self.person_histories[tracking_id]
        if len(history) < 2:
            if len(history) == 1:
                return history[-1][0], history[-1][1], 0.5
            return 0.0, 0.0, 0.0
        
        # Calculate velocity
        recent = list(history)[-5:]
        vx = (recent[-1][0] - recent[0][0]) / max(0.1, recent[-1][2] - recent[0][2])
        vy = (recent[-1][1] - recent[0][1]) / max(0.1, recent[-1][2] - recent[0][2])
        
        # Predict
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
        
        recent = list(history)[-5:]
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.01:
            return 0.0
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        speed = math.sqrt(dx**2 + dy**2) / dt
        
        return speed


# ==================== DIGITAL TWIN VISUALIZER ====================
class DigitalTwinVisualizer:
    """Real-time visualization matching simulation environment"""
    
    def __init__(self, goal_x=None, goal_y=None, ppm=50, show_axes=True, show_edges=True,
                 model_dir=None, model_name="TD3_gat_humans"):
        rospy.init_node('digital_twin_visualizer', anonymous=True)
        
        # Goal position (can be None initially)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.has_goal = (goal_x is not None and goal_y is not None)
        
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
        self.raw_scan_ranges = None
        self.raw_scan_angle_min = -math.pi
        self.raw_scan_angle_increment = None 
        self.detected_persons = []  # List of dicts
        self.cmd_vel_linear = 0.0
        self.cmd_vel_angular = 0.0
        
        # Trajectory predictor
        self.trajectory_predictor = SimpleTrajectoryPredictor()
        
        # Person tracking for trails
        self.person_trails = {}  # {tracking_id: deque([(x, y), ...])}
        self.max_trail_length = 20
        
        # Visualization settings
        self.ppm = ppm  # pixels per meter
        self.view_center_x = 0.0  # Robot-centered view
        self.view_center_y = 0.0
        self.view_width = 20.0  # meters (adaptive based on LiDAR)
        self.view_height = 20.0
        self.show_axes = show_axes  # Toggle coordinate axes
        self.show_edges = show_edges  # Toggle graph edges (press 'E' to toggle)
        self.show_features = True  # Toggle embedding features panel (press 'F')
        
        # GAT model components
        self.L = LIDAR_BEAMS  # 20
        self.R = 1  # Robot node
        self.H = MAX_HUMANS  # 3
        self.gat = None
        self.actor = None
        self.edge_index = None
        self.gat_enabled = False
        
        # GAT inference results
        self.current_embedding = None
        self.current_attention = None
        self.attention_map = {}  # For edge coloring
        self.current_action = np.array([0.0, 0.0])
        
        # Embedding interpreter
        self.embedding_interpreter = EmbeddingInterpreter()
        
        # Load models if available
        if model_dir and TORCH_AVAILABLE:
            self.load_gat_models(model_dir, model_name)
        
        # Pygame setup
        pygame.init()
        self.screen_width = 1728
        self.screen_height = 972
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Digital Twin - Husky GAT Navigation")
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font = pygame.font.SysFont('Arial', 14)
            self.font_large = pygame.font.SysFont('Arial', 18, bold=True)
            self.font_small = pygame.font.SysFont('Arial', 11)
        except:
            self.font = pygame.font.Font(None, 18)
            self.font_large = pygame.font.Font(None, 22)
            self.font_small = pygame.font.Font(None, 14)
        
        # ROS subscribers
        rospy.Subscriber('/husky/scan_raw', LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.raw_scan_callback, queue_size=1)  # NEW: Full scan
        rospy.Subscriber('/husky/odom_raw', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/person_detections', Float32MultiArray, self.camera_callback, queue_size=1)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        
        # Optional: Subscribe to goal updates (e.g., from move_base or custom topic)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        
        # Subscribe to planned path from deploy_v5_recovery.py
        rospy.Subscriber('/gat_planned_path', Path, self.planned_path_callback, queue_size=1)
        rospy.Subscriber('/gat_navigation_stats', String, self.nav_stats_callback, queue_size=1)
        rospy.Subscriber('/gat_quality_score', Float32, self.quality_callback, queue_size=1)
        
        # Planned path storage
        self.planned_path = []  # List of (x, y) world coordinates
        self.nav_stats = {}
        self.quality_score = 0.0
        self.show_planned_path = True  # Toggle with 'P' key
        
        # Timing
        self.last_render_time = time.time()
        self.fps = 0
        
        rospy.loginfo("Digital Twin Visualizer initialized")
        if self.has_goal:
            rospy.loginfo(f"Initial Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        else:
            rospy.loginfo("No initial goal - waiting for goal topic or manual specification")
        rospy.loginfo(f"Coordinate axes: {'ON' if self.show_axes else 'OFF'}")
        rospy.loginfo(f"Graph edges: {'ON' if self.show_edges else 'OFF'}")
        rospy.loginfo(f"GAT inference: {'ENABLED' if self.gat_enabled else 'DISABLED'}")
        rospy.loginfo(f"Planned path: {'ON' if self.show_planned_path else 'OFF'}")
        rospy.loginfo("Subscribed to /gat_planned_path for path visualization")
    
    def goal_callback(self, msg):
        """Update goal position from ROS topic"""
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True
        rospy.loginfo(f"Goal updated: ({self.goal_x:.2f}, {self.goal_y:.2f})")
    
    def planned_path_callback(self, msg):
        """Update planned path from deploy_v5_recovery.py"""
        self.planned_path = []
        for pose in msg.poses:
            self.planned_path.append((pose.pose.position.x, pose.pose.position.y))
    
    def nav_stats_callback(self, msg):
        """Update navigation statistics"""
        try:
            self.nav_stats = json.loads(msg.data)
        except:
            pass
    
    def quality_callback(self, msg):
        """Update quality score"""
        self.quality_score = msg.data
    
    def load_gat_models(self, model_dir, model_name="TD3_gat_humans"):
        """Load GAT and Actor models for inference"""
        if not TORCH_AVAILABLE:
            rospy.logwarn("PyTorch not available - GAT features disabled")
            return False
        
        gat_path = os.path.join(model_dir, f"{model_name}_gat.pth")
        actor_path = os.path.join(model_dir, f"{model_name}_actor.pth")
        
        if not os.path.exists(gat_path) or not os.path.exists(actor_path):
            rospy.logwarn(f"GAT models not found in {model_dir}")
            return False
        
        try:
            # Initialize networks
            self.gat = GATNetwork(
                lidar_channels=4, robot_channels=4, human_channels=6,
                out_channels=16, num_heads=4,
                lidar_nodes=self.L, robot_nodes=self.R, human_nodes=self.H
            ).to(device)
            
            state_dim = 16 * (self.L + self.R + self.H)  # 24 * 16 = 384
            self.actor = Actor(state_dim, action_dim=2).to(device)
            
            # Load weights
            self.gat.load_state_dict(torch.load(gat_path, map_location=device))
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            
            self.gat.eval()
            self.actor.eval()
            
            # Create edge index
            self.edge_index = create_edge_index(self.L, self.R, self.H).to(device)
            
            self.gat_enabled = True
            rospy.loginfo(f"GAT models loaded from {model_dir}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to load GAT models: {e}")
            return False
    
    def build_gat_observation(self):
        """Build observation tensors for GAT inference"""
        if self.lidar_ranges is None:
            return None, None, None
        
        # LiDAR nodes: [normalized_distance, cos(angle), sin(angle), 0]
        lidar = self.downsample_lidar(self.lidar_ranges)
        
        lidar_nodes = []
        for dist, ang in zip(lidar, BEAM_ANGLES):
            lidar_nodes.append([
                dist / LIDAR_RANGE,
                math.cos(ang),
                math.sin(ang),
                0.0
            ])
        lidar_nodes = np.array(lidar_nodes, dtype=np.float32)
        
        # Robot node: [goal_x_ego, goal_y_ego, linear_vel, angular_vel]
        if self.has_goal:
            dx_world = self.goal_x - self.robot_x
            dy_world = self.goal_y - self.robot_y
            dx_ego = dx_world * math.cos(self.robot_theta) + dy_world * math.sin(self.robot_theta)
            dy_ego = -dx_world * math.sin(self.robot_theta) + dy_world * math.cos(self.robot_theta)
        else:
            dx_ego, dy_ego = 0.0, 0.0
        
        world_max = 20.0  # Match simulation
        robot_node = np.array([[
            dx_ego / world_max,
            dy_ego / world_max,
            self.robot_vx,
            self.robot_wz
        ]], dtype=np.float32)
        
        # Human nodes: [x_rel/cam_max, y_rel/cam_max, speed, conf, pred_x, pred_y]
        cam_max = 15.0
        human_nodes = np.zeros((self.H, 6), dtype=np.float32)
        
        for i, person in enumerate(self.detected_persons[:self.H]):
            # Transform to robot frame
            dx = person['x_odom'] - self.robot_x
            dy = person['y_odom'] - self.robot_y
            x_rel = dx * math.cos(-self.robot_theta) - dy * math.sin(-self.robot_theta)
            y_rel = dx * math.sin(-self.robot_theta) + dy * math.cos(-self.robot_theta)
            
            # GAT future-position feature: 100% CV at sim waypoint (~0.31s)
            pred_x, pred_y, conf = self.trajectory_predictor.predict(
                person['tracking_id'], dt=GAT_PRED_HORIZON
            )
            pred_dx = pred_x - self.robot_x
            pred_dy = pred_y - self.robot_y
            pred_x_rel = pred_dx * math.cos(-self.robot_theta) - pred_dy * math.sin(-self.robot_theta)
            pred_y_rel = pred_dx * math.sin(-self.robot_theta) + pred_dy * math.cos(-self.robot_theta)
            
            # Estimate speed from trajectory history
            speed = self.trajectory_predictor.get_speed(person['tracking_id']) if hasattr(self.trajectory_predictor, 'get_speed') else 0.5
            
            human_nodes[i] = [
                x_rel / cam_max,
                y_rel / cam_max,
                min(speed / 1.5, 1.0),
                person['confidence'],
                pred_x_rel / cam_max,
                pred_y_rel / cam_max
            ]
        
        return lidar_nodes, robot_node, human_nodes
    
    @torch.no_grad()
    def run_gat_inference(self):
        """Run GAT inference and update attention/embedding"""
        if not self.gat_enabled:
            return
        
        lidar_nodes, robot_node, human_nodes = self.build_gat_observation()
        if lidar_nodes is None:
            return
        
        try:
            # Convert to tensors
            lidar_t = torch.tensor(lidar_nodes, dtype=torch.float32, device=device)
            robot_t = torch.tensor(robot_node, dtype=torch.float32, device=device)
            human_t = torch.tensor(human_nodes, dtype=torch.float32, device=device)
            
            # Run GAT with attention
            z, attention_weights = self.gat(lidar_t, robot_t, human_t, 
                                           self.edge_index, return_attention=True)
            
            # Get robot embedding (node index L)
            self.current_embedding = z[self.L].cpu().numpy()
            
            # Get action
            self.current_action = self.actor(z.view(1, -1)).cpu().numpy().flatten()
            
            # Process attention for edge coloring
            self.attention_map = self.filter_attention_weights(attention_weights)
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"GAT inference error: {e}")
    
    def filter_attention_weights(self, attention_weights, top_k=5):
        """Filter attention weights to highlight top-k edges per category"""
        if len(attention_weights) < 2:
            return {}
        
        edge_index, alpha = attention_weights[1]  # Layer 2
        
        if hasattr(edge_index, 'cpu'):
            edge_index = edge_index.cpu().numpy()
        if hasattr(alpha, 'cpu'):
            alpha = alpha.cpu().numpy()
        
        alpha_avg = alpha.mean(axis=1) if len(alpha.shape) > 1 else alpha
        
        # Categorize edges
        edge_groups = {
            'lidar_to_lidar': [],
            'robot_to_lidar': [],
            'robot_to_human': [],
            'human_to_lidar': [],
            'human_to_human': []
        }
        
        for i in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, i])
            tgt_idx = int(edge_index[1, i])
            attention = float(alpha_avg[i])
            
            if src_idx < self.L and tgt_idx < self.L:
                edge_groups['lidar_to_lidar'].append((src_idx, tgt_idx, attention))
            elif (src_idx == self.L and tgt_idx < self.L) or (tgt_idx == self.L and src_idx < self.L):
                edge_groups['robot_to_lidar'].append((src_idx, tgt_idx, attention))
            elif (src_idx == self.L and tgt_idx > self.L) or (tgt_idx == self.L and src_idx > self.L):
                edge_groups['robot_to_human'].append((src_idx, tgt_idx, attention))
            elif (src_idx < self.L and tgt_idx > self.L) or (tgt_idx < self.L and src_idx > self.L):
                edge_groups['human_to_lidar'].append((src_idx, tgt_idx, attention))
            elif src_idx > self.L and tgt_idx > self.L:
                edge_groups['human_to_human'].append((src_idx, tgt_idx, attention))
        
        # Process each category: keep top-k edges
        filtered_map = {}
        for group_name, edges in edge_groups.items():
            if len(edges) == 0:
                continue
            
            attentions = [e[2] for e in edges]
            if len(set(attentions)) <= 1:
                continue
            
            sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
            k = min(top_k, len(sorted_edges))
            top_edges = sorted_edges[:k]
            
            top_attentions = [e[2] for e in top_edges]
            alpha_min = min(top_attentions)
            alpha_max = max(top_attentions)
            
            for src_idx, tgt_idx, attention in top_edges:
                if alpha_max > alpha_min:
                    alpha_norm = (attention - alpha_min) / (alpha_max - alpha_min)
                else:
                    alpha_norm = 1.0
                
                filtered_map[(src_idx, tgt_idx)] = alpha_norm
                filtered_map[(tgt_idx, src_idx)] = alpha_norm
        
        return filtered_map

    def raw_scan_callback(self, msg):
        """Process full raw LiDAR scan for boundary visualization"""
        self.raw_scan_ranges = np.array(msg.ranges)
        self.raw_scan_angle_min = msg.angle_min
        if len(msg.ranges) > 1 and msg.angle_increment != 0.0:
            self.raw_scan_angle_increment = msg.angle_increment
    
    def lidar_callback(self, msg):
        """Process front-180° LiDAR used for GAT nodes"""
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_angle_min = msg.angle_min
        if len(msg.ranges) > 1 and msg.angle_increment != 0.0:
            self.lidar_angle_increment = msg.angle_increment
    
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
                    
                    # Transform to robot frame
                    robot_x_rel = cam_z
                    robot_y_rel = -cam_x
                    
                    # Transform to odom frame
                    person_x_odom = self.robot_x + robot_x_rel * math.cos(self.robot_theta) - robot_y_rel * math.sin(self.robot_theta)
                    person_y_odom = self.robot_y + robot_x_rel * math.sin(self.robot_theta) + robot_y_rel * math.cos(self.robot_theta)
                    
                    # Update trajectory predictor
                    self.trajectory_predictor.update(tracking_id, person_x_odom, person_y_odom)
                    
                    # Update trails
                    if tracking_id not in self.person_trails:
                        self.person_trails[tracking_id] = deque(maxlen=self.max_trail_length)
                    self.person_trails[tracking_id].append((person_x_odom, person_y_odom))
                    
                    self.detected_persons.append({
                        'x_odom': person_x_odom,
                        'y_odom': person_y_odom,
                        'confidence': conf,
                        'tracking_id': tracking_id
                    })
        except Exception as e:
            rospy.logwarn(f"Error processing camera data: {e}")
    
    def cmd_vel_callback(self, msg):
        """Track commanded velocities"""
        self.cmd_vel_linear = msg.linear.x
        self.cmd_vel_angular = msg.angular.z
    
    def downsample_lidar(self, ranges, return_validity=False):
        """20 sim beams over front 180°, nearest real ray in that direction."""
        if ranges is None or len(ranges) == 0:
            downsampled = np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float32)
            valid_mask = np.zeros(LIDAR_BEAMS, dtype=bool)
        else:
            downsampled = sample_sim_beams(
                ranges, self.lidar_angle_min, self.lidar_angle_increment,
                max_range=LIDAR_RANGE,
            )
            valid_mask = downsampled < (LIDAR_RANGE - 0.1)
        
        if return_validity:
            return downsampled, valid_mask
        return downsampled
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen pixels (robot-centered with rotation)"""
        # Translate to robot-relative coordinates
        rel_x = x - self.robot_x
        rel_y = y - self.robot_y
        
        # Rotate so robot always faces "up" (negative theta to rotate world opposite)
        # Subtract pi/2 to correct for 90° sensor frame offset
        cos_theta = math.cos(-self.robot_theta + math.pi/2)
        sin_theta = math.sin(-self.robot_theta + math.pi/2)
        
        rotated_x = rel_x * cos_theta - rel_y * sin_theta
        rotated_y = rel_x * sin_theta + rel_y * cos_theta
        
        # Convert to screen coordinates (Y inverted for screen)
        screen_x = int(self.screen_width / 2 + rotated_x * self.ppm)
        screen_y = int(self.screen_height * 0.75 - rotated_y * self.ppm)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen pixels to world coordinates (reverse of world_to_screen)"""
        # Reverse the screen coordinate conversion
        rotated_x = (screen_x - self.screen_width / 2) / self.ppm
        rotated_y = (self.screen_height * 0.75 - screen_y) / self.ppm
        
        # Reverse the rotation
        cos_theta = math.cos(-self.robot_theta + math.pi/2)
        sin_theta = math.sin(-self.robot_theta + math.pi/2)
        
        # Inverse rotation matrix
        rel_x = rotated_x * cos_theta + rotated_y * sin_theta
        rel_y = -rotated_x * sin_theta + rotated_y * cos_theta
        
        # Add robot position back
        world_x = rel_x + self.robot_x
        world_y = rel_y + self.robot_y
        
        return world_x, world_y
    
    def draw_cursor_position(self):
        """Draw world coordinates at cursor position"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Convert to world coordinates
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
        
        # Format the coordinate text
        coord_text = f"({world_x:.2f}, {world_y:.2f})"
        
        # Create background box for better visibility
        text_surface = self.font.render(coord_text, True, (255, 255, 255))
        text_width = text_surface.get_width()
        text_height = text_surface.get_height()
        
        # Position label slightly offset from cursor
        label_x = mouse_x + 15
        label_y = mouse_y - 10
        
        # Keep label on screen
        if label_x + text_width + 10 > self.screen_width:
            label_x = mouse_x - text_width - 20
        if label_y < 5:
            label_y = mouse_y + 20
        
        # Draw background box
        padding = 4
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (label_x - padding, label_y - padding, 
                         text_width + padding * 2, text_height + padding * 2))
        pygame.draw.rect(self.screen, (100, 200, 100), 
                        (label_x - padding, label_y - padding, 
                         text_width + padding * 2, text_height + padding * 2), 1)
        
        # Draw text
        self.screen.blit(text_surface, (label_x, label_y))
        
        # Draw small crosshair at cursor
        pygame.draw.line(self.screen, (100, 200, 100), 
                        (mouse_x - 8, mouse_y), (mouse_x + 8, mouse_y), 1)
        pygame.draw.line(self.screen, (100, 200, 100), 
                        (mouse_x, mouse_y - 8), (mouse_x, mouse_y + 8), 1)

    def draw_coordinate_axes(self):
        """Draw X and Y coordinate axes in bottom-right corner for reference"""
        if not self.show_axes:
            return
        
        # Fixed position in bottom-right corner
        corner_x = self.screen_width - 120
        corner_y = self.screen_height - 120
        axis_length = 60  # pixels
        
        # Background box
        box_size = 110
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (corner_x - 15, corner_y - 15, box_size, box_size))
        pygame.draw.rect(self.screen, (80, 80, 80), 
                        (corner_x - 15, corner_y - 15, box_size, box_size), 2)
        
        # Calculate axis directions based on robot heading
        # The view rotates with the robot, so we need to show how world axes appear
        # Subtract pi/2 to account for the view rotation offset
        view_rotation = -self.robot_theta + math.pi/2
        
        # X-axis direction in screen space
        x_dir_x = math.cos(-view_rotation)
        x_dir_y = -math.sin(-view_rotation)  # Negative because screen Y is inverted
        
        # Y-axis direction in screen space
        y_dir_x = math.cos(-view_rotation + math.pi/2)
        y_dir_y = -math.sin(-view_rotation + math.pi/2)
        
        # Draw origin point
        pygame.draw.circle(self.screen, (200, 200, 200), (corner_x, corner_y), 4, 0)
        
        # Draw X-axis (Red)
        x_end_x = int(corner_x + axis_length * x_dir_x)
        x_end_y = int(corner_y + axis_length * x_dir_y)
        pygame.draw.line(self.screen, COLOR_X_AXIS, (corner_x, corner_y), (x_end_x, x_end_y), 3)
        
        # X-axis arrowhead
        arrow_size = 10
        x_arrow_angle = math.atan2(x_end_y - corner_y, x_end_x - corner_x)
        x_arrow_points = [
            (x_end_x, x_end_y),
            (x_end_x - arrow_size * math.cos(x_arrow_angle - math.pi/6),
             x_end_y - arrow_size * math.sin(x_arrow_angle - math.pi/6)),
            (x_end_x - arrow_size * math.cos(x_arrow_angle + math.pi/6),
             x_end_y - arrow_size * math.sin(x_arrow_angle + math.pi/6))
        ]
        pygame.draw.polygon(self.screen, COLOR_X_AXIS, x_arrow_points)
        
        # X label
        x_label = self.font.render("X", True, COLOR_X_AXIS)
        self.screen.blit(x_label, (x_end_x + 5, x_end_y - 8))
        
        # Draw Y-axis (Green)
        y_end_x = int(corner_x + axis_length * y_dir_x)
        y_end_y = int(corner_y + axis_length * y_dir_y)
        pygame.draw.line(self.screen, COLOR_Y_AXIS, (corner_x, corner_y), (y_end_x, y_end_y), 3)
        
        # Y-axis arrowhead
        y_arrow_angle = math.atan2(y_end_y - corner_y, y_end_x - corner_x)
        y_arrow_points = [
            (y_end_x, y_end_y),
            (y_end_x - arrow_size * math.cos(y_arrow_angle - math.pi/6),
             y_end_y - arrow_size * math.sin(y_arrow_angle - math.pi/6)),
            (y_end_x - arrow_size * math.cos(y_arrow_angle + math.pi/6),
             y_end_y - arrow_size * math.sin(y_arrow_angle + math.pi/6))
        ]
        pygame.draw.polygon(self.screen, COLOR_Y_AXIS, y_arrow_points)
        
        # Y label
        y_label = self.font.render("Y", True, COLOR_Y_AXIS)
        self.screen.blit(y_label, (y_end_x + 5, y_end_y - 8))
        
        # Robot position text
        pos_text = self.font_small.render(f"({self.robot_x:.1f}, {self.robot_y:.1f})", True, (150, 150, 150))
        self.screen.blit(pos_text, (corner_x - 10, corner_y + 15))
    
    def draw_lidar_outline(self):
        """Draw environment boundary using full 360° Velodyne LiDAR scan as points"""
        if self.raw_scan_ranges is None:
            return
        
        # Use full raw scan (not downsampled)
        ranges = np.array(self.raw_scan_ranges)
        num_beams = len(ranges)
        
        # Filter out invalid readings
        valid_mask = ~np.isnan(ranges) & ~np.isinf(ranges) & (ranges > 0.1) & (ranges < LIDAR_RANGE)
        
        # Use the LaserScan angle fields so dots sit on the real rays
        if self.raw_scan_angle_increment is not None:
            angles = self.raw_scan_angle_min + np.arange(num_beams) * self.raw_scan_angle_increment
        else:
            angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)
        
        # Draw each valid scan point
        for i in range(num_beams):
            if valid_mask[i]:
                dist = ranges[i]
                angle = angles[i]
                
                # Transform to world coordinates
                # Add robot's heading to align with robot's orientation in world
                beam_angle = angle + self.robot_theta
                world_x = self.robot_x + dist * math.cos(beam_angle)
                world_y = self.robot_y + dist * math.sin(beam_angle)
                
                # Convert to screen coordinates (automatically handles rotation)
                screen_x, screen_y = self.world_to_screen(world_x, world_y)
                
                # Only draw if point is visible on screen
                if 0 <= screen_x < self.screen_width and 0 <= screen_y < self.screen_height:
                    # Draw point - slightly larger for better visibility
                    pygame.draw.circle(self.screen, COLOR_BOUNDARY, (screen_x, screen_y), 2, 0)

    
    def draw_lidar_beams_and_nodes(self):
        """Draw LiDAR beams and node positions (only for valid readings)"""
        if self.lidar_ranges is None:
            return
        
        lidar, valid_mask = self.downsample_lidar(self.lidar_ranges, return_validity=True)
        robot_screen_x, robot_screen_y = self.world_to_screen(self.robot_x, self.robot_y)
        
        for i, dist in enumerate(lidar):
            # Skip nodes at max range (invalid/no obstacle detected)
            if not valid_mask[i]:
                continue
                
            beam_angle = BEAM_ANGLES[i] + self.robot_theta
            node_x = self.robot_x + dist * math.cos(beam_angle)
            node_y = self.robot_y + dist * math.sin(beam_angle)
            node_screen_x, node_screen_y = self.world_to_screen(node_x, node_y)
            
            # Draw beam line
            pygame.draw.line(self.screen, COLOR_LIDAR_BEAM,
                           (robot_screen_x, robot_screen_y),
                           (node_screen_x, node_screen_y), 1)
            
            # Draw node circle
            pygame.draw.circle(self.screen, COLOR_LIDAR_NODE,
                             (node_screen_x, node_screen_y), 6, 0)
    
    def get_lidar_node_positions(self):
        """Get screen positions of valid LiDAR nodes (not at max range)"""
        if self.lidar_ranges is None:
            return []
        
        lidar, valid_mask = self.downsample_lidar(self.lidar_ranges, return_validity=True)
        positions = []
        
        for i, dist in enumerate(lidar):
            # Only include valid nodes (not at max range)
            if not valid_mask[i]:
                continue
                
            beam_angle = BEAM_ANGLES[i] + self.robot_theta
            node_x = self.robot_x + dist * math.cos(beam_angle)
            node_y = self.robot_y + dist * math.sin(beam_angle)
            positions.append(self.world_to_screen(node_x, node_y))
        
        return positions
    
    def get_human_node_positions(self):
        """Get screen positions of detected humans (up to MAX_HUMANS)"""
        positions = []
        for person in self.detected_persons[:MAX_HUMANS]:
            pos = self.world_to_screen(person['x_odom'], person['y_odom'])
            positions.append(pos)
        return positions
    
    def get_attention_edge_color(self, src_idx, tgt_idx, base_color, default_color=(60, 60, 60)):
        """Get edge color based on attention weight"""
        attention = self.attention_map.get((src_idx, tgt_idx), None)
        if attention is None:
            attention = self.attention_map.get((tgt_idx, src_idx), None)
        
        if attention is not None:
            # Interpolate from gray to category color based on attention
            return (
                int(default_color[0] + (base_color[0] - default_color[0]) * attention),
                int(default_color[1] + (base_color[1] - default_color[1]) * attention),
                int(default_color[2] + (base_color[2] - default_color[2]) * attention)
            ), max(1, int(1 + 3 * attention))
        return default_color, 1
    
    def draw_graph_edges(self):
        """Draw all graph edges with attention-weighted colors"""
        if not self.show_edges:
            return
        
        # Get node positions with indices
        if self.lidar_ranges is None:
            return
        
        lidar, valid_mask = self.downsample_lidar(self.lidar_ranges, return_validity=True)
        
        # Build lidar positions with their original indices (for attention lookup)
        lidar_positions_with_idx = []
        for i, dist in enumerate(lidar):
            if valid_mask[i]:
                beam_angle = BEAM_ANGLES[i] + self.robot_theta
                node_x = self.robot_x + dist * math.cos(beam_angle)
                node_y = self.robot_y + dist * math.sin(beam_angle)
                lidar_positions_with_idx.append((i, self.world_to_screen(node_x, node_y)))
        
        robot_pos = self.world_to_screen(self.robot_x, self.robot_y)
        robot_idx = self.L  # Robot node index
        
        # Human positions with indices
        human_positions_with_idx = []
        for i, person in enumerate(self.detected_persons[:MAX_HUMANS]):
            pos = self.world_to_screen(person['x_odom'], person['y_odom'])
            human_positions_with_idx.append((self.L + 1 + i, pos))
        
        if not lidar_positions_with_idx:
            return
        
        default_color = (60, 60, 60)
        
        # 1. LiDAR-to-LiDAR edges (fully connected)
        base_color = ATTENTION_COLORS['lidar_to_lidar']
        for i, (idx_i, pos_i) in enumerate(lidar_positions_with_idx):
            for j, (idx_j, pos_j) in enumerate(lidar_positions_with_idx):
                if i >= j:
                    continue
                color, width = self.get_attention_edge_color(idx_i, idx_j, base_color, default_color)
                pygame.draw.line(self.screen, color, pos_i, pos_j, width)
        
        # 2. Robot-to-LiDAR edges
        base_color = ATTENTION_COLORS['robot_to_lidar']
        for idx, lidar_pos in lidar_positions_with_idx:
            color, width = self.get_attention_edge_color(robot_idx, idx, base_color, default_color)
            pygame.draw.line(self.screen, color, robot_pos, lidar_pos, width)
        
        # 3. Robot-to-Human edges
        base_color = ATTENTION_COLORS['robot_to_human']
        for human_idx, human_pos in human_positions_with_idx:
            color, width = self.get_attention_edge_color(robot_idx, human_idx, base_color, default_color)
            pygame.draw.line(self.screen, color, robot_pos, human_pos, max(width, 2))
        
        # 4. Human-to-LiDAR edges
        base_color = ATTENTION_COLORS['human_to_lidar']
        for human_idx, human_pos in human_positions_with_idx:
            for lidar_idx, lidar_pos in lidar_positions_with_idx:
                color, width = self.get_attention_edge_color(human_idx, lidar_idx, base_color, default_color)
                pygame.draw.line(self.screen, color, human_pos, lidar_pos, width)
        
        # 5. Human-to-Human edges
        base_color = ATTENTION_COLORS['human_to_human']
        for i, (idx_i, pos_i) in enumerate(human_positions_with_idx):
            for j, (idx_j, pos_j) in enumerate(human_positions_with_idx):
                if i >= j:
                    continue
                color, width = self.get_attention_edge_color(idx_i, idx_j, base_color, default_color)
                pygame.draw.line(self.screen, color, pos_i, pos_j, max(width, 2))
    
    def draw_robot(self):
        """Draw Husky robot body (matching simulation) - always facing up"""
        robot_screen_x, robot_screen_y = self.world_to_screen(self.robot_x, self.robot_y)
        
        # Robot now always faces "up" (0 degrees in screen space = pi/2 in standard coords)
        # Since world rotates around robot, robot is always at heading 0 in screen space
        screen_heading = math.pi / 2  # Facing up on screen
        
        # Body rectangle
        half_length = HUSKY_LENGTH / 2
        half_width = HUSKY_WIDTH / 2
        corners = [(-half_length, -half_width), (half_length, -half_width),
                  (half_length, half_width), (-half_length, half_width)]
        
        screen_corners = []
        for lx, ly in corners:
            # Rotate by screen_heading
            screen_x = robot_screen_x + (lx * math.cos(screen_heading) - ly * math.sin(screen_heading)) * self.ppm
            screen_y = robot_screen_y - (lx * math.sin(screen_heading) + ly * math.cos(screen_heading)) * self.ppm
            screen_corners.append((int(screen_x), int(screen_y)))
        
        pygame.draw.polygon(self.screen, COLOR_ROBOT_BODY, screen_corners)
        pygame.draw.polygon(self.screen, COLOR_ROBOT_OUTLINE, screen_corners, 3)
        
        # Wheels
        wheel_positions = [
            (HUSKY_WHEELBASE * 0.45, HUSKY_TRACK_WIDTH / 2),
            (HUSKY_WHEELBASE * 0.45, -HUSKY_TRACK_WIDTH / 2),
            (-HUSKY_WHEELBASE * 0.45, HUSKY_TRACK_WIDTH / 2),
            (-HUSKY_WHEELBASE * 0.45, -HUSKY_TRACK_WIDTH / 2)
        ]
        
        wheel_radius_pixels = int(HUSKY_WHEEL_RADIUS * self.ppm)
        for wheel_lx, wheel_ly in wheel_positions:
            wheel_screen_x = robot_screen_x + (wheel_lx * math.cos(screen_heading) - wheel_ly * math.sin(screen_heading)) * self.ppm
            wheel_screen_y = robot_screen_y - (wheel_lx * math.sin(screen_heading) + wheel_ly * math.cos(screen_heading)) * self.ppm
            pygame.draw.circle(self.screen, COLOR_ROBOT_WHEELS,
                             (int(wheel_screen_x), int(wheel_screen_y)), wheel_radius_pixels)
        
        # Sensor payload
        sensor_length = HUSKY_LENGTH * 0.4
        sensor_width = HUSKY_WIDTH * 0.6
        sensor_corners = [
            (-sensor_length/2, -sensor_width/2), (sensor_length/2, -sensor_width/2),
            (sensor_length/2, sensor_width/2), (-sensor_length/2, sensor_width/2)
        ]
        
        sensor_screen_corners = []
        for lx, ly in sensor_corners:
            screen_x = robot_screen_x + (lx * math.cos(screen_heading) - ly * math.sin(screen_heading)) * self.ppm
            screen_y = robot_screen_y - (lx * math.sin(screen_heading) + ly * math.cos(screen_heading)) * self.ppm
            sensor_screen_corners.append((int(screen_x), int(screen_y)))
        
        pygame.draw.polygon(self.screen, (120, 120, 120), sensor_screen_corners)
        
        # Center dot and heading arrow
        pygame.draw.circle(self.screen, (255, 255, 255), (robot_screen_x, robot_screen_y), 4, 0)
        
        arrow_length = HUSKY_LENGTH * 0.8
        arrow_end_screen_x = robot_screen_x + arrow_length * math.cos(screen_heading) * self.ppm
        arrow_end_screen_y = robot_screen_y - arrow_length * math.sin(screen_heading) * self.ppm
        
        pygame.draw.line(self.screen, COLOR_ROBOT_HEADING,
                        (robot_screen_x, robot_screen_y),
                        (int(arrow_end_screen_x), int(arrow_end_screen_y)), 4)
        
        # Arrowhead
        arrow_size = HUSKY_LENGTH * 0.15 * self.ppm
        left_arrow_x = arrow_end_screen_x - arrow_size * math.cos(screen_heading - math.pi/6)
        left_arrow_y = arrow_end_screen_y + arrow_size * math.sin(screen_heading - math.pi/6)
        right_arrow_x = arrow_end_screen_x - arrow_size * math.cos(screen_heading + math.pi/6)
        right_arrow_y = arrow_end_screen_y + arrow_size * math.sin(screen_heading + math.pi/6)
        
        pygame.draw.polygon(self.screen, COLOR_ROBOT_HEADING, [
            (int(arrow_end_screen_x), int(arrow_end_screen_y)),
            (int(left_arrow_x), int(left_arrow_y)),
            (int(right_arrow_x), int(right_arrow_y))
        ])
    
    def draw_humans(self):
        """Draw detected humans with trails and predictions"""
        # Draw trails
        for tracking_id, trail in self.person_trails.items():
            if len(trail) > 1:
                trail_points = [self.world_to_screen(x, y) for (x, y) in trail]
                pygame.draw.lines(self.screen, COLOR_HUMAN_TRAIL, False, trail_points, 2)
        
        # Draw human circles with 'H' label
        for person in self.detected_persons:
            person_screen_x, person_screen_y = self.world_to_screen(
                person['x_odom'], person['y_odom']
            )
            radius = int(0.5 * self.ppm)  # 0.5m radius
            
            # Draw filled circle (pale color)
            pygame.draw.circle(self.screen, COLOR_HUMAN_CIRCLE,
                             (person_screen_x, person_screen_y), radius, 0)
            # Draw darker outline
            pygame.draw.circle(self.screen, (180, 140, 100),
                             (person_screen_x, person_screen_y), radius, 2)
            
            # Draw 'H' label in center
            h_font = pygame.font.SysFont('Arial', int(radius * 1.2), bold=True)
            h_text = h_font.render('H', True, (80, 50, 30))  # Dark brown text
            text_rect = h_text.get_rect(center=(person_screen_x, person_screen_y))
            self.screen.blit(h_text, text_rect)
        
        # Draw predictions
        for person in self.detected_persons:
            pred_x, pred_y, conf = self.trajectory_predictor.predict(
                person['tracking_id'], dt=1.0
            )
            if conf > 0.3:
                # Draw prediction line
                pred_screen_x, pred_screen_y = self.world_to_screen(pred_x, pred_y)
                person_screen_x, person_screen_y = self.world_to_screen(
                    person['x_odom'], person['y_odom']
                )
                pygame.draw.line(self.screen, COLOR_PREDICTION,
                               (person_screen_x, person_screen_y),
                               (pred_screen_x, pred_screen_y), 3)
                
                # Draw uncertainty circle
                uncertainty_radius = int(0.3 * self.ppm * (2 - conf))
                pygame.draw.circle(self.screen, COLOR_PREDICTION,
                                 (pred_screen_x, pred_screen_y),
                                 uncertainty_radius, 1)
    
    def draw_goal(self):
        """Draw goal star if available"""
        if not self.has_goal:
            # Draw "No Goal" indicator in corner
            text = self.font.render("No Goal Set", True, COLOR_NO_GOAL)
            self.screen.blit(text, (self.screen_width - 150, self.screen_height - 40))
            return
        
        goal_screen_x, goal_screen_y = self.world_to_screen(self.goal_x, self.goal_y)
        star_radius = int(0.375 * self.ppm)
        inner_radius = int(star_radius * 0.4)
        star_points = []
        
        for i in range(10):
            angle = math.pi / 2 + (i * math.pi / 5)
            radius = star_radius if i % 2 == 0 else inner_radius
            px = goal_screen_x + int(radius * math.cos(angle))
            py = goal_screen_y - int(radius * math.sin(angle))
            star_points.append((px, py))
        
        pygame.draw.polygon(self.screen, COLOR_GOAL, star_points, 0)
        pygame.draw.polygon(self.screen, COLOR_GOAL_OUTLINE, star_points, 2)
    
    def draw_planned_path(self):
        """Draw the planned path from continuous planning"""
        if not self.show_planned_path or not self.planned_path or len(self.planned_path) < 2:
            return
        
        # Filter path to only show points AHEAD of the robot (not behind)
        # We do this by finding the closest point to current robot position
        # and only showing points from there onward
        filtered_path = []
        robot_pos = (self.robot_x, self.robot_y)
        
        # Find the closest point index in the path
        min_dist = float('inf')
        start_idx = 0
        for i, (x, y) in enumerate(self.planned_path):
            dist = math.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                start_idx = i
        
        # Take points from closest point onward (path ahead)
        filtered_path = self.planned_path[start_idx:]
        
        if len(filtered_path) < 2:
            # If too few points ahead, just draw line to goal
            if self.has_goal:
                robot_sx, robot_sy = self.world_to_screen(self.robot_x, self.robot_y)
                goal_sx, goal_sy = self.world_to_screen(self.goal_x, self.goal_y)
                pygame.draw.line(self.screen, COLOR_PLANNED_PATH, 
                               (robot_sx, robot_sy), (goal_sx, goal_sy), 2)
            return
        
        # Convert world coordinates to screen coordinates
        screen_points = []
        for x, y in filtered_path:
            sx, sy = self.world_to_screen(x, y)
            # Only include points that are on screen
            if -100 < sx < self.screen_width + 100 and -100 < sy < self.screen_height + 100:
                screen_points.append((sx, sy))
        
        if len(screen_points) < 2:
            return
        
        # Determine color based on quality score
        if self.quality_score > 0.7:
            path_color = COLOR_QUALITY_GOOD
        elif self.quality_score > 0.4:
            path_color = COLOR_QUALITY_MEDIUM
        else:
            path_color = COLOR_PLANNED_PATH
        
        # Draw line from robot to first path point
        robot_sx, robot_sy = self.world_to_screen(self.robot_x, self.robot_y)
        pygame.draw.line(self.screen, path_color, 
                        (robot_sx, robot_sy), screen_points[0], 3)
        
        # Draw the path as a line
        pygame.draw.lines(self.screen, path_color, False, screen_points, 3)
        
        # Draw waypoint markers at intervals
        num_points = len(screen_points)
        if num_points > 3:
            step = max(1, num_points // 6)
            for i in range(0, num_points, step):
                pygame.draw.circle(self.screen, COLOR_PLANNED_PATH_WAYPOINT,
                                 screen_points[i], 5, 0)
        
        # Draw endpoint marker
        if screen_points:
            end_x, end_y = screen_points[-1]
            pygame.draw.circle(self.screen, (255, 255, 0), (end_x, end_y), 8, 2)
            pygame.draw.circle(self.screen, (255, 255, 0), (end_x, end_y), 4, 0)
    
    def draw_planning_stats_panel(self):
        """Draw planning and quality statistics panel"""
        if not self.show_planned_path:
            return
        
        # Panel position (bottom left, above embedding panel)
        panel_x = 10
        panel_y = self.screen_height - 280
        panel_w = 200
        panel_h = 45
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, (80, 80, 80), (panel_x, panel_y, panel_w, panel_h), 1)
        
        # Quality bar
        quality_text = self.font_small.render(f"Nav Quality: {self.quality_score:.2f}", True, (200, 200, 200))
        self.screen.blit(quality_text, (panel_x + 5, panel_y + 5))
        
        # Quality bar
        bar_x = panel_x + 5
        bar_y = panel_y + 22
        bar_w = 190
        bar_h = 15
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        
        # Fill based on quality
        fill_w = int(bar_w * min(1.0, self.quality_score))
        if self.quality_score > 0.7:
            fill_color = COLOR_QUALITY_GOOD
        elif self.quality_score > 0.4:
            fill_color = COLOR_QUALITY_MEDIUM
        else:
            fill_color = COLOR_QUALITY_LOW
        
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_w, bar_h), 1)
    
    def draw_info_panel(self):
        """Draw information panel"""
        panel_x = 10
        panel_y = 10
        line_height = 20
        
        # Build info lines
        info_lines = [
            f"Robot: ({self.robot_x:.2f}, {self.robot_y:.2f})",
            f"Heading: {math.degrees(self.robot_theta):.1f}°"
        ]
        
        if self.has_goal:
            dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 +
                                    (self.goal_y - self.robot_y)**2)
            info_lines.append(f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
            info_lines.append(f"Distance: {dist_to_goal:.2f}m")
        else:
            info_lines.append("Goal: Not Set")
            info_lines.append("Distance: N/A")
        
        info_lines.extend([
            f"Linear Vel: {self.cmd_vel_linear:.2f} m/s",
            f"Angular Vel: {self.cmd_vel_angular:.2f} rad/s",
            f"Humans: {len(self.detected_persons)}",
            f"FPS: {self.fps:.1f}"
        ])
        
        # Draw background
        panel_width = 250
        panel_height = len(info_lines) * line_height + 20
        pygame.draw.rect(self.screen, (40, 40, 40),
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(text_surface, (panel_x + 10, panel_y + 10 + i * line_height))
    
    def draw_attention_legend(self):
        """Draw attention color legend"""
        legend_x = self.screen_width - 240
        legend_y = 20
        line_height = 25
        
        # Background
        legend_height = 30 + len(ATTENTION_COLORS) * line_height + 10
        if not self.gat_enabled:
            legend_height += 40  # Extra space for warning
        
        pygame.draw.rect(self.screen, (40, 40, 40),
                        (legend_x, legend_y, 220, legend_height))
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (legend_x, legend_y, 220, legend_height), 2)
        
        # Title
        if self.gat_enabled:
            title_surface = self.font_large.render("Attention Categories:", True, (255, 255, 255))
        else:
            title_surface = self.font_large.render("Attention (DISABLED):", True, (255, 100, 100))
        self.screen.blit(title_surface, (legend_x + 10, legend_y + 8))
        
        # Warning if GAT not enabled
        current_y = legend_y + 35
        if not self.gat_enabled:
            warn1 = self.font_small.render("Run with --model-dir to", True, (255, 200, 100))
            warn2 = self.font_small.render("enable attention colors", True, (255, 200, 100))
            self.screen.blit(warn1, (legend_x + 10, current_y))
            self.screen.blit(warn2, (legend_x + 10, current_y + 15))
            current_y += 40
        
        # Entries
        legend_entries = [
            ("LiDAR-to-LiDAR", ATTENTION_COLORS['lidar_to_lidar']),
            ("Robot-to-LiDAR", ATTENTION_COLORS['robot_to_lidar']),
            ("Robot-to-Human", ATTENTION_COLORS['robot_to_human']),
            ("Human-to-LiDAR", ATTENTION_COLORS['human_to_lidar']),
            ("Human-to-Human", ATTENTION_COLORS['human_to_human'])
        ]
        
        for label, color in legend_entries:
            # Line sample (dimmed if GAT disabled)
            draw_color = color if self.gat_enabled else (80, 80, 80)
            pygame.draw.line(self.screen, draw_color,
                           (legend_x + 10, current_y + 8),
                           (legend_x + 50, current_y + 8), 3)
            
            # Label
            text_color = (200, 200, 200) if self.gat_enabled else (120, 120, 120)
            text_surface = self.font.render(label, True, text_color)
            self.screen.blit(text_surface, (legend_x + 60, current_y))
            current_y += line_height
    
    def draw_embedding_panel(self):
        """Draw the current navigation mode using the t-SNE labels."""
        if not self.show_features:
            return

        lidar_min = None
        forward_clearance = 0.0
        if self.lidar_ranges is not None:
            lidar = self.downsample_lidar(self.lidar_ranges)
            if len(lidar):
                lidar_min = float(np.min(lidar))
                front = np.asarray(lidar)[8:13]
                forward_clearance = float(np.mean(front)) if len(front) else lidar_min

        if self.has_goal:
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            goal_dist = math.hypot(dx, dy)
            goal_angle = math.atan2(dy, dx) - self.robot_theta
            while goal_angle > math.pi:
                goal_angle -= 2 * math.pi
            while goal_angle < -math.pi:
                goal_angle += 2 * math.pi
        else:
            goal_dist = 10.0
            goal_angle = math.pi

        closest_human = float('inf')
        for person in self.detected_persons:
            d = math.hypot(person['x_odom'] - self.robot_x, person['y_odom'] - self.robot_y)
            if d < closest_human:
                closest_human = d

        label, color = self.embedding_interpreter.classify_state(
            lidar_min, forward_clearance, goal_dist, goal_angle,
            abs(self.cmd_vel_linear), abs(self.cmd_vel_angular), closest_human
        )
        self.embedding_interpreter.draw(
            self.screen, label, color,
            10, self.screen_height - 48,
            self.font_large
        )
    
    def render(self):
        """Main render loop"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:  # Toggle axes with 'A' key
                    self.show_axes = not self.show_axes
                    rospy.loginfo(f"Coordinate axes: {'ON' if self.show_axes else 'OFF'}")
                elif event.key == pygame.K_e:  # Toggle edges with 'E' key
                    self.show_edges = not self.show_edges
                    rospy.loginfo(f"Graph edges: {'ON' if self.show_edges else 'OFF'}")
                elif event.key == pygame.K_f:  # Toggle features panel with 'F' key
                    self.show_features = not self.show_features
                    rospy.loginfo(f"Features panel: {'ON' if self.show_features else 'OFF'}")
                elif event.key == pygame.K_p:  # Toggle planned path with 'P' key
                    self.show_planned_path = not self.show_planned_path
                    rospy.loginfo(f"Planned path: {'ON' if self.show_planned_path else 'OFF'}")
        
        # Run GAT inference if enabled
        if self.gat_enabled:
            self.run_gat_inference()
        
        # Clear screen
        self.screen.fill(COLOR_BACKGROUND)
        
        # Draw elements (order matters for layering)
        self.draw_lidar_outline()
        self.draw_graph_edges()  # Draw edges under nodes (now attention-weighted)
        self.draw_lidar_beams_and_nodes()
        self.draw_planned_path()  # Draw planned path (behind goal and robot)
        self.draw_goal()
        self.draw_humans()
        self.draw_robot()
        
        # Draw UI
        self.draw_info_panel()
        self.draw_attention_legend()
        self.draw_embedding_panel()  # Embedding features and semantic state
        # Planning stats panel removed (continuous planning disabled by default)
        self.draw_coordinate_axes()  # Draw axes in corner (on top of everything)
        self.draw_cursor_position()  # Show world coordinates at cursor
        
        # Update display
        pygame.display.flip()
        
        # Update FPS
        current_time = time.time()
        if current_time - self.last_render_time > 0:
            self.fps = 1.0 / (current_time - self.last_render_time)
        self.last_render_time = current_time
        
        return True
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Starting Digital Twin visualization")
        rospy.loginfo("Keyboard controls:")
        rospy.loginfo("  'A' - Toggle coordinate axes")
        rospy.loginfo("  'E' - Toggle graph edges")
        rospy.loginfo("  'F' - Toggle embedding features panel")
        rospy.loginfo("  'P' - Toggle planned path visualization")
        
        rate = rospy.Rate(CONTROL_RATE)
        while not rospy.is_shutdown():
            if not self.render():
                break
            self.clock.tick(CONTROL_RATE)
            rate.sleep()
        
        pygame.quit()
        rospy.loginfo("Digital Twin visualizer stopped")


# ==================== MAIN ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Digital Twin Visualization for Husky')
    parser.add_argument('--goal-x', type=float, default=None, help='Initial goal X coordinate (optional)')
    parser.add_argument('--goal-y', type=float, default=None, help='Initial goal Y coordinate (optional)')
    parser.add_argument('--ppm', type=int, default=50, help='Pixels per meter (default: 50)')
    parser.add_argument('--no-axes', action='store_true', help='Disable coordinate axes at startup')
    parser.add_argument('--no-edges', action='store_true', help='Disable graph edges at startup')
    parser.add_argument('--model-dir', type=str, default=None, 
                       help='Directory containing GAT model files (enables attention visualization)')
    parser.add_argument('--model-name', type=str, default='TD3_gat_humans',
                       help='Model name prefix (default: TD3_gat_humans)')
    args = parser.parse_args()
    
    try:
        visualizer = DigitalTwinVisualizer(
            args.goal_x, args.goal_y, args.ppm,
            show_axes=not args.no_axes,
            show_edges=not args.no_edges,
            model_dir=args.model_dir,
            model_name=args.model_name
        )
        visualizer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Visualizer interrupted")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
