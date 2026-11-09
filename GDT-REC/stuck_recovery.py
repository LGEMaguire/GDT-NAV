#!/usr/bin/env python3

"""
Stuck Recovery System for GAT-TD3 Husky Navigation

Features:
1. Stuck detection and recovery training
2. Continuous lookahead planning (runs ahead of robot)
3. Quality-gated continuous light updates
4. Path visualization for explainability

When the robot gets stuck (edge case or blocked path):
1. Pause navigation and detect stuck state
2. Take a snapshot of the environment from LiDAR data
3. Create a simulated environment from the snapshot
4. Clone the model and quick-train to find a recovery path
5. Yaw in place to the logged heading from when the twin set off
   toward the first waypoint, blend logged actions with the live policy
   (so moving humans still change the command) until that waypoint,
   then hand back to the original policy
6. Apply a light update to the original model

Continuous Planning Mode:
- Constantly simulates ahead to predict path
- Provides trajectory visualization
- Quality-gated micro-updates during successful navigation

Author: DRLtrain
"""

import os
import math
import copy
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from graph_utils import create_edge_index
from lidar_front import BEAM_ANGLES

# Constants
STUCK_VELOCITY_THRESHOLD = 0.05  # m/s - robot considered not moving if below this
STUCK_PROGRESS_THRESHOLD = 0.1  # meters - minimum progress expected in STUCK_CHECK_WINDOW
STUCK_CHECK_WINDOW = 5.0  # seconds - window to check for progress
STUCK_DETECTION_COUNT = 3  # Number of consecutive stuck detections before triggering
HUSKY_WIDTH = 1.5  # meters - minimum gap for Husky to fit through
WAYPOINT_REACH_DIST = 0.4  # meters — short chain steps must actually be reached
LOGGED_ACTION_BLEND = 0.5  # weight on the logged action; the rest is live DRL
HEADING_MATCH_RAD = 0.26  # ~15 degrees, yaw-only align after the waypoint
BLEND_STEP_FACTOR = 4  # give up on the blend after this many times the log length
ALIGN_TIMEOUT_S = 15.0  # yaw align gives up and hands back to full DRL
LIDAR_RANGE = 10.0
LIDAR_BEAMS = 20
MAX_HUMANS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== NEURAL NETWORK MODELS (same as training) ====================
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
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))


class Critic(nn.Module):
    """Twin Q-network critic"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.c1 = nn.Linear(state_dim + action_dim, 256)
        self.c2 = nn.Linear(256, 128)
        self.c3 = nn.Linear(128, 1)
        self.c4 = nn.Linear(state_dim + action_dim, 256)
        self.c5 = nn.Linear(256, 128)
        self.c6 = nn.Linear(128, 1)
    
    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.Q1(sa), self.Q2(sa)
    
    def Q1(self, sa):
        x = F.relu(self.c1(sa))
        x = F.relu(self.c2(x))
        return self.c3(x)
    
    def Q2(self, sa):
        x = F.relu(self.c4(sa))
        x = F.relu(self.c5(x))
        return self.c6(x)


# ==================== DATA STRUCTURES ====================
@dataclass
class EnvironmentSnapshot:
    """Captured environment state when stuck is detected"""
    timestamp: float
    robot_x: float
    robot_y: float
    robot_theta: float
    goal_x: float
    goal_y: float
    
    # LiDAR data
    lidar_ranges: np.ndarray  # Raw ranges
    lidar_points_world: np.ndarray  # (N, 2) world coordinates
    
    # Computed obstacles
    boundary_polygon: np.ndarray  # (M, 2) boundary vertices
    obstacle_segments: List[Tuple[np.ndarray, np.ndarray]]  # Line segments
    blocked_regions: List[Tuple[float, float, float]]  # (x, y, radius) for gaps < 1.5m
    
    # Detected humans (if any)
    humans: List[Dict]


class StuckDetector:
    """Monitors robot state and detects stuck conditions"""
    
    def __init__(self, 
                 velocity_threshold: float = STUCK_VELOCITY_THRESHOLD,
                 progress_threshold: float = STUCK_PROGRESS_THRESHOLD,
                 check_window: float = STUCK_CHECK_WINDOW,
                 detection_count: int = STUCK_DETECTION_COUNT):
        
        self.velocity_threshold = velocity_threshold
        self.progress_threshold = progress_threshold
        self.check_window = check_window
        self.detection_count = detection_count
        
        # History tracking
        self.position_history = deque(maxlen=100)  # (timestamp, x, y, theta)
        self.velocity_history = deque(maxlen=50)   # (timestamp, vx, wz)
        self.goal_distance_history = deque(maxlen=100)  # (timestamp, distance)
        
        self.consecutive_stuck_count = 0
        self.last_check_time = 0.0
        self.is_stuck = False
        
    def update(self, robot_x: float, robot_y: float, robot_theta: float,
               vx: float, wz: float, goal_x: float, goal_y: float) -> bool:
        """Update detector with current state, returns True if stuck detected"""
        current_time = time.time()
        
        # Record position
        self.position_history.append((current_time, robot_x, robot_y, robot_theta))
        self.velocity_history.append((current_time, vx, wz))
        
        # Calculate distance to goal
        goal_dist = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        self.goal_distance_history.append((current_time, goal_dist))
        
        # Check stuck conditions every 0.5 seconds
        if current_time - self.last_check_time < 0.5:
            return self.is_stuck
        
        self.last_check_time = current_time
        
        # Need enough history
        if len(self.position_history) < 10:
            return False
        
        # Check 1: Low velocity for extended period
        recent_velocities = [(t, v, w) for t, v, w in self.velocity_history 
                            if current_time - t < self.check_window]
        if len(recent_velocities) > 5:
            avg_v = np.mean([abs(v) for _, v, _ in recent_velocities])
            low_velocity = avg_v < self.velocity_threshold
        else:
            low_velocity = False
        
        # Check 2: No progress toward goal
        old_distances = [d for t, d in self.goal_distance_history 
                        if current_time - t > self.check_window * 0.8 
                        and current_time - t < self.check_window]
        recent_distances = [d for t, d in self.goal_distance_history 
                          if current_time - t < 1.0]
        
        if old_distances and recent_distances:
            progress = np.mean(old_distances) - np.mean(recent_distances)
            no_progress = progress < self.progress_threshold
        else:
            no_progress = False
        
        # Check 3: Oscillating behavior (turning back and forth)
        recent_positions = [(x, y) for t, x, y, _ in self.position_history 
                           if current_time - t < self.check_window]
        if len(recent_positions) > 10:
            pos_array = np.array(recent_positions)
            movement_range = np.max(pos_array, axis=0) - np.min(pos_array, axis=0)
            oscillating = np.all(movement_range < 0.5) and len(recent_positions) > 20
        else:
            oscillating = False
        
        # Stuck if any two conditions are met
        stuck_indicators = sum([low_velocity, no_progress, oscillating])
        
        if stuck_indicators >= 2:
            self.consecutive_stuck_count += 1
        else:
            self.consecutive_stuck_count = max(0, self.consecutive_stuck_count - 1)
        
        self.is_stuck = self.consecutive_stuck_count >= self.detection_count
        
        return self.is_stuck
    
    def reset(self):
        """Reset detector state"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.goal_distance_history.clear()
        self.consecutive_stuck_count = 0
        self.is_stuck = False


# ==================== ENVIRONMENT SNAPSHOT ====================
class EnvironmentSnapshotBuilder:
    """Creates a snapshot of the environment from LiDAR data"""
    
    def __init__(self, husky_width: float = HUSKY_WIDTH):
        self.husky_width = husky_width
        
    def capture(self, 
                robot_x: float, robot_y: float, robot_theta: float,
                goal_x: float, goal_y: float,
                lidar_ranges: np.ndarray,
                raw_scan_ranges: Optional[np.ndarray] = None,
                detected_persons: Optional[List[Dict]] = None) -> EnvironmentSnapshot:
        """
        Capture current environment as a snapshot
        
        Args:
            robot_x, robot_y, robot_theta: Robot pose in world frame
            goal_x, goal_y: Goal position
            lidar_ranges: Downsampled LiDAR (20 beams)
            raw_scan_ranges: Full 360° scan (optional, for better boundary)
            detected_persons: List of detected humans
        """
        # Convert LiDAR to world coordinates
        lidar_points = self._lidar_to_world(
            robot_x, robot_y, robot_theta, 
            lidar_ranges if raw_scan_ranges is None else raw_scan_ranges,
            full_scan=(raw_scan_ranges is not None)
        )
        
        # Create boundary from LiDAR points (connect the dots)
        boundary = self._create_boundary(lidar_points)
        
        # Find blocked regions (gaps < husky_width between LiDAR nodes)
        blocked = self._find_blocked_regions(
            robot_x, robot_y, robot_theta, lidar_ranges
        )
        
        # Create obstacle segments from boundary
        segments = self._boundary_to_segments(boundary)
        
        return EnvironmentSnapshot(
            timestamp=time.time(),
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
            goal_x=goal_x,
            goal_y=goal_y,
            lidar_ranges=lidar_ranges.copy(),
            lidar_points_world=lidar_points,
            boundary_polygon=boundary,
            obstacle_segments=segments,
            blocked_regions=blocked,
            humans=detected_persons or []
        )
    
    def _lidar_to_world(self, rx: float, ry: float, rth: float,
                        ranges: np.ndarray, full_scan: bool = False) -> np.ndarray:
        """Convert LiDAR ranges to world coordinates"""
        points = []
        n_beams = len(ranges)
        
        if full_scan:
            # Full 360° scan
            angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
        elif n_beams == len(BEAM_ANGLES):
            # Same 20 rays as sim / deploy: -90° … +81°, step π/20
            angles = BEAM_ANGLES
        else:
            angles = -math.pi / 2.0 + np.arange(n_beams) * (math.pi / n_beams)
        
        for i, (r, angle) in enumerate(zip(ranges, angles)):
            if np.isnan(r) or np.isinf(r) or r < 0.1 or r > LIDAR_RANGE - 0.1:
                continue
            
            # Transform to world
            beam_angle = angle + rth
            wx = rx + r * math.cos(beam_angle)
            wy = ry + r * math.sin(beam_angle)
            points.append([wx, wy])
        
        return np.array(points) if points else np.zeros((0, 2))
    
    def _create_boundary(self, points: np.ndarray) -> np.ndarray:
        """Create boundary polygon from LiDAR points (connect dots)"""
        if len(points) < 3:
            return np.zeros((0, 2))
        
        try:
            # Use convex hull as outer boundary approximation
            hull = ConvexHull(points)
            boundary = points[hull.vertices]
            return boundary
        except Exception:
            # Fallback: return sorted points by angle from centroid
            centroid = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1], 
                               points[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            return points[sorted_indices]
    
    def _find_blocked_regions(self, rx: float, ry: float, rth: float,
                              lidar_ranges: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Find regions where Husky cannot fit (gaps < husky_width between adjacent nodes)
        Returns list of (x, y, radius) blocking obstacles
        """
        blocked = []
        n_beams = len(lidar_ranges)
        if n_beams == len(BEAM_ANGLES):
            angles = BEAM_ANGLES
        else:
            angles = -math.pi / 2.0 + np.arange(n_beams) * (math.pi / n_beams)
        
        for i in range(n_beams - 1):
            r1, r2 = lidar_ranges[i], lidar_ranges[i + 1]
            
            # Skip invalid readings
            if r1 > LIDAR_RANGE - 0.1 or r2 > LIDAR_RANGE - 0.1:
                continue
            if np.isnan(r1) or np.isnan(r2):
                continue
            
            # Calculate world positions
            a1, a2 = angles[i] + rth, angles[i + 1] + rth
            x1 = rx + r1 * math.cos(a1)
            y1 = ry + r1 * math.sin(a1)
            x2 = rx + r2 * math.cos(a2)
            y2 = ry + r2 * math.sin(a2)
            
            # Calculate gap
            gap = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # If gap is less than Husky width, mark as blocked
            if gap < self.husky_width:
                # Place a virtual obstacle at the midpoint
                # Use smaller radius to avoid trapping robot inside obstacle
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                radius = max(0.15, (self.husky_width - gap) / 2 * 0.6)  # Reduced from 0.3
                blocked.append((mid_x, mid_y, radius))
        
        return blocked
    
    def _boundary_to_segments(self, boundary: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert boundary polygon to line segments"""
        segments = []
        n = len(boundary)
        if n < 2:
            return segments
        
        for i in range(n):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % n]
            segments.append((p1, p2))
        
        return segments


# ==================== SIMULATED RECOVERY ENVIRONMENT ====================
class RecoverySimEnv:
    """
    Simulation environment created from a snapshot for recovery training
    
    Matches the training approach from pygame_env_rosv5.py but with:
    - Shorter episodes
    - Grace period for close obstacles (robot starts stuck near them)
    - Identical reward structure for consistency
    """
    
    def __init__(self, snapshot: EnvironmentSnapshot, 
                 lidar_nodes: int = LIDAR_BEAMS,
                 max_humans: int = MAX_HUMANS,
                 grace_period_steps: int = 10,
                 collision_grace_dist: float = 0.6,
                 use_waypoints: bool = True):
        self.snapshot = snapshot
        self.L = lidar_nodes
        self.H = max_humans
        
        # Robot state (start from snapshot position)
        self.robot_x = snapshot.robot_x
        self.robot_y = snapshot.robot_y
        self.robot_theta = snapshot.robot_theta
        self.v = 0.0
        self.w = 0.0
        
        # Final goal
        self.final_goal_x = snapshot.goal_x
        self.final_goal_y = snapshot.goal_y
        
        # Convert blocked regions to circle obstacles
        self.obstacles = []
        for bx, by, br in snapshot.blocked_regions:
            self.obstacles.append({'x': bx, 'y': by, 'r': br})
        
        # === SIMULATED HUMANS WITH MOVEMENT ===
        # Initialize humans from snapshot with their predicted trajectories
        self.sim_humans = []
        for human in snapshot.humans:
            self.sim_humans.append({
                'x': human.get('x_odom', 0.0),
                'y': human.get('y_odom', 0.0),
                'vx': human.get('vx', 0.0),  # Velocity from tracking
                'vy': human.get('vy', 0.0),
                'start_x': human.get('x_odom', 0.0),  # Initial position
                'start_y': human.get('y_odom', 0.0),
                'confidence': human.get('confidence', 0.9),
                'radius': 0.4  # Human collision radius
            })
        
        # Add boundary segments as wall constraints (MUST be before waypoint generation)
        self.boundary_segments = snapshot.obstacle_segments
        self.boundary_points = snapshot.lidar_points_world
        
        # === WAYPOINT SYSTEM ===
        # Create intermediate waypoints to guide robot through gaps
        self.use_waypoints = use_waypoints
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_reach_dist = WAYPOINT_REACH_DIST
        
        if use_waypoints:
            self._generate_waypoints()
        
        # Current goal (first waypoint or final goal)
        if self.waypoints:
            self.goal_x = self.waypoints[0][0]
            self.goal_y = self.waypoints[0][1]
        else:
            self.goal_x = self.final_goal_x
            self.goal_y = self.final_goal_y
        
        # Constants
        self.dt = 0.1  # 10 Hz (TIME_DELTA)
        self.v_max = 1.0
        self.w_max = 2.0
        self.collision_dist = 0.5
        self.goal_dist = 0.75
        
        # === GRACE PERIOD FOR CLOSE OBSTACLES ===
        # Robot starts stuck, so ignore obstacles it's already near
        self.grace_period_steps = grace_period_steps
        self.collision_grace_dist = collision_grace_dist
        self._initial_close_obstacles = []  # List of (x, y) that were initially close
        self._initial_close_boundary = []   # List of (x, y) boundary points initially close
        self._identify_initial_close_obstacles()
        
        # === REWARD TRACKING (matches pygame_env_rosv5.py) ===
        self.accumulated_turn = 0.0
        self.turn_direction = 0
        
        # State
        self.step_count = 0
        self.max_steps = 200  # Shorter episodes for recovery
        self.prev_dist = self._dist_to_goal()
        self.done = False
        
        # Last LiDAR for reward calculation
        self._last_lidar = None
        
    def _identify_initial_close_obstacles(self):
        """Identify obstacles that are close at the start (grace period)"""
        self._initial_close_obstacles = []
        self._initial_close_boundary = []
        
        rx, ry = self.snapshot.robot_x, self.snapshot.robot_y
        
        # Check circle obstacles
        for obs in self.obstacles:
            d = math.sqrt((rx - obs['x'])**2 + (ry - obs['y'])**2)
            if d < obs['r'] + self.collision_grace_dist:
                self._initial_close_obstacles.append((obs['x'], obs['y']))
        
        # Check boundary points
        for px, py in self.boundary_points:
            d = math.sqrt((rx - px)**2 + (ry - py)**2)
            if d < self.collision_grace_dist:
                self._initial_close_boundary.append((px, py))
    
    def _generate_waypoints(self):
        """
        Short steps from the robot to the goal, starting in the open
        sector to the robot's left when that is where the scan is empty.

        Only real LiDAR returns count. The synthetic blocked-region
        circles fill visual gaps between beams and were rejecting the
        opening that is obvious on the digital twin.
        """
        self.waypoints = []
        x, y = self.snapshot.robot_x, self.snapshot.robot_y
        th = self.snapshot.robot_theta
        gx, gy = self.final_goal_x, self.final_goal_y
        step = 0.6
        min_clearance = 0.25
        min_path_gap = 0.15

        def usable(nx, ny, ox, oy):
            return (self._nearest_lidar_dist(nx, ny) >= min_clearance
                    and self._lidar_segment_gap(ox, oy, nx, ny) >= min_path_gap)

        # First point: the clearest short step on the robot's left.
        first = None
        for deg in (30, 45, 60, 75, 90, 15, 105, 120):
            ang = th + math.radians(deg)
            for dist in (0.8, 1.1, 0.6):
                nx = x + dist * math.cos(ang)
                ny = y + dist * math.sin(ang)
                if not usable(nx, ny, x, y):
                    continue
                gap = self._lidar_segment_gap(x, y, nx, ny)
                cand = (gap, dist, nx, ny)
                if first is None or cand[0] > first[0]:
                    first = cand
                break
        if first is not None:
            x, y = first[2], first[3]
            self.waypoints.append((x, y))
            print(f"[RecoverySimEnv] Left opening at ({x:.2f}, {y:.2f})")

        for _ in range(8):
            dist = math.hypot(gx - x, gy - y)
            if dist < 0.75:
                break
            ux, uy = (gx - x) / dist, (gy - y) / dist
            span = min(step, max(0.35, dist * 0.45))
            best = None
            for ang_deg in (0, 25, -25, 50, -50, 75, -75, 100, -100):
                ang = math.radians(ang_deg)
                ca, sa = math.cos(ang), math.sin(ang)
                dx = ux * ca - uy * sa
                dy = ux * sa + uy * ca
                nx, ny = x + dx * span, y + dy * span
                if not usable(nx, ny, x, y):
                    continue
                goal_dist = math.hypot(gx - nx, gy - ny)
                cand = (goal_dist, abs(ang_deg), nx, ny)
                if best is None or cand < best:
                    best = cand
            # Allow a small detour; only stop when every option goes backwards.
            if best is None or best[0] > dist + 0.15:
                break
            x, y = best[2], best[3]
            self.waypoints.append((x, y))

        self.waypoints.append((gx, gy))
        print(f"[RecoverySimEnv] Waypoint chain: {len(self.waypoints)} points")
        for i, (wx, wy) in enumerate(self.waypoints):
            label = "GOAL" if i == len(self.waypoints) - 1 else f"WP{i}"
            print(f"  {label}: ({wx:.2f}, {wy:.2f})")

    def _nearest_lidar_dist(self, x: float, y: float) -> float:
        """Distance to the nearest LiDAR return. Blocked-region circles are ignored."""
        nearest = float('inf')
        for px, py in self.boundary_points:
            nearest = min(nearest, math.hypot(px - x, py - y))
        return nearest

    def _lidar_segment_gap(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """Smallest distance from a LiDAR return to the open segment."""
        dx = x1 - x0
        dy = y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-8:
            return 0.0
        gap = float('inf')
        for px, py in self.boundary_points:
            t = ((px - x0) * dx + (py - y0) * dy) / seg2
            if t < 0.08 or t > 0.92:
                continue
            dist = math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
            if dist < gap:
                gap = dist
        return gap

    def _nearest_obstacle_dist(self, x: float, y: float) -> float:
        """Distance to the nearest LiDAR return or blocked node."""
        nearest = float('inf')
        for px, py in self.boundary_points:
            nearest = min(nearest, math.hypot(px - x, py - y))
        for obs in self.obstacles:
            nearest = min(nearest, math.hypot(obs['x'] - x, obs['y'] - y) - obs['r'])
        return nearest

    def _segment_gap(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """Smallest distance from a return or node to the open segment."""
        dx = x1 - x0
        dy = y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-8:
            return 0.0
        gap = float('inf')
        def consider(px, py, radius):
            nonlocal gap
            t = ((px - x0) * dx + (py - y0) * dy) / seg2
            if t < 0.08 or t > 0.92:
                return
            dist = math.hypot(px - (x0 + t * dx), py - (y0 + t * dy)) - radius
            if dist < gap:
                gap = dist
        for px, py in self.boundary_points:
            consider(px, py, 0.0)
        for obs in self.obstacles:
            consider(obs['x'], obs['y'], obs['r'])
        return gap
    
    def _find_clear_waypoint(self, base_x: float, base_y: float, 
                             dir_x: float, dir_y: float,
                             min_clearance: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Find a clear waypoint near the base position.

        The point itself must be clear of obstacles, and the straight
        segment from the robot to that point must not pass through a
        LiDAR return or a blocked-region node.
        """
        rx, ry = self.snapshot.robot_x, self.snapshot.robot_y

        def acceptable(x, y):
            return (self._is_position_clear(x, y, min_clearance=min_clearance)
                    and self._segment_is_clear(rx, ry, x, y))

        if acceptable(base_x, base_y):
            return base_x, base_y
        
        # Search perpendicular to direction (left and right)
        perp_x = -dir_y
        perp_y = dir_x
        
        for offset in [0.5, 1.0, 1.5, 2.0, 2.5]:
            # Try left
            test_x = base_x + perp_x * offset
            test_y = base_y + perp_y * offset
            if acceptable(test_x, test_y):
                return test_x, test_y
            
            # Try right
            test_x = base_x - perp_x * offset
            test_y = base_y - perp_y * offset
            if acceptable(test_x, test_y):
                return test_x, test_y
        
        # Couldn't find clear position
        return None, None

    def _segment_is_clear(self, x0: float, y0: float, x1: float, y1: float,
                          margin: float = 0.3) -> bool:
        """True when no LiDAR point or blocked node lies on the open segment."""
        dx = x1 - x0
        dy = y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-8:
            return False

        def on_segment(px, py, radius):
            t = ((px - x0) * dx + (py - y0) * dy) / seg2
            # Ignore the robot's own neighbourhood and the waypoint endpoint
            if t < 0.12 or t > 0.92:
                return False
            qx = x0 + t * dx
            qy = y0 + t * dy
            return math.hypot(px - qx, py - qy) < radius

        for px, py in self.boundary_points:
            if on_segment(px, py, margin):
                return False
        for obs in self.obstacles:
            if on_segment(obs['x'], obs['y'], obs['r'] + 0.15):
                return False
        return True
    
    def _is_position_clear(self, x: float, y: float, min_clearance: float = 0.8) -> bool:
        """Check if a position has enough clearance from obstacles"""
        # Check circle obstacles
        for obs in self.obstacles:
            d = math.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            if d < obs['r'] + min_clearance:
                return False
        
        # Check boundary points
        for px, py in self.boundary_points:
            d = math.sqrt((x - px)**2 + (y - py)**2)
            if d < min_clearance:
                return False
        
        return True
    
    def _check_waypoint_progress(self):
        """Check if current waypoint is reached and advance to next"""
        if not self.waypoints:
            return
        
        # Check bounds - if we've passed all waypoints, nothing to do
        if self.current_waypoint_idx >= len(self.waypoints):
            return
        
        # Distance to current waypoint
        wx, wy = self.waypoints[self.current_waypoint_idx]
        dist = math.sqrt((self.robot_x - wx)**2 + (self.robot_y - wy)**2)
        
        if dist < self.waypoint_reach_dist:
            # Waypoint reached!
            old_idx = self.current_waypoint_idx
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.waypoints):
                # Advance to next waypoint
                self.goal_x = self.waypoints[self.current_waypoint_idx][0]
                self.goal_y = self.waypoints[self.current_waypoint_idx][1]
                print(f"[RecoverySimEnv] Waypoint {old_idx} reached! Now targeting WP{self.current_waypoint_idx}")
            else:
                # All waypoints reached, target final goal
                self.goal_x = self.final_goal_x
                self.goal_y = self.final_goal_y
                print(f"[RecoverySimEnv] All waypoints reached! Now targeting FINAL GOAL")
        
    def reset(self) -> np.ndarray:
        """Reset to snapshot position"""
        self.robot_x = self.snapshot.robot_x
        self.robot_y = self.snapshot.robot_y
        self.robot_theta = self.snapshot.robot_theta
        self.v = 0.0
        self.w = 0.0
        self.step_count = 0
        self.done = False
        
        # Reset turn tracking
        self.accumulated_turn = 0.0
        self.turn_direction = 0
        
        # Reset waypoint tracking
        self.current_waypoint_idx = 0
        if self.waypoints:
            self.goal_x = self.waypoints[0][0]
            self.goal_y = self.waypoints[0][1]
        else:
            self.goal_x = self.final_goal_x
            self.goal_y = self.final_goal_y
        
        self.prev_dist = self._dist_to_goal()
        
        # Reset humans to starting positions
        for human in self.sim_humans:
            human['x'] = human['start_x']
            human['y'] = human['start_y']
        
        # Re-identify initial close obstacles
        self._identify_initial_close_obstacles()
        
        state = self._get_state()
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, bool]:
        """
        Execute action
        action[0]: linear velocity command (0 to 1)
        action[1]: angular velocity command (-1 to 1)
        """
        self.step_count += 1
        
        # Apply action with first-order lag (matches training)
        v_cmd = float(action[0]) * self.v_max
        w_cmd = float(action[1]) * self.w_max
        
        tau = 0.1
        alpha = self.dt / (tau + self.dt)
        self.v = self.v + alpha * (v_cmd - self.v)
        self.w = self.w + alpha * (w_cmd - self.w)
        
        # Integrate kinematics
        self.robot_theta += self.w * self.dt
        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        self.robot_x += self.v * math.cos(self.robot_theta) * self.dt
        self.robot_y += self.v * math.sin(self.robot_theta) * self.dt
        
        # === MOVE HUMANS ALONG PREDICTED TRAJECTORIES ===
        self._move_humans()
        
        # === CHECK WAYPOINT PROGRESS ===
        # Advance to next waypoint if current one reached
        if self.use_waypoints:
            self._check_waypoint_progress()
        
        # Get LiDAR for reward calculation
        lidar = self._simulate_lidar()
        self._last_lidar = lidar
        
        # Check collision (with grace period)
        collision = self._check_collision()
        
        # Check goal - use FINAL goal for success check, not intermediate waypoints
        dist_to_final = math.sqrt((self.final_goal_x - self.robot_x)**2 + 
                                  (self.final_goal_y - self.robot_y)**2)
        target = dist_to_final < self.goal_dist
        
        # Distance to current waypoint/goal (for reward)
        dist = self._dist_to_goal()
        
        # Timeout
        timeout = self.step_count >= self.max_steps
        
        # Reward (matches pygame_env_rosv5.py)
        # Give bonus for reaching waypoints
        waypoint_bonus = 0.0
        if self.use_waypoints and self.current_waypoint_idx > 0:
            # Robot has made progress through waypoints
            waypoint_bonus = 2.0 * self.current_waypoint_idx  # Bonus per waypoint reached
        
        reward = self._compute_reward(target, collision, timeout, dist, lidar)
        reward += waypoint_bonus
        self.prev_dist = dist
        
        self.done = target or collision or timeout
        
        state = self._build_node_features(lidar)
        return state, reward, self.done, target, collision
    
    def _dist_to_goal(self) -> float:
        return math.sqrt((self.goal_x - self.robot_x)**2 + 
                        (self.goal_y - self.robot_y)**2)
    
    def _move_humans(self):
        """Move simulated humans along their predicted trajectories"""
        for human in self.sim_humans:
            # Move at predicted velocity
            human['x'] += human['vx'] * self.dt
            human['y'] += human['vy'] * self.dt
            
            # Keep humans within reasonable bounds (don't let them wander too far)
            # Limit to 5m from their starting position
            dx = human['x'] - human['start_x']
            dy = human['y'] - human['start_y']
            dist_from_start = math.sqrt(dx*dx + dy*dy)
            if dist_from_start > 5.0:
                # Reverse direction
                human['vx'] = -human['vx']
                human['vy'] = -human['vy']
    
    def _check_human_collision(self) -> bool:
        """Check collision with simulated humans"""
        for human in self.sim_humans:
            d = math.sqrt((self.robot_x - human['x'])**2 + 
                         (self.robot_y - human['y'])**2)
            if d < self.collision_dist + human['radius']:
                return True
        return False
    
    def _is_in_grace_period(self) -> bool:
        """
        Check if still in grace period for initial close obstacles
        Grace period is active until FIRST waypoint is reached (not step-based)
        This ensures the robot can escape the stuck zone without collision penalties
        """
        # Grace period ends when first waypoint is reached
        if self.use_waypoints and len(self.waypoints) > 1:
            # First waypoint is the gap waypoint, once we pass it grace ends
            return self.current_waypoint_idx == 0
        else:
            # Fallback to step-based if no waypoints
            return self.step_count <= self.grace_period_steps
    
    def _check_collision(self) -> bool:
        """Check collision with obstacles, boundaries, and humans (with grace period)"""
        in_grace = self._is_in_grace_period()
        
        # Check human collisions (no grace period for humans - they move!)
        if self._check_human_collision():
            return True
        
        # Check circle obstacles
        for obs in self.obstacles:
            d = math.sqrt((self.robot_x - obs['x'])**2 + 
                         (self.robot_y - obs['y'])**2)
            
            # If in grace period, check if this was an initially close obstacle
            if in_grace and (obs['x'], obs['y']) in self._initial_close_obstacles:
                # Use a smaller collision distance for grace obstacles
                # Allow robot to be close but not drive INTO them
                effective_collision_dist = 0.3  # Reduced from 0.5
                if d < obs['r'] + effective_collision_dist:
                    # Only collide if robot is moving toward obstacle
                    if self.v > 0.1:  # Moving forward
                        # Check if obstacle is in front
                        obs_angle = math.atan2(obs['y'] - self.robot_y, 
                                             obs['x'] - self.robot_x)
                        angle_diff = abs((obs_angle - self.robot_theta + math.pi) % (2*math.pi) - math.pi)
                        if angle_diff < math.pi / 3:  # In front 60° cone
                            return True
            else:
                # Normal collision check
                if d < obs['r'] + self.collision_dist:
                    return True
        
        # Check boundary points
        for px, py in self.boundary_points:
            d = math.sqrt((self.robot_x - px)**2 + (self.robot_y - py)**2)
            
            if in_grace and (px, py) in self._initial_close_boundary:
                # Grace period: allow closer proximity but not driving into
                effective_collision_dist = 0.3
                if d < effective_collision_dist:
                    if self.v > 0.1:
                        obs_angle = math.atan2(py - self.robot_y, px - self.robot_x)
                        angle_diff = abs((obs_angle - self.robot_theta + math.pi) % (2*math.pi) - math.pi)
                        if angle_diff < math.pi / 3:
                            return True
            else:
                if d < self.collision_dist:
                    return True
        
        return False
    
    def _compute_reward(self, target: bool, collision: bool, 
                       timeout: bool, dist: float, lidar: np.ndarray) -> float:
        """
        Compute reward matching pygame_env_rosv5.py reward function
        """
        if target:
            return 10.0
        if collision:
            return -2.0
        if timeout:
            return -1.0
        
        # Progress reward
        progress = self.prev_dist - dist
        
        v_actual = self.v
        w_actual = self.w
        
        # === TURN PENALTY (matches training) ===
        forward_threshold = 0.1
        turn_velocity_threshold = 0.05
        turn_angle_threshold = np.pi / 2
        
        if v_actual > forward_threshold and abs(w_actual) < turn_velocity_threshold:
            self.accumulated_turn = 0.0
            self.turn_direction = 0
            turn_penalty = 0.0
        elif abs(w_actual) > turn_velocity_threshold:
            current_direction = np.sign(w_actual)
            if current_direction == self.turn_direction or self.turn_direction == 0:
                self.turn_direction = current_direction
                angular_displacement = abs(w_actual) * self.dt
                self.accumulated_turn += angular_displacement
                if self.accumulated_turn > turn_angle_threshold:
                    excess_turn = self.accumulated_turn - turn_angle_threshold
                    turn_penalty = -0.05 * excess_turn
                else:
                    turn_penalty = 0.0
            else:
                self.accumulated_turn = abs(w_actual) * self.dt
                self.turn_direction = current_direction
                turn_penalty = 0.0
        else:
            turn_penalty = 0.0
        
        # Forward bonus
        fwd_threshold = 0.5
        fwd_bonus = 0.01 if v_actual > fwd_threshold else 0.0
        
        # === SAFETY PENALTY (matches training) ===
        min_node_dist = float(np.min(lidar))
        
        safety_thresholds = [
            (1.0, 0.0),
            (0.9, -2.0/45.0),
            (0.8, -4.0/45.0),
            (0.7, -6.0/45.0),
            (0.6, -8.0/45.0),
            (0.0, -10.0/45.0)
        ]
        
        safety_penalty = 0.0
        for threshold, penalty in safety_thresholds:
            if min_node_dist <= threshold:
                safety_penalty = penalty
                break
        
        return progress + turn_penalty + safety_penalty + fwd_bonus
    
    def _get_state(self) -> np.ndarray:
        """Build state matching training format"""
        # Simulate LiDAR scan
        lidar = self._simulate_lidar()
        
        # Build node features
        state = self._build_node_features(lidar)
        
        return state
    
    def _simulate_lidar(self) -> np.ndarray:
        """Simulate LiDAR scan in the snapshot environment"""
        dists = np.ones(self.L) * LIDAR_RANGE
        
        for i in range(self.L):
            beam_angle = float(BEAM_ANGLES[i]) + self.robot_theta
            dx, dy = math.cos(beam_angle), math.sin(beam_angle)
            
            min_d = LIDAR_RANGE
            
            # Check obstacle intersections
            for obs in self.obstacles:
                d = self._ray_circle_intersect(
                    self.robot_x, self.robot_y, dx, dy,
                    obs['x'], obs['y'], obs['r']
                )
                if d is not None:
                    min_d = min(min_d, d)
            
            # Check boundary point proximity
            for px, py in self.boundary_points:
                # Simple distance check along ray
                d = self._ray_point_distance(
                    self.robot_x, self.robot_y, dx, dy, px, py
                )
                if d is not None:
                    min_d = min(min_d, d)
            
            dists[i] = min_d
        
        return dists
    
    def _ray_circle_intersect(self, rx, ry, dx, dy, cx, cy, r) -> Optional[float]:
        """Ray-circle intersection"""
        fx, fy = rx - cx, ry - cy
        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - r*r
        
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        
        disc = math.sqrt(disc)
        t1 = (-b - disc) / (2*a)
        t2 = (-b + disc) / (2*a)
        
        ts = [t for t in (t1, t2) if t >= 0 and t <= LIDAR_RANGE]
        return min(ts) if ts else None
    
    def _ray_point_distance(self, rx, ry, dx, dy, px, py) -> Optional[float]:
        """Distance from ray to point, if within threshold"""
        # Project point onto ray
        vx, vy = px - rx, py - ry
        t = max(0, vx*dx + vy*dy)  # projection distance
        
        # Point on ray closest to (px, py)
        closest_x = rx + t * dx
        closest_y = ry + t * dy
        
        # Distance from closest point to actual point
        perp_dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        
        # If perpendicular distance is small, return the ray distance
        if perp_dist < 0.3 and t > 0 and t < LIDAR_RANGE:
            return t
        return None
    
    def _build_node_features(self, lidar: np.ndarray) -> np.ndarray:
        """Build node features matching training format"""
        # LiDAR nodes (L x 4)
        lidar_feats = np.zeros((self.L, 4), dtype=np.float32)
        for i, d in enumerate(lidar):
            ang = float(BEAM_ANGLES[i])
            lidar_feats[i, 0] = min(d, LIDAR_RANGE) / LIDAR_RANGE
            lidar_feats[i, 1] = math.cos(ang)
            lidar_feats[i, 2] = math.sin(ang)
            lidar_feats[i, 3] = 0.0
        
        # Robot node (1 x 4)
        dx_world = self.goal_x - self.robot_x
        dy_world = self.goal_y - self.robot_y
        dx_ego = dx_world * math.cos(self.robot_theta) + dy_world * math.sin(self.robot_theta)
        dy_ego = -dx_world * math.sin(self.robot_theta) + dy_world * math.cos(self.robot_theta)
        
        world_max = 20.0
        robot_feats = np.array([[
            dx_ego / world_max,
            dy_ego / world_max,
            self.v,
            self.w
        ]], dtype=np.float32)
        
        # Human nodes (H x 6) - use MOVING simulated humans
        human_feats = np.zeros((self.H, 6), dtype=np.float32)
        for i, human in enumerate(self.sim_humans[:self.H]):
            # Current position (world to robot frame)
            dx = human['x'] - self.robot_x
            dy = human['y'] - self.robot_y
            x_rel = dx * math.cos(-self.robot_theta) - dy * math.sin(-self.robot_theta)
            y_rel = dx * math.sin(-self.robot_theta) + dy * math.cos(-self.robot_theta)
            
            # Predicted position (1 second ahead, also in robot frame)
            pred_x = human['x'] + human['vx'] * 1.0
            pred_y = human['y'] + human['vy'] * 1.0
            dx_pred = pred_x - self.robot_x
            dy_pred = pred_y - self.robot_y
            x_pred_rel = dx_pred * math.cos(-self.robot_theta) - dy_pred * math.sin(-self.robot_theta)
            y_pred_rel = dx_pred * math.sin(-self.robot_theta) + dy_pred * math.cos(-self.robot_theta)
            
            # Compute speed from velocity
            speed = math.sqrt(human['vx']**2 + human['vy']**2)
            
            human_feats[i] = [
                x_rel / 15.0,
                y_rel / 15.0,
                min(speed, 2.0) / 2.0,  # Normalized speed (0-1)
                human.get('confidence', 0.9),
                x_pred_rel / 15.0,  # Predicted position
                y_pred_rel / 15.0
            ]
        
        # Concatenate
        state = np.concatenate([
            lidar_feats.flatten(),
            robot_feats.flatten(),
            human_feats.flatten()
        ])
        
        return state


# ==================== RECOVERY ENVIRONMENT VISUALIZER ====================
class RecoveryVisualizer:
    """
    Pygame visualization window for the recovery environment
    Shows the snapshot with:
    - LiDAR boundary outline (joined dots)
    - Blocked regions (obstacles where gaps < 1.5m)
    - Graph nodes (LiDAR, Robot, Human)
    - Robot trajectory during training
    """
    
    def __init__(self, snapshot: EnvironmentSnapshot, 
                 window_title: str = "Recovery Training Environment",
                 ppm: int = 40,  # pixels per meter
                 window_size: Tuple[int, int] = (800, 800)):
        
        self.snapshot = snapshot
        self.ppm = ppm
        self.window_size = window_size
        self.window_title = window_title
        
        # Calculate world bounds from LiDAR points
        if len(snapshot.lidar_points_world) > 0:
            points = snapshot.lidar_points_world
            self.world_min_x = np.min(points[:, 0]) - 2
            self.world_max_x = np.max(points[:, 0]) + 2
            self.world_min_y = np.min(points[:, 1]) - 2
            self.world_max_y = np.max(points[:, 1]) + 2
        else:
            # Fallback to robot position + range
            self.world_min_x = snapshot.robot_x - 12
            self.world_max_x = snapshot.robot_x + 12
            self.world_min_y = snapshot.robot_y - 12
            self.world_max_y = snapshot.robot_y + 12
        
        self.world_width = self.world_max_x - self.world_min_x
        self.world_height = self.world_max_y - self.world_min_y
        
        # Initialize pygame
        self._pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        
        # Trajectory history
        self.trajectory = []
        
    def _init_pygame(self):
        """Initialize pygame (lazy initialization)"""
        if self._pygame_initialized:
            return True
        
        try:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption(self.window_title)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 14)
            self.font_large = pygame.font.SysFont('Arial', 18, bold=True)
            self._pygame_initialized = True
            return True
        except Exception as e:
            print(f"[RecoveryVisualizer] Could not initialize pygame: {e}")
            return False
    
    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Normalize to 0-1
        nx = (wx - self.world_min_x) / self.world_width
        ny = (wy - self.world_min_y) / self.world_height
        
        # Convert to screen (Y inverted)
        sx = int(nx * self.window_size[0])
        sy = int((1 - ny) * self.window_size[1])
        
        return sx, sy
    
    def add_trajectory_point(self, x: float, y: float):
        """Add a point to the trajectory"""
        self.trajectory.append((x, y))
    
    def clear_trajectory(self):
        """Clear the trajectory"""
        self.trajectory = []
    
    def render(self, robot_x: float, robot_y: float, robot_theta: float,
               lidar_ranges: Optional[np.ndarray] = None,
               episode: int = 0, step: int = 0, reward: float = 0.0,
               show_edges: bool = True,
               waypoints: Optional[List[Tuple[float, float]]] = None,
               current_waypoint_idx: int = 0,
               current_goal: Optional[Tuple[float, float]] = None,
               sim_humans: Optional[List[Dict]] = None) -> bool:
        """
        Render the recovery environment
        
        Returns True if window should stay open, False if user closed it
        """
        if not self._init_pygame():
            return False
        
        import pygame
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    self._pygame_initialized = False
                    return False
        
        # Clear screen (dark background)
        self.screen.fill((25, 25, 35))
        
        # Draw grid
        self._draw_grid()
        
        # Draw LiDAR boundary (connected dots)
        self._draw_lidar_boundary()
        
        # Draw blocked regions (virtual obstacles)
        self._draw_blocked_regions()
        
        # Draw goal and waypoints
        self._draw_goal(waypoints=waypoints, 
                       current_waypoint_idx=current_waypoint_idx,
                       current_goal=current_goal)
        
        # Draw trajectory
        self._draw_trajectory()
        
        # Draw robot with LiDAR rays
        self._draw_robot(robot_x, robot_y, robot_theta, lidar_ranges)
        
        # Draw humans (moving ones if provided, else from snapshot)
        self._draw_humans(sim_humans)
        
        # Draw LiDAR nodes
        self._draw_lidar_nodes(robot_x, robot_y, robot_theta, lidar_ranges)
        
        # Draw edges if requested
        if show_edges and lidar_ranges is not None:
            self._draw_graph_edges(robot_x, robot_y, robot_theta, lidar_ranges)
        
        # Draw info panel
        self._draw_info_panel(episode, step, reward, robot_x, robot_y)
        
        # Draw legend
        self._draw_legend()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
        return True
    
    def _draw_grid(self):
        """Draw background grid"""
        import pygame
        grid_color = (40, 40, 50)
        
        # Draw vertical lines every meter
        for x in np.arange(math.floor(self.world_min_x), math.ceil(self.world_max_x) + 1, 1.0):
            sx, _ = self._world_to_screen(x, 0)
            pygame.draw.line(self.screen, grid_color, (sx, 0), (sx, self.window_size[1]), 1)
        
        # Draw horizontal lines every meter
        for y in np.arange(math.floor(self.world_min_y), math.ceil(self.world_max_y) + 1, 1.0):
            _, sy = self._world_to_screen(0, y)
            pygame.draw.line(self.screen, grid_color, (0, sy), (self.window_size[0], sy), 1)
    
    def _draw_lidar_boundary(self):
        """Draw LiDAR boundary points and connections"""
        import pygame
        
        if len(self.snapshot.lidar_points_world) < 2:
            return
        
        boundary_color = (120, 120, 160)
        node_color = (180, 180, 220)
        
        # Draw boundary polygon if available
        if len(self.snapshot.boundary_polygon) > 2:
            screen_points = [self._world_to_screen(p[0], p[1]) 
                           for p in self.snapshot.boundary_polygon]
            pygame.draw.polygon(self.screen, (40, 40, 60), screen_points, 0)  # Fill
            pygame.draw.polygon(self.screen, boundary_color, screen_points, 2)  # Outline
        
        # Draw individual LiDAR points
        for point in self.snapshot.lidar_points_world:
            sx, sy = self._world_to_screen(point[0], point[1])
            pygame.draw.circle(self.screen, node_color, (sx, sy), 3)
    
    def _draw_blocked_regions(self):
        """Draw blocked regions (gaps too small for Husky)"""
        import pygame
        
        for bx, by, br in self.snapshot.blocked_regions:
            sx, sy = self._world_to_screen(bx, by)
            sr = int(br * self.ppm)
            
            # Red circle for blocked region
            pygame.draw.circle(self.screen, (200, 60, 60), (sx, sy), max(sr, 5), 0)
            pygame.draw.circle(self.screen, (255, 100, 100), (sx, sy), max(sr, 5), 2)
            
            # X mark
            pygame.draw.line(self.screen, (255, 100, 100), 
                           (sx - 5, sy - 5), (sx + 5, sy + 5), 2)
            pygame.draw.line(self.screen, (255, 100, 100), 
                           (sx - 5, sy + 5), (sx + 5, sy - 5), 2)
    
    def _draw_humans(self, sim_humans: Optional[List[Dict]] = None):
        """Draw humans with pale circle and 'H' label"""
        import pygame
        
        # Use moving sim_humans if provided, else fall back to snapshot
        humans = sim_humans if sim_humans else self.snapshot.humans
        
        for human in humans:
            # Get position (different keys for sim_humans vs snapshot)
            if 'x' in human:  # sim_humans format
                hx, hy = human['x'], human['y']
                vx, vy = human.get('vx', 0), human.get('vy', 0)
            else:  # snapshot format
                hx = human.get('x_odom', 0)
                hy = human.get('y_odom', 0)
                vx, vy = human.get('vx', 0), human.get('vy', 0)
            
            sx, sy = self._world_to_screen(hx, hy)
            radius = int(0.4 * self.ppm)  # 0.4m radius
            
            # Pale peach circle
            pygame.draw.circle(self.screen, (255, 220, 180), (sx, sy), radius, 0)
            # Darker outline
            pygame.draw.circle(self.screen, (180, 140, 100), (sx, sy), radius, 2)
            
            # Draw 'H' label in center
            h_font = pygame.font.SysFont('Arial', int(radius * 1.4), bold=True)
            h_text = h_font.render('H', True, (80, 50, 30))  # Dark brown
            text_rect = h_text.get_rect(center=(sx, sy))
            self.screen.blit(h_text, text_rect)
            
            # Draw velocity arrow if moving
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 0.1:
                arrow_len = min(speed * self.ppm * 0.5, 30)  # Max 30px
                angle = math.atan2(vy, vx)
                end_x = sx + int(arrow_len * math.cos(angle))
                end_y = sy - int(arrow_len * math.sin(angle))  # Flip Y for screen coords
                pygame.draw.line(self.screen, (255, 150, 100), (sx, sy), (end_x, end_y), 2)
                # Arrowhead
                pygame.draw.circle(self.screen, (255, 150, 100), (end_x, end_y), 3, 0)
    
    def _draw_goal(self, waypoints: Optional[List[Tuple[float, float]]] = None,
                   current_waypoint_idx: int = 0,
                   current_goal: Optional[Tuple[float, float]] = None):
        """Draw the goal position and waypoints"""
        import pygame
        
        # Draw waypoints if provided
        if waypoints:
            for i, (wx, wy) in enumerate(waypoints[:-1]):  # Don't draw final (it's the goal)
                sx, sy = self._world_to_screen(wx, wy)
                
                if i < current_waypoint_idx:
                    # Completed waypoint - green checkmark
                    color = (100, 200, 100)
                    pygame.draw.circle(self.screen, color, (sx, sy), 10, 0)
                    # Checkmark
                    pygame.draw.line(self.screen, (255, 255, 255), 
                                   (sx - 5, sy), (sx - 2, sy + 4), 2)
                    pygame.draw.line(self.screen, (255, 255, 255), 
                                   (sx - 2, sy + 4), (sx + 5, sy - 4), 2)
                elif i == current_waypoint_idx:
                    # Current waypoint - bright yellow target
                    pygame.draw.circle(self.screen, (255, 255, 0), (sx, sy), 12, 0)
                    pygame.draw.circle(self.screen, (255, 200, 0), (sx, sy), 12, 2)
                    pygame.draw.circle(self.screen, (255, 100, 0), (sx, sy), 6, 0)
                    label = self.font.render(f"WP{i}", True, (255, 255, 0))
                    self.screen.blit(label, (sx + 15, sy - 8))
                else:
                    # Future waypoint - dim
                    pygame.draw.circle(self.screen, (100, 100, 150), (sx, sy), 8, 0)
                    pygame.draw.circle(self.screen, (150, 150, 200), (sx, sy), 8, 1)
            
            # Draw lines connecting waypoints
            if len(waypoints) > 1:
                for i in range(len(waypoints) - 1):
                    p1 = self._world_to_screen(waypoints[i][0], waypoints[i][1])
                    p2 = self._world_to_screen(waypoints[i+1][0], waypoints[i+1][1])
                    
                    if i < current_waypoint_idx:
                        line_color = (80, 150, 80)  # Completed - green
                    elif i == current_waypoint_idx:
                        line_color = (200, 200, 100)  # Current - yellow
                    else:
                        line_color = (80, 80, 120)  # Future - dim
                    
                    pygame.draw.line(self.screen, line_color, p1, p2, 2)
        
        # Draw current goal indicator (arrow from robot)
        if current_goal:
            cg_sx, cg_sy = self._world_to_screen(current_goal[0], current_goal[1])
            # Pulsing circle around current goal
            pulse = int(10 + 3 * math.sin(time.time() * 5))
            pygame.draw.circle(self.screen, (255, 200, 0), (cg_sx, cg_sy), pulse, 2)
        
        # Draw final goal (star)
        gx, gy = self.snapshot.goal_x, self.snapshot.goal_y
        sx, sy = self._world_to_screen(gx, gy)
        
        # Star shape
        star_radius = 15
        inner_radius = 6
        star_points = []
        for i in range(10):
            angle = math.pi / 2 + (i * math.pi / 5)
            radius = star_radius if i % 2 == 0 else inner_radius
            px = sx + int(radius * math.cos(angle))
            py = sy - int(radius * math.sin(angle))
            star_points.append((px, py))
        
        pygame.draw.polygon(self.screen, (100, 255, 100), star_points, 0)
        pygame.draw.polygon(self.screen, (60, 200, 60), star_points, 2)
        
        # Label
        label = self.font.render("GOAL", True, (100, 255, 100))
        self.screen.blit(label, (sx - 15, sy + 20))
    
    def _draw_trajectory(self):
        """Draw robot trajectory"""
        import pygame
        
        if len(self.trajectory) < 2:
            return
        
        # Draw as gradient from old (dim) to new (bright)
        n = len(self.trajectory)
        for i in range(1, n):
            alpha = i / n
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            
            p1 = self._world_to_screen(self.trajectory[i-1][0], self.trajectory[i-1][1])
            p2 = self._world_to_screen(self.trajectory[i][0], self.trajectory[i][1])
            
            pygame.draw.line(self.screen, color, p1, p2, 2)
    
    def _draw_robot(self, robot_x: float, robot_y: float, robot_theta: float,
                    lidar_ranges: Optional[np.ndarray] = None):
        """Draw robot and optional LiDAR rays"""
        import pygame
        
        sx, sy = self._world_to_screen(robot_x, robot_y)
        
        # Robot body (rectangle)
        robot_length = 0.9  # Husky length
        robot_width = 0.5   # Husky width
        
        # Calculate corners
        cos_th = math.cos(robot_theta)
        sin_th = math.sin(robot_theta)
        
        corners = [
            (-robot_length/2, -robot_width/2),
            (robot_length/2, -robot_width/2),
            (robot_length/2, robot_width/2),
            (-robot_length/2, robot_width/2)
        ]
        
        screen_corners = []
        for lx, ly in corners:
            wx = robot_x + lx * cos_th - ly * sin_th
            wy = robot_y + lx * sin_th + ly * cos_th
            screen_corners.append(self._world_to_screen(wx, wy))
        
        pygame.draw.polygon(self.screen, (220, 220, 220), screen_corners)
        pygame.draw.polygon(self.screen, (180, 180, 180), screen_corners, 2)
        
        # Heading arrow
        arrow_length = robot_length * 0.8
        arrow_end_x = robot_x + arrow_length * cos_th
        arrow_end_y = robot_y + arrow_length * sin_th
        arrow_end = self._world_to_screen(arrow_end_x, arrow_end_y)
        pygame.draw.line(self.screen, (255, 255, 0), (sx, sy), arrow_end, 3)
        
        # Draw LiDAR rays if provided
        if lidar_ranges is not None:
            for i, r in enumerate(lidar_ranges):
                beam_angle = float(BEAM_ANGLES[i]) + robot_theta
                
                # Limit visualization range
                vis_r = min(r, LIDAR_RANGE)
                
                end_x = robot_x + vis_r * math.cos(beam_angle)
                end_y = robot_y + vis_r * math.sin(beam_angle)
                end_screen = self._world_to_screen(end_x, end_y)
                
                # Color based on distance
                if r < 1.0:
                    ray_color = (255, 100, 100)  # Red - close
                elif r < 2.0:
                    ray_color = (255, 200, 100)  # Orange - medium
                else:
                    ray_color = (100, 200, 100)  # Green - far
                
                pygame.draw.line(self.screen, (50, 50, 60), (sx, sy), end_screen, 1)
    
    def _draw_lidar_nodes(self, robot_x: float, robot_y: float, robot_theta: float,
                          lidar_ranges: Optional[np.ndarray] = None):
        """Draw LiDAR nodes as graph nodes"""
        import pygame
        
        if lidar_ranges is None:
            return
        
        for i, r in enumerate(lidar_ranges):
            beam_angle = float(BEAM_ANGLES[i]) + robot_theta
            
            node_x = robot_x + r * math.cos(beam_angle)
            node_y = robot_y + r * math.sin(beam_angle)
            node_screen = self._world_to_screen(node_x, node_y)
            
            # Node color based on distance
            if r < 1.0:
                node_color = (255, 100, 100)
            elif r < 2.0:
                node_color = (255, 200, 100)
            else:
                node_color = (150, 200, 255)
            
            pygame.draw.circle(self.screen, node_color, node_screen, 6, 0)
            pygame.draw.circle(self.screen, (255, 255, 255), node_screen, 6, 1)
    
    def _draw_graph_edges(self, robot_x: float, robot_y: float, robot_theta: float,
                          lidar_ranges: np.ndarray):
        """Draw edges between nodes that are connected in the graph"""
        import pygame
        
        L = len(BEAM_ANGLES)
        robot_screen = self._world_to_screen(robot_x, robot_y)
        
        # Get LiDAR node positions
        lidar_nodes = []
        for i, r in enumerate(lidar_ranges):
            beam_angle = float(BEAM_ANGLES[i]) + robot_theta
            node_x = robot_x + r * math.cos(beam_angle)
            node_y = robot_y + r * math.sin(beam_angle)
            lidar_nodes.append(self._world_to_screen(node_x, node_y))
        
        # Draw robot-to-LiDAR edges (robot attends to all LiDAR)
        for node in lidar_nodes:
            pygame.draw.line(self.screen, (70, 100, 130), robot_screen, node, 1)
        
        # Draw LiDAR-to-LiDAR edges (adjacent)
        for i in range(L - 1):
            # Check if gap is small (connected nodes)
            r1, r2 = lidar_ranges[i], lidar_ranges[i + 1]
            a1 = float(BEAM_ANGLES[i]) + robot_theta
            a2 = float(BEAM_ANGLES[i + 1]) + robot_theta
            
            x1 = robot_x + r1 * math.cos(a1)
            y1 = robot_y + r1 * math.sin(a1)
            x2 = robot_x + r2 * math.cos(a2)
            y2 = robot_y + r2 * math.sin(a2)
            
            gap = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if gap < 1.5:  # Connected (gap too small for Husky)
                edge_color = (200, 100, 100)  # Red - blocked path
                edge_width = 2
            else:
                edge_color = (80, 120, 80)  # Green - passable
                edge_width = 1
            
            pygame.draw.line(self.screen, edge_color, lidar_nodes[i], lidar_nodes[i+1], edge_width)
    
    def _draw_info_panel(self, episode: int, step: int, reward: float, 
                         robot_x: float, robot_y: float):
        """Draw information panel"""
        import pygame
        
        panel_x = 10
        panel_y = 10
        line_height = 20
        
        # Background
        panel_rect = pygame.Rect(panel_x - 5, panel_y - 5, 200, 130)
        pygame.draw.rect(self.screen, (30, 30, 40), panel_rect)
        pygame.draw.rect(self.screen, (80, 80, 100), panel_rect, 1)
        
        # Title
        title = self.font_large.render("RECOVERY TRAINING", True, (255, 200, 100))
        self.screen.blit(title, (panel_x, panel_y))
        
        # Info lines
        info_lines = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {reward:.3f}",
            f"Robot: ({robot_x:.2f}, {robot_y:.2f})",
            f"Goal: ({self.snapshot.goal_x:.2f}, {self.snapshot.goal_y:.2f})",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (200, 200, 220))
            self.screen.blit(text, (panel_x, panel_y + 25 + i * line_height))
    
    def _draw_legend(self):
        """Draw color legend"""
        import pygame
        
        legend_x = self.window_size[0] - 180
        legend_y = 10
        line_height = 20
        
        # Background
        panel_rect = pygame.Rect(legend_x - 5, legend_y - 5, 175, 150)
        pygame.draw.rect(self.screen, (30, 30, 40), panel_rect)
        pygame.draw.rect(self.screen, (80, 80, 100), panel_rect, 1)
        
        # Title
        title = self.font_large.render("LEGEND", True, (255, 255, 255))
        self.screen.blit(title, (legend_x, legend_y))
        
        # Legend entries
        entries = [
            ((100, 255, 100), "Goal"),
            ((200, 60, 60), "Blocked Region"),
            ((180, 180, 220), "LiDAR Boundary"),
            ((200, 100, 100), "Gap < 1.5m (blocked)"),
            ((80, 120, 80), "Gap >= 1.5m (passable)"),
            ((100, 200, 255), "Trajectory"),
        ]
        
        for i, (color, label) in enumerate(entries):
            # Color sample
            pygame.draw.rect(self.screen, color, 
                           (legend_x, legend_y + 25 + i * line_height, 15, 12))
            # Label
            text = self.font.render(label, True, (200, 200, 220))
            self.screen.blit(text, (legend_x + 20, legend_y + 23 + i * line_height))
    
    def close(self):
        """Close the visualization window"""
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False


# ==================== QUICK RETRAIN MODULE ====================
class QuickRetrainer:
    """
    Quick retraining module for stuck recovery
    Clones the original model and fine-tunes for the specific situation
    """
    
    def __init__(self, gat_path: str, actor_path: str,
                 learning_rate: float = 3e-4,
                 max_episodes: int = 50,
                 max_steps_per_episode: int = 200):
        
        self.gat_path = gat_path
        self.actor_path = actor_path
        self.lr = learning_rate
        self.max_episodes = max_episodes
        self.max_steps = max_steps_per_episode
        
        # Model parameters
        self.L = LIDAR_BEAMS
        self.R = 1
        self.H = MAX_HUMANS
        self.total_nodes = self.L + self.R + self.H
        self.gat_out = 16
        
        self.edge_index = create_edge_index(self.L, self.R, self.H).to(device)
        
        # Original models (for reference)
        self.original_gat = None
        self.original_actor = None
        
        # Recovery models (will be trained)
        self.recovery_gat = None
        self.recovery_actor = None
        self.recovery_critic = None
        self.train_episodes = []
        self.goto_full_log = []
        self.waypoint_chain = []
        
    def load_models(self):
        """Load original models"""
        # Create original models
        self.original_gat = GATNetwork(
            lidar_channels=4, robot_channels=4, human_channels=6,
            out_channels=self.gat_out, num_heads=4,
            lidar_nodes=self.L, robot_nodes=self.R, human_nodes=self.H
        ).to(device)
        
        state_dim = self.gat_out * self.total_nodes
        self.original_actor = Actor(state_dim, action_dim=2).to(device)
        
        # Load weights
        self.original_gat.load_state_dict(
            torch.load(self.gat_path, map_location=device)
        )
        self.original_actor.load_state_dict(
            torch.load(self.actor_path, map_location=device)
        )
        
        self.original_gat.eval()
        self.original_actor.eval()
        
        print("[QuickRetrainer] Original models loaded")
        
    def clone_for_recovery(self):
        """Create cloned models for recovery training"""
        # Deep copy the models
        self.recovery_gat = GATNetwork(
            lidar_channels=4, robot_channels=4, human_channels=6,
            out_channels=self.gat_out, num_heads=4,
            lidar_nodes=self.L, robot_nodes=self.R, human_nodes=self.H
        ).to(device)
        
        state_dim = self.gat_out * self.total_nodes
        self.recovery_actor = Actor(state_dim, action_dim=2).to(device)
        self.recovery_critic = Critic(state_dim, action_dim=2).to(device)
        
        # Copy weights
        self.recovery_gat.load_state_dict(self.original_gat.state_dict())
        self.recovery_actor.load_state_dict(self.original_actor.state_dict())
        
        # Set to training mode
        self.recovery_gat.train()
        self.recovery_actor.train()
        self.recovery_critic.train()
        
        # Create optimizers (higher learning rate for quick adaptation)
        self.actor_opt = torch.optim.Adam(self.recovery_actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.recovery_critic.parameters(), lr=self.lr)
        self.gat_opt = torch.optim.Adam(self.recovery_gat.parameters(), lr=self.lr * 0.1)  # Lower LR for GAT
        
        print("[QuickRetrainer] Recovery models cloned")
    
    def _flat_to_nodes(self, flat_state: np.ndarray):
        """Convert flat state to node tensors"""
        lidar_size = self.L * 4
        robot_size = self.R * 4
        human_size = self.H * 6
        
        lidar = torch.tensor(flat_state[:lidar_size], 
                            dtype=torch.float32, device=device).view(self.L, 4)
        robot = torch.tensor(flat_state[lidar_size:lidar_size + robot_size],
                            dtype=torch.float32, device=device).view(self.R, 4)
        human = torch.tensor(flat_state[lidar_size + robot_size:lidar_size + robot_size + human_size],
                            dtype=torch.float32, device=device).view(self.H, 6)
        
        return lidar, robot, human
    
    @torch.no_grad()
    def get_action(self, flat_state: np.ndarray, explore: bool = False, 
                   noise_scale: float = 0.05) -> np.ndarray:
        """Get action from recovery policy"""
        lidar, robot, human = self._flat_to_nodes(flat_state)
        z = self.recovery_gat(lidar, robot, human, self.edge_index).view(1, -1)
        action = self.recovery_actor(z).cpu().numpy().flatten()
        
        if explore:
            # Lower noise to preserve policy behavior
            noise = np.random.normal(0, noise_scale, size=2)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def _goto_command(self, env) -> np.ndarray:
        """Proportional go-to-goal in env units: linear [0, 1], angular [-1, 1]."""
        tx, ty = env.goal_x, env.goal_y
        err = math.atan2(ty - env.robot_y, tx - env.robot_x) - env.robot_theta
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        w = float(np.clip(err / 0.8, -1.0, 1.0))
        # Keep a little forward speed while turning. Stopping until the
        # heading is perfect is what made the twin (and then the robot) spin.
        if abs(err) > 2.2:
            v = 0.0
        else:
            v = float(np.clip(0.35 + 0.65 * math.cos(err), 0.2, 1.0))
        return np.array([v, w], dtype=np.float32)
    
    def _rollout_goto(self, env, visualizer, verbose: bool):
        """
        Drive to the current waypoint with go-to-goal, then the same
        controller on to the final goal. The env switches its target
        when WP1 is reached.
        """
        env.reset()
        log = []
        trajectory = [(env.robot_x, env.robot_y)]
        reached_wp = False
        reached_goal = False
        pose = None
        has_wp1 = len(env.waypoints) > 1
        
        if visualizer:
            visualizer.clear_trajectory()
            visualizer.add_trajectory_point(env.robot_x, env.robot_y)
        
        for step in range(self.max_steps * 2):
            if has_wp1 and env.current_waypoint_idx == 0:
                segment = 'wp1'
            else:
                segment = 'goal'
            action_env = self._goto_command(env)
            log.append({
                'action': action_env.copy(),
                'x': env.robot_x,
                'y': env.robot_y,
                'theta': env.robot_theta,
                'step': step,
                'segment': segment,
            })
            _next, _reward, done, target, _collision = env.step(action_env)
            trajectory.append((env.robot_x, env.robot_y))
            
            if visualizer:
                visualizer.add_trajectory_point(env.robot_x, env.robot_y)
                lidar = env._simulate_lidar()
                if not visualizer.render(
                    env.robot_x, env.robot_y, env.robot_theta,
                    lidar_ranges=lidar,
                    episode=0, step=step, reward=0.0,
                    show_edges=True,
                    waypoints=env.waypoints if env.use_waypoints else None,
                    current_waypoint_idx=env.current_waypoint_idx,
                    current_goal=(env.goal_x, env.goal_y),
                    sim_humans=env.sim_humans
                ):
                    visualizer = None
            
            if pose is None and env.current_waypoint_idx > 0:
                reached_wp = True
                wx, wy = env.waypoints[env.current_waypoint_idx - 1]
                pose = (float(wx), float(wy), float(env.robot_theta))
                if verbose:
                    print(f"[Goto] WP1 reached at step {step + 1}: "
                          f"({wx:.2f}, {wy:.2f}), heading {math.degrees(env.robot_theta):.1f}°")
            
            if target:
                reached_goal = True
                if pose is None:
                    pose = (float(env.final_goal_x), float(env.final_goal_y), float(env.robot_theta))
                if verbose:
                    print(f"[Goto] Goal reached at step {step + 1}")
                break
            if done:
                if verbose:
                    print(f"[Goto] Stopped at step {step + 1} "
                          f"(dist {env._dist_to_goal():.2f} m)")
                break
        
        return log, trajectory, reached_wp, reached_goal, pose
    
    def train_on_snapshot(self, snapshot: EnvironmentSnapshot,
                         verbose: bool = True,
                         visualize: bool = False) -> Tuple[bool, List[np.ndarray], list, Optional[Tuple[float, float, float]]]:
        """
        Train recovery model on snapshot environment
        
        Args:
            snapshot: Environment snapshot to train on
            verbose: Print progress
            visualize: Open pygame window showing training
        
        Returns:
            success: Whether a successful path was found
            trajectory: List of (x, y) positions if successful
            action_log: List of actions to reach first waypoint (for replay on real robot)
            waypoint_pose: (waypoint_x, waypoint_y, robot_theta) when the DT
                first reached that waypoint, or None
        """
        # Create simulation environment with extended grace period
        # Larger grace period helps robot escape initial close obstacles
        env = RecoverySimEnv(snapshot, self.L, self.H,
                            grace_period_steps=25,      # Increased from 10
                            collision_grace_dist=0.9)   # Increased from 0.6
        
        # Create visualizer if requested
        visualizer = None
        if visualize:
            visualizer = RecoveryVisualizer(snapshot, "Recovery Training")
            print("[QuickRetrainer] Visualization window opened")
        
        # Simple replay buffer for quick training
        replay_buffer = []
        
        success = False
        best_trajectory = []
        best_dist_to_goal = float('inf')
        
        self.train_episodes = []
        print(f"\n[QuickRetrainer] Starting recovery training...")
        print(f"  Grace period: {env.grace_period_steps} steps")
        print(f"  Initial close obstacles: {len(env._initial_close_obstacles)}")
        print(f"  Initial close boundary pts: {len(env._initial_close_boundary)}")
        print(f"  Waypoints: {len(env.waypoints)}")
        
        # Debug: Test that cloned model produces reasonable actions
        test_state = env.reset()
        test_action = self.get_action(test_state, explore=False)
        print(f"  Initial test action (no noise): linear={test_action[0]:.3f}, angular={test_action[1]:.3f}")
        
        # Debug: Show goal info from state
        lidar_size = self.L * 4
        robot_size = self.R * 4
        robot_feats = test_state[lidar_size:lidar_size + robot_size]
        print(f"  Goal in robot frame: dx={robot_feats[0]*20:.2f}m, dy={robot_feats[1]*20:.2f}m")
        
        # Warmup parameters
        CRITIC_WARMUP_STEPS = 64  # Train critic only for first N transitions
        POLICY_UPDATE_FREQ = 2   # Update actor every N critic updates
        train_steps = 0
        
        # === ACTION LOGGING FOR REPLAY ===
        # Log actions to reach FIRST waypoint - this is the critical gap crossing
        best_action_log = []  # Actions to reach first waypoint in best episode
        best_first_waypoint_reached = False
        # (waypoint_x, waypoint_y, robot_theta) when the DT reached the first waypoint
        best_waypoint_pose = None
        
        # Go-to-goal to WP1, then the same controller to the goal.
        # These commands are what gets logged and replayed. The policy is not
        # used to find the path.
        goto_log, goto_traj, reached_wp, reached_goal, goto_pose = self._rollout_goto(
            env, visualizer, verbose
        )
        if visualizer is not None:
            visualizer.close()
        self.goto_full_log = goto_log
        n_wp = sum(1 for e in goto_log if e.get('segment') == 'wp1')
        n_goal = sum(1 for e in goto_log if e.get('segment') == 'goal')
        print(f"[Goto] Logged {n_wp} actions to WP1 and {n_goal} actions to the goal")
        final_dist = math.hypot(env.final_goal_x - env.robot_x, env.final_goal_y - env.robot_y)
        self.train_episodes.append({
            'episode': 0,
            'reward': 0.0,
            'final_dist': float(final_dist),
            'steps': int(len(goto_log)),
            'reached_goal': bool(reached_goal),
            'first_waypoint_step': n_wp if n_wp else None,
            'buffer': 0,
            'method': 'goto',
        })
        replay_log = goto_log
        self.waypoint_chain = [(float(a), float(b)) for a, b in env.waypoints]
        success = len(self.waypoint_chain) > 0
        return success, goto_traj, replay_log, goto_pose

        for episode in range(self.max_episodes):
            state = env.reset()
            trajectory = [(env.robot_x, env.robot_y)]
            episode_reward = 0
            
            # Track actions for this episode
            episode_action_log = []  # (action_env, robot_x, robot_y, robot_theta)
            first_waypoint_step = None  # Step when first waypoint was reached
            episode_waypoint_pose = None  # (wx, wy, theta) at first waypoint
            
            # Clear visualizer trajectory
            if visualizer:
                visualizer.clear_trajectory()
                visualizer.add_trajectory_point(env.robot_x, env.robot_y)
            
            # Annealing exploration noise: start low, stay low
            noise_scale = 0.03  # Very low noise to preserve original policy
            
            for step in range(self.max_steps):
                # Get action with low exploration noise
                action = self.get_action(state, explore=True, noise_scale=noise_scale)
                
                # Convert to env format
                action_env = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float32)
                
                # === LOG ACTION FOR REPLAY ===
                # Log action with pose BEFORE step (so we know where to apply it)
                prev_waypoint_idx = env.current_waypoint_idx
                episode_action_log.append({
                    'action': action_env.copy(),
                    'x': env.robot_x,
                    'y': env.robot_y,
                    'theta': env.robot_theta,
                    'step': step
                })
                
                # Step environment
                next_state, reward, done, target, collision = env.step(action_env)
                
                # Check if first waypoint was just reached
                if first_waypoint_step is None and env.current_waypoint_idx > 0:
                    first_waypoint_step = step + 1
                    reached_idx = env.current_waypoint_idx - 1
                    wx, wy = env.waypoints[reached_idx]
                    episode_waypoint_pose = (float(wx), float(wy), float(env.robot_theta))
                    print(f"[QuickRetrainer] First waypoint reached at step {first_waypoint_step}! "
                          f"pose=({wx:.2f}, {wy:.2f}, {math.degrees(env.robot_theta):.1f}°)")
                
                # Store transition
                replay_buffer.append((state, action, reward, next_state, float(done)))
                
                # Track trajectory
                trajectory.append((env.robot_x, env.robot_y))
                episode_reward += reward
                
                # Update visualizer
                if visualizer:
                    visualizer.add_trajectory_point(env.robot_x, env.robot_y)
                    lidar = env._simulate_lidar()
                    
                    # Render with waypoint info and moving humans (returns False if window closed)
                    if not visualizer.render(
                        env.robot_x, env.robot_y, env.robot_theta,
                        lidar_ranges=lidar,
                        episode=episode, step=step, reward=reward,
                        show_edges=True,
                        waypoints=env.waypoints if env.use_waypoints else None,
                        current_waypoint_idx=env.current_waypoint_idx,
                        current_goal=(env.goal_x, env.goal_y),
                        sim_humans=env.sim_humans  # Pass moving humans
                    ):
                        # User closed window
                        visualizer = None
                
                state = next_state
                
                if target:
                    success = True
                    best_trajectory = trajectory
                    
                    # === SAVE ACTION LOG TO FIRST WAYPOINT ===
                    if first_waypoint_step is not None:
                        best_action_log = episode_action_log[:first_waypoint_step]
                        best_first_waypoint_reached = True
                        best_waypoint_pose = episode_waypoint_pose
                        print(f"[QuickRetrainer] Saved {len(best_action_log)} actions to reach first waypoint")
                    else:
                        # No waypoints or reached goal before first waypoint
                        best_action_log = episode_action_log
                        best_waypoint_pose = (
                            float(env.final_goal_x), float(env.final_goal_y), float(env.robot_theta)
                        )
                        print(f"[QuickRetrainer] Saved {len(best_action_log)} actions (full path)")
                    
                    if verbose:
                        print(f"[QuickRetrainer] SUCCESS at episode {episode}, step {step}")
                    break
                
                if done:
                    break
            
            # Track best attempt
            final_dist = env._dist_to_goal()
            if final_dist < best_dist_to_goal:
                best_dist_to_goal = final_dist
                if not success:
                    best_trajectory = trajectory
                    # Also save action log for best non-successful attempt
                    if first_waypoint_step is not None:
                        best_action_log = episode_action_log[:first_waypoint_step]
                        best_waypoint_pose = episode_waypoint_pose
                    elif not best_first_waypoint_reached:
                        best_action_log = episode_action_log
            
            # Train on collected data with warmup
            if len(replay_buffer) >= 32:
                # During warmup: only train critic
                # After warmup: train both with delayed policy updates
                in_warmup = len(replay_buffer) < CRITIC_WARMUP_STEPS
                update_actor = not in_warmup and (train_steps % POLICY_UPDATE_FREQ == 0)
                
                self._train_step(replay_buffer[-128:], update_actor=update_actor)
                train_steps += 1
            
            episode_record = {
                'episode': int(episode),
                'reward': float(episode_reward),
                'final_dist': float(final_dist),
                'steps': int(env.step_count),
                'reached_goal': bool(success),
                'first_waypoint_step': None if first_waypoint_step is None else int(first_waypoint_step),
                'buffer': int(len(replay_buffer)),
            }
            self.train_episodes.append(episode_record)
            if verbose and (episode % 10 == 0 or success):
                warmup_status = "(warmup)" if len(replay_buffer) < CRITIC_WARMUP_STEPS else ""
                print(f"[QuickRetrainer] Episode {episode}: reward={episode_reward:.2f}, "
                      f"dist={final_dist:.2f}, buffer={len(replay_buffer)} {warmup_status}")
            
            # Early exit if successful
            if success:
                # Do a few more training iterations to consolidate
                for _ in range(5):
                    if len(replay_buffer) >= 32:
                        self._train_step(replay_buffer[-128:])
                break
        
        # Close visualizer
        if visualizer:
            visualizer.close()
        
        # Log the action sequence summary
        if best_action_log and verbose:
            print(f"\n[QuickRetrainer] Action replay sequence:")
            print(f"  Total actions logged: {len(best_action_log)}")
            if best_action_log:
                first_pos = best_action_log[0]
                last_pos = best_action_log[-1]
                print(f"  Start: ({first_pos['x']:.2f}, {first_pos['y']:.2f})")
                print(f"  End:   ({last_pos['x']:.2f}, {last_pos['y']:.2f})")
        
        return success, best_trajectory, best_action_log, best_waypoint_pose
    
    def _train_step(self, transitions: list, batch_size: int = 32,
                   discount: float = 0.99, tau: float = 0.005,
                   update_actor: bool = True):
        """
        Single training step with optional actor update
        
        Args:
            transitions: Replay buffer transitions
            batch_size: Batch size
            discount: Discount factor
            tau: Target network update rate
            update_actor: Whether to update the actor (False during warmup)
        """
        if len(transitions) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(transitions), batch_size, replace=False)
        batch = [transitions[i] for i in indices]
        
        # Convert to numpy arrays first to avoid slow tensor creation
        states_np = np.array([t[0] for t in batch], dtype=np.float32)
        actions_np = np.array([t[1] for t in batch], dtype=np.float32)
        rewards_np = np.array([[t[2]] for t in batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in batch], dtype=np.float32)
        dones_np = np.array([[t[4]] for t in batch], dtype=np.float32)
        
        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
        dones = torch.tensor(dones_np, dtype=torch.float32, device=device)
        
        # Compute embeddings (freeze GAT during critic training)
        z_list = []
        z_next_list = []
        with torch.no_grad():  # Don't update GAT during embedding computation
            for i in range(batch_size):
                lidar, robot, human = self._flat_to_nodes(states[i].cpu().numpy())
                z = self.recovery_gat(lidar, robot, human, self.edge_index).view(1, -1)
                z_list.append(z)
                
                lidar_n, robot_n, human_n = self._flat_to_nodes(next_states[i].cpu().numpy())
                z_n = self.recovery_gat(lidar_n, robot_n, human_n, self.edge_index).view(1, -1)
                z_next_list.append(z_n)
        
        z = torch.cat(z_list, dim=0)
        z_next = torch.cat(z_next_list, dim=0)
        
        # Critic update only (don't touch GAT during recovery to preserve representations)
        with torch.no_grad():
            next_actions = self.recovery_actor(z_next)
            # Add noise for TD3 policy smoothing
            noise = torch.randn_like(next_actions) * 0.1
            noise = noise.clamp(-0.2, 0.2)
            next_actions = (next_actions + noise).clamp(-1, 1)
            
            target_q1, target_q2 = self.recovery_critic(z_next, next_actions)
            target_q = rewards + (1 - dones) * discount * torch.min(target_q1, target_q2)
        
        current_q1, current_q2 = self.recovery_critic(z, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor update (delayed, only if warmup is complete)
        if update_actor:
            # Re-compute embeddings for actor update (with gradient)
            z_actor_list = []
            for i in range(batch_size):
                lidar, robot, human = self._flat_to_nodes(states[i].cpu().numpy())
                z_a = self.recovery_gat(lidar, robot, human, self.edge_index).view(1, -1)
                z_actor_list.append(z_a)
            z_actor = torch.cat(z_actor_list, dim=0)
            
            new_actions = self.recovery_actor(z_actor)
            actor_loss = -self.recovery_critic.Q1(torch.cat([z_actor, new_actions], dim=1)).mean()
            
            self.actor_opt.zero_grad()
            self.gat_opt.zero_grad()  # Allow very small GAT updates with actor
            actor_loss.backward()
            
            # Gradient clipping to prevent large changes
            torch.nn.utils.clip_grad_norm_(self.recovery_actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.recovery_gat.parameters(), 0.5)
            
            self.actor_opt.step()
            self.gat_opt.step()
    
    def get_light_update_weights(self, blend_factor: float = 0.05) -> Tuple[dict, dict]:
        """
        Get weights for light update to original model
        
        Args:
            blend_factor: How much of recovery model to blend (0.05 = 5%)
        
        Returns:
            gat_weights, actor_weights: Blended weights
        """
        gat_weights = {}
        actor_weights = {}
        
        # Blend GAT weights
        for key in self.original_gat.state_dict().keys():
            orig = self.original_gat.state_dict()[key]
            recov = self.recovery_gat.state_dict()[key]
            gat_weights[key] = (1 - blend_factor) * orig + blend_factor * recov
        
        # Blend Actor weights
        for key in self.original_actor.state_dict().keys():
            orig = self.original_actor.state_dict()[key]
            recov = self.recovery_actor.state_dict()[key]
            actor_weights[key] = (1 - blend_factor) * orig + blend_factor * recov
        
        return gat_weights, actor_weights


# ==================== NAVIGATION QUALITY TRACKER ====================
class NavigationQualityTracker:
    """
    Tracks navigation quality metrics for quality-gated continuous updates
    Only allows updates when navigation is demonstrably successful
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 min_progress_rate: float = 0.02,  # meters per step minimum
                 max_collision_rate: float = 0.0,
                 min_velocity: float = 0.1,
                 smoothness_threshold: float = 0.5):
        
        self.window_size = window_size
        self.min_progress_rate = min_progress_rate
        self.max_collision_rate = max_collision_rate
        self.min_velocity = min_velocity
        self.smoothness_threshold = smoothness_threshold
        
        # Tracking buffers
        self.distances_to_goal = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        self.angular_velocities = deque(maxlen=window_size)
        self.emergency_stops = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Quality metrics
        self.current_quality_score = 0.0
        self.update_allowed = False
        self.consecutive_good_steps = 0
        self.min_good_steps_for_update = 20
        
    def update(self, goal_dist: float, vx: float, wz: float, 
               emergency_stop: bool = False) -> Dict:
        """
        Update tracker with current step data
        Returns quality metrics
        """
        current_time = time.time()
        self.distances_to_goal.append(goal_dist)
        self.velocities.append(abs(vx))
        self.angular_velocities.append(abs(wz))
        self.emergency_stops.append(1.0 if emergency_stop else 0.0)
        self.timestamps.append(current_time)
        
        # Calculate quality metrics
        metrics = self._compute_metrics()
        
        # Update quality score (0-1)
        self.current_quality_score = self._compute_quality_score(metrics)
        
        # Determine if update is allowed
        if self.current_quality_score > 0.7:
            self.consecutive_good_steps += 1
        else:
            self.consecutive_good_steps = max(0, self.consecutive_good_steps - 2)
        
        self.update_allowed = (
            self.consecutive_good_steps >= self.min_good_steps_for_update and
            self.current_quality_score > 0.6
        )
        
        metrics['quality_score'] = self.current_quality_score
        metrics['update_allowed'] = self.update_allowed
        metrics['consecutive_good'] = self.consecutive_good_steps
        
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """Compute quality metrics from buffers"""
        if len(self.distances_to_goal) < 5:
            return {
                'progress_rate': 0.0,
                'avg_velocity': 0.0,
                'velocity_variance': 1.0,
                'collision_rate': 1.0,
                'smoothness': 0.0
            }
        
        # Progress rate (distance reduction per step)
        distances = list(self.distances_to_goal)
        if len(distances) >= 10:
            progress_rate = (distances[0] - distances[-1]) / len(distances)
        else:
            progress_rate = (distances[0] - distances[-1]) / max(1, len(distances))
        
        # Average velocity
        avg_velocity = np.mean(list(self.velocities))
        
        # Velocity variance (lower = smoother)
        velocity_variance = np.var(list(self.velocities)) if len(self.velocities) > 1 else 0.0
        
        # Angular velocity variance (smoothness measure)
        angular_variance = np.var(list(self.angular_velocities)) if len(self.angular_velocities) > 1 else 0.0
        
        # Collision/emergency stop rate
        collision_rate = np.mean(list(self.emergency_stops))
        
        # Smoothness (inverse of angular variance)
        smoothness = 1.0 / (1.0 + angular_variance)
        
        return {
            'progress_rate': progress_rate,
            'avg_velocity': avg_velocity,
            'velocity_variance': velocity_variance,
            'collision_rate': collision_rate,
            'smoothness': smoothness
        }
    
    def _compute_quality_score(self, metrics: Dict) -> float:
        """Compute overall quality score (0-1)"""
        score = 0.0
        
        # Progress component (40%)
        if metrics['progress_rate'] > self.min_progress_rate:
            score += 0.4
        elif metrics['progress_rate'] > 0:
            score += 0.4 * (metrics['progress_rate'] / self.min_progress_rate)
        
        # Velocity component (20%)
        if metrics['avg_velocity'] > self.min_velocity:
            score += 0.2
        elif metrics['avg_velocity'] > 0:
            score += 0.2 * (metrics['avg_velocity'] / self.min_velocity)
        
        # No collision component (25%)
        if metrics['collision_rate'] <= self.max_collision_rate:
            score += 0.25
        else:
            score += 0.25 * max(0, 1 - metrics['collision_rate'])
        
        # Smoothness component (15%)
        score += 0.15 * min(1.0, metrics['smoothness'] / self.smoothness_threshold)
        
        return min(1.0, score)
    
    def reset(self):
        """Reset all tracking"""
        self.distances_to_goal.clear()
        self.velocities.clear()
        self.angular_velocities.clear()
        self.emergency_stops.clear()
        self.timestamps.clear()
        self.current_quality_score = 0.0
        self.update_allowed = False
        self.consecutive_good_steps = 0
    
    def is_update_allowed(self) -> bool:
        """Check if quality gate allows update"""
        return self.update_allowed


# ==================== PLANNING ROLLOUT ====================
class PlanningRollout:
    """
    Runs policy ahead without training - just prediction
    Used for continuous path visualization
    """
    
    def __init__(self, gat: nn.Module, actor: nn.Module, 
                 edge_index: torch.Tensor,
                 lidar_nodes: int = LIDAR_BEAMS,
                 max_humans: int = MAX_HUMANS):
        self.gat = gat
        self.actor = actor
        self.edge_index = edge_index
        self.L = lidar_nodes
        self.H = max_humans
        
    def _flat_to_nodes(self, flat_state: np.ndarray):
        """Convert flat state to node tensors"""
        lidar_size = self.L * 4
        robot_size = 1 * 4
        human_size = self.H * 6
        
        lidar = torch.tensor(flat_state[:lidar_size], 
                            dtype=torch.float32, device=device).view(self.L, 4)
        robot = torch.tensor(flat_state[lidar_size:lidar_size + robot_size],
                            dtype=torch.float32, device=device).view(1, 4)
        human = torch.tensor(flat_state[lidar_size + robot_size:lidar_size + robot_size + human_size],
                            dtype=torch.float32, device=device).view(self.H, 6)
        
        return lidar, robot, human
    
    @torch.no_grad()
    def get_action(self, flat_state: np.ndarray) -> np.ndarray:
        """Get action from policy (no exploration)"""
        lidar, robot, human = self._flat_to_nodes(flat_state)
        z = self.gat(lidar, robot, human, self.edge_index).view(1, -1)
        action = self.actor(z).cpu().numpy().flatten()
        return action
    
    def plan_ahead(self, snapshot: 'EnvironmentSnapshot', 
                   steps: int = 100,
                   early_stop_at_goal: bool = True) -> Tuple[List[Tuple[float, float]], bool, float]:
        """
        Run policy ahead to predict path
        
        Args:
            snapshot: Current environment snapshot
            steps: Maximum steps to simulate
            early_stop_at_goal: Stop if goal is reached
            
        Returns:
            trajectory: List of (x, y) positions
            reached_goal: Whether simulation reached goal
            final_distance: Distance to goal at end
        """
        # Create simulation environment from snapshot
        env = RecoverySimEnv(snapshot, self.L, self.H)
        state = env.reset()
        
        trajectory = [(env.robot_x, env.robot_y)]
        reached_goal = False
        
        for step in range(steps):
            # Get action from policy
            action = self.get_action(state)
            
            # Convert to env format (linear: 0-1, angular: -1 to 1)
            action_env = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float32)
            
            # Step environment
            state, reward, done, target, collision = env.step(action_env)
            
            # Record position
            trajectory.append((env.robot_x, env.robot_y))
            
            # Check termination
            if target and early_stop_at_goal:
                reached_goal = True
                break
            
            if collision:
                break
            
            if done:
                break
        
        final_distance = env._dist_to_goal()
        
        return trajectory, reached_goal, final_distance
    
    def plan_multiple_horizons(self, snapshot: 'EnvironmentSnapshot',
                               horizons: List[int] = [25, 50, 100]) -> Dict:
        """
        Plan at multiple time horizons for richer visualization
        
        Returns dict with trajectories at different horizons
        """
        results = {}
        
        for horizon in horizons:
            traj, reached, dist = self.plan_ahead(snapshot, steps=horizon)
            results[horizon] = {
                'trajectory': traj,
                'reached_goal': reached,
                'final_distance': dist
            }
        
        return results


# ==================== CONTINUOUS LEARNER ====================
class ContinuousLearner:
    """
    Manages continuous light updates during successful navigation
    Only updates when quality metrics are good
    """
    
    def __init__(self, 
                 gat: nn.Module, 
                 actor: nn.Module,
                 edge_index: torch.Tensor,
                 micro_blend_factor: float = 0.005,  # 0.5% for micro updates
                 buffer_size: int = 500,
                 min_buffer_for_update: int = 100,
                 update_interval: int = 50):
        
        self.gat = gat
        self.actor = actor
        self.edge_index = edge_index
        self.micro_blend_factor = micro_blend_factor
        self.buffer_size = buffer_size
        self.min_buffer_for_update = min_buffer_for_update
        self.update_interval = update_interval
        
        # Experience buffer (only good experiences)
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Model parameters
        self.L = LIDAR_BEAMS
        self.H = MAX_HUMANS
        self.total_nodes = self.L + 1 + self.H
        self.gat_out = 16
        
        # Shadow models for learning (clones of main models)
        self.shadow_gat = None
        self.shadow_actor = None
        self.shadow_critic = None
        
        # Optimizers
        self.gat_opt = None
        self.actor_opt = None
        self.critic_opt = None
        
        # Statistics
        self.total_updates = 0
        self.steps_since_update = 0
        self.update_history = []
        
        # Quality tracker
        self.quality_tracker = NavigationQualityTracker()
        
        self._initialized = False
    
    def initialize_shadow_models(self):
        """Initialize shadow models for learning"""
        if self._initialized:
            return
        
        # Create shadow copies
        self.shadow_gat = GATNetwork(
            lidar_channels=4, robot_channels=4, human_channels=6,
            out_channels=self.gat_out, num_heads=4,
            lidar_nodes=self.L, robot_nodes=1, human_nodes=self.H
        ).to(device)
        
        state_dim = self.gat_out * self.total_nodes
        self.shadow_actor = Actor(state_dim, action_dim=2).to(device)
        self.shadow_critic = Critic(state_dim, action_dim=2).to(device)
        
        # Copy weights from main models
        self.shadow_gat.load_state_dict(self.gat.state_dict())
        self.shadow_actor.load_state_dict(self.actor.state_dict())
        
        # Set to training mode
        self.shadow_gat.train()
        self.shadow_actor.train()
        self.shadow_critic.train()
        
        # Create optimizers with small learning rate
        self.gat_opt = torch.optim.Adam(self.shadow_gat.parameters(), lr=1e-5)
        self.actor_opt = torch.optim.Adam(self.shadow_actor.parameters(), lr=5e-5)
        self.critic_opt = torch.optim.Adam(self.shadow_critic.parameters(), lr=5e-5)
        
        self._initialized = True
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, 
                       reward: float, next_state: np.ndarray, done: bool,
                       goal_dist: float, vx: float, wz: float,
                       emergency_stop: bool = False) -> Dict:
        """
        Add experience and update quality tracking
        Returns quality metrics
        """
        # Update quality tracker
        metrics = self.quality_tracker.update(goal_dist, vx, wz, emergency_stop)
        
        # Only add good experiences
        if metrics['quality_score'] > 0.5 and not emergency_stop:
            self.experience_buffer.append({
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_state': next_state.copy(),
                'done': float(done)
            })
        
        self.steps_since_update += 1
        
        # Check if we should do an update
        if (self.steps_since_update >= self.update_interval and
            self.quality_tracker.is_update_allowed() and
            len(self.experience_buffer) >= self.min_buffer_for_update):
            
            self._do_micro_update()
            self.steps_since_update = 0
        
        return metrics
    
    def _flat_to_nodes(self, flat_state: np.ndarray):
        """Convert flat state to node tensors"""
        lidar_size = self.L * 4
        robot_size = 1 * 4
        human_size = self.H * 6
        
        lidar = torch.tensor(flat_state[:lidar_size], 
                            dtype=torch.float32, device=device).view(self.L, 4)
        robot = torch.tensor(flat_state[lidar_size:lidar_size + robot_size],
                            dtype=torch.float32, device=device).view(1, 4)
        human = torch.tensor(flat_state[lidar_size + robot_size:lidar_size + robot_size + human_size],
                            dtype=torch.float32, device=device).view(self.H, 6)
        
        return lidar, robot, human
    
    def _do_micro_update(self, batch_size: int = 32, n_iterations: int = 3):
        """Perform a small training update on shadow models"""
        if not self._initialized:
            self.initialize_shadow_models()
        
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sync shadow models with main models before update
        self.shadow_gat.load_state_dict(self.gat.state_dict())
        self.shadow_actor.load_state_dict(self.actor.state_dict())
        
        for _ in range(n_iterations):
            # Sample batch
            indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
            batch = [self.experience_buffer[i] for i in indices]
            
            # Convert to numpy arrays first to avoid slow tensor creation warning
            states_np = np.array([t['state'] for t in batch], dtype=np.float32)
            actions_np = np.array([t['action'] for t in batch], dtype=np.float32)
            rewards_np = np.array([[t['reward']] for t in batch], dtype=np.float32)
            next_states_np = np.array([t['next_state'] for t in batch], dtype=np.float32)
            dones_np = np.array([[t['done']] for t in batch], dtype=np.float32)
            
            states = torch.tensor(states_np, dtype=torch.float32, device=device)
            actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
            dones = torch.tensor(dones_np, dtype=torch.float32, device=device)
            
            # Compute embeddings
            z_list = []
            z_next_list = []
            for i in range(batch_size):
                lidar, robot, human = self._flat_to_nodes(states[i].cpu().numpy())
                z = self.shadow_gat(lidar, robot, human, self.edge_index).view(1, -1)
                z_list.append(z)
                
                lidar_n, robot_n, human_n = self._flat_to_nodes(next_states[i].cpu().numpy())
                z_n = self.shadow_gat(lidar_n, robot_n, human_n, self.edge_index).view(1, -1)
                z_next_list.append(z_n)
            
            z = torch.cat(z_list, dim=0)
            z_next = torch.cat(z_next_list, dim=0)
            
            # Critic update
            with torch.no_grad():
                next_actions = self.shadow_actor(z_next)
                target_q1, target_q2 = self.shadow_critic(z_next, next_actions)
                target_q = rewards + (1 - dones) * 0.99 * torch.min(target_q1, target_q2)
            
            current_q1, current_q2 = self.shadow_critic(z, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.critic_opt.zero_grad()
            self.gat_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            self.gat_opt.step()
            
            # Actor update
            new_actions = self.shadow_actor(z.detach())
            actor_loss = -self.shadow_critic.Q1(torch.cat([z.detach(), new_actions], dim=1)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        
        # Apply micro blend to main models
        self._blend_to_main()
        
        self.total_updates += 1
        self.update_history.append({
            'timestamp': time.time(),
            'quality_score': self.quality_tracker.current_quality_score,
            'buffer_size': len(self.experience_buffer)
        })
    
    def _blend_to_main(self):
        """Blend shadow model weights into main models"""
        # Blend GAT
        main_state = self.gat.state_dict()
        shadow_state = self.shadow_gat.state_dict()
        for key in main_state.keys():
            main_state[key] = (1 - self.micro_blend_factor) * main_state[key] + \
                             self.micro_blend_factor * shadow_state[key]
        self.gat.load_state_dict(main_state)
        
        # Blend Actor
        main_state = self.actor.state_dict()
        shadow_state = self.shadow_actor.state_dict()
        for key in main_state.keys():
            main_state[key] = (1 - self.micro_blend_factor) * main_state[key] + \
                             self.micro_blend_factor * shadow_state[key]
        self.actor.load_state_dict(main_state)
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'total_updates': self.total_updates,
            'buffer_size': len(self.experience_buffer),
            'quality_score': self.quality_tracker.current_quality_score,
            'update_allowed': self.quality_tracker.update_allowed,
            'consecutive_good': self.quality_tracker.consecutive_good_steps
        }
    
    def reset(self):
        """Reset learner state"""
        self.experience_buffer.clear()
        self.quality_tracker.reset()
        self.steps_since_update = 0


# ==================== STUCK RECOVERY CONTROLLER ====================
class StuckRecoveryController:
    """
    Main controller that integrates stuck detection, snapshot capture,
    quick retraining, recovery deployment, continuous planning, and light updates.
    
    Features:
    - Stuck detection and recovery training
    - Continuous lookahead planning with path visualization
    - Quality-gated continuous light updates
    """
    
    def __init__(self, gat_path: str, actor_path: str,
                 light_update_blend: float = 0.03,
                 enable_continuous_planning: bool = True,
                 enable_continuous_learning: bool = True,
                 planning_interval: float = 0.5,
                 planning_horizon: int = 100,
                 visualize_recovery: bool = False):
        
        self.gat_path = gat_path
        self.actor_path = actor_path
        self.light_update_blend = light_update_blend
        
        # Feature flags
        self.enable_continuous_planning = enable_continuous_planning
        self.enable_continuous_learning = enable_continuous_learning
        self.planning_interval = planning_interval
        self.planning_horizon = planning_horizon
        self.visualize_recovery = visualize_recovery  # Open pygame window during recovery training
        
        # Components
        self.stuck_detector = StuckDetector()
        self.snapshot_builder = EnvironmentSnapshotBuilder()
        self.retrainer = QuickRetrainer(gat_path, actor_path)
        
        # Planning components (initialized later)
        self.planner = None
        self.continuous_learner = None
        
        # State
        self.recovery_mode = False
        self.current_snapshot = None
        self.recovery_trajectory = None
        self.recovery_waypoint_idx = 0
        
        # === WAYPOINT BLEND STATE ===
        # Logged sim actions blended with live DRL until the real robot
        # reaches the digital-twin waypoint. No open-loop drive past it.
        self.action_replay_log = []      # List of {action, x, y, theta, step}
        self.action_replay_idx = 0       # Current index in the log
        self.action_replay_active = False
        self.blend_steps = 0
        self.max_blend_steps = 0
        self.blend_debug = None
        self.waypoint_chain = []
        self.chain_idx = 0
        self.last_attempt = None
        self.waypoint_x = None
        self.waypoint_y = None
        self.waypoint_theta = None       # Log heading when the twin set off
        
        # === HEADING ALIGNMENT ===
        # Before the blend, yaw in place to that logged set-off heading
        self.positioning_active = False
        self.positioning_phase = None    # 'turn'
        self._align_start_time = 0.0
        
        # Models
        self.gat = None
        self.actor = None
        self.edge_index = None
        
        # Planning state (thread-safe)
        self._planned_path = []
        self._planned_path_lock = threading.Lock()
        self._last_planning_time = 0.0
        self._planning_thread = None
        self._planning_running = False
        
        # Recovery cooldown (to prevent rapid re-triggering after failed recovery)
        self._recovery_cooldown_until = 0.0
        
        # Stats
        self.total_recoveries = 0
        self.successful_recoveries = 0
        self.planning_stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'avg_plan_time': 0.0
        }
        
    def initialize(self):
        """Initialize models and components"""
        self.retrainer.load_models()
        
        # Use original models for normal operation
        self.gat = self.retrainer.original_gat
        self.actor = self.retrainer.original_actor
        self.edge_index = self.retrainer.edge_index
        
        # Initialize planning rollout
        if self.enable_continuous_planning:
            self.planner = PlanningRollout(
                self.gat, self.actor, self.edge_index,
                lidar_nodes=LIDAR_BEAMS, max_humans=MAX_HUMANS
            )
            print("[StuckRecoveryController] Continuous planning enabled")
        
        # Initialize continuous learner
        if self.enable_continuous_learning:
            self.continuous_learner = ContinuousLearner(
                self.gat, self.actor, self.edge_index,
                micro_blend_factor=0.005  # 0.5% micro updates
            )
            print("[StuckRecoveryController] Continuous learning enabled")
        
        print("[StuckRecoveryController] Initialized")
        
    def update(self, robot_x: float, robot_y: float, robot_theta: float,
               vx: float, wz: float, goal_x: float, goal_y: float,
               lidar_ranges: np.ndarray, 
               raw_scan_ranges: Optional[np.ndarray] = None,
               detected_persons: Optional[List[Dict]] = None,
               current_action: Optional[np.ndarray] = None,
               reward: float = 0.0,
               emergency_stop: bool = False) -> Tuple[bool, str]:
        """
        Update controller with current state
        
        Args:
            robot_x, robot_y, robot_theta: Robot pose
            vx, wz: Robot velocities
            goal_x, goal_y: Goal position
            lidar_ranges: Downsampled LiDAR
            raw_scan_ranges: Full 360° scan (optional)
            detected_persons: Detected humans
            current_action: Current action being taken (for learning)
            reward: Current reward (for learning)
            emergency_stop: Whether emergency stop is active
        
        Returns:
            stuck_detected: True if stuck was just detected
            status: Current status message
        """
        goal_dist = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        
        # Update continuous learner if enabled
        if self.enable_continuous_learning and self.continuous_learner is not None:
            if current_action is not None and hasattr(self, '_last_state'):
                # Build current state
                current_state = self._build_flat_state(
                    lidar_ranges, robot_x, robot_y, robot_theta,
                    goal_x, goal_y, vx, wz, detected_persons
                )
                
                # Add experience
                self.continuous_learner.add_experience(
                    self._last_state, current_action, reward,
                    current_state, False, goal_dist, vx, wz, emergency_stop
                )
                
                self._last_state = current_state
            else:
                # Initialize state tracking
                self._last_state = self._build_flat_state(
                    lidar_ranges, robot_x, robot_y, robot_theta,
                    goal_x, goal_y, vx, wz, detected_persons
                )
        
        # Run continuous planning if enabled (and not in recovery mode)
        if self.enable_continuous_planning and not self.recovery_mode:
            current_time = time.time()
            if current_time - self._last_planning_time >= self.planning_interval:
                self._run_planning(
                    robot_x, robot_y, robot_theta,
                    goal_x, goal_y, lidar_ranges,
                    raw_scan_ranges, detected_persons
                )
                self._last_planning_time = current_time
        
        if self.recovery_mode:
            return False, "RECOVERY_MODE"
        
        # Check recovery cooldown (prevent rapid re-triggering after failed recovery)
        current_time = time.time()
        if current_time < self._recovery_cooldown_until:
            return False, "RECOVERY_COOLDOWN"
        
        # Check for stuck condition
        is_stuck = self.stuck_detector.update(
            robot_x, robot_y, robot_theta, vx, wz, goal_x, goal_y
        )
        
        if is_stuck:
            print("\n" + "="*60)
            print("[STUCK DETECTED] Initiating recovery procedure...")
            print("="*60)
            
            # Capture snapshot
            self.current_snapshot = self.snapshot_builder.capture(
                robot_x, robot_y, robot_theta,
                goal_x, goal_y,
                lidar_ranges,
                raw_scan_ranges,
                detected_persons
            )
            
            print(f"[Snapshot] Robot: ({robot_x:.2f}, {robot_y:.2f})")
            print(f"[Snapshot] Goal: ({goal_x:.2f}, {goal_y:.2f})")
            print(f"[Snapshot] Blocked regions: {len(self.current_snapshot.blocked_regions)}")
            print(f"[Snapshot] Boundary points: {len(self.current_snapshot.lidar_points_world)}")
            
            return True, "STUCK_DETECTED"
        
        return False, "NORMAL"

    def refresh_snapshot(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
                         lidar_ranges, raw_scan_ranges=None, detected_persons=None):
        """Recapture the world after the robot has turned to face the goal."""
        self.current_snapshot = self.snapshot_builder.capture(
            robot_x, robot_y, robot_theta,
            goal_x, goal_y,
            lidar_ranges,
            raw_scan_ranges,
            detected_persons
        )
        print(f"[Snapshot] Facing goal at heading {math.degrees(robot_theta):.1f}°")
        print(f"[Snapshot] LiDAR points: {len(self.current_snapshot.lidar_points_world)}")
    
    def _build_flat_state(self, lidar_ranges: np.ndarray,
                          robot_x: float, robot_y: float, robot_theta: float,
                          goal_x: float, goal_y: float,
                          vx: float, wz: float,
                          detected_persons: Optional[List[Dict]]) -> np.ndarray:
        """Build flat state vector for learning"""
        # Expect the 20 beams already sampled at BEAM_ANGLES (deploy downsample).
        lidar = np.asarray(lidar_ranges, dtype=np.float32)
        if len(lidar) != LIDAR_BEAMS:
            fixed = np.full(LIDAR_BEAMS, LIDAR_RANGE, dtype=np.float32)
            n = min(len(lidar), LIDAR_BEAMS)
            fixed[:n] = lidar[:n]
            lidar = fixed
        
        # Clean up
        lidar[np.isnan(lidar)] = LIDAR_RANGE
        lidar[np.isinf(lidar)] = LIDAR_RANGE
        lidar = np.clip(lidar, 0, LIDAR_RANGE)
        
        # LiDAR features — same angles as pygame_env_rosv5 / deploy_v5
        angles = BEAM_ANGLES
        lidar_feats = np.zeros((LIDAR_BEAMS, 4), dtype=np.float32)
        for i, (dist, angle) in enumerate(zip(lidar, angles)):
            lidar_feats[i] = [
                dist / LIDAR_RANGE,
                np.cos(angle),
                np.sin(angle),
                0.0
            ]
        
        # Robot features
        dx_world = goal_x - robot_x
        dy_world = goal_y - robot_y
        dx_ego = dx_world * math.cos(robot_theta) + dy_world * math.sin(robot_theta)
        dy_ego = -dx_world * math.sin(robot_theta) + dy_world * math.cos(robot_theta)
        
        robot_feats = np.array([[
            dx_ego / 20.0,
            dy_ego / 20.0,
            vx,
            wz
        ]], dtype=np.float32)
        
        # Human features
        human_feats = np.zeros((MAX_HUMANS, 6), dtype=np.float32)
        if detected_persons:
            for i, person in enumerate(detected_persons[:MAX_HUMANS]):
                human_feats[i] = [
                    person.get('x_rel', 0) / 15.0,
                    person.get('y_rel', 0) / 15.0,
                    0.5,  # Default speed
                    person.get('confidence', 0.9),
                    person.get('x_rel', 0) / 15.0,
                    person.get('y_rel', 0) / 15.0
                ]
        
        return np.concatenate([
            lidar_feats.flatten(),
            robot_feats.flatten(),
            human_feats.flatten()
        ])
    
    def _run_planning_async(self, robot_x: float, robot_y: float, robot_theta: float,
                            goal_x: float, goal_y: float,
                            lidar_ranges: np.ndarray,
                            raw_scan_ranges: Optional[np.ndarray],
                            detected_persons: Optional[List[Dict]]):
        """Run planning in background thread (non-blocking)"""
        # Don't start new planning if one is already running
        if self._planning_running:
            return
        
        # Copy data for thread safety
        lidar_copy = lidar_ranges.copy() if lidar_ranges is not None else None
        raw_scan_copy = raw_scan_ranges.copy() if raw_scan_ranges is not None else None
        persons_copy = copy.deepcopy(detected_persons) if detected_persons else None
        
        def planning_worker():
            self._planning_running = True
            try:
                self._do_planning(
                    robot_x, robot_y, robot_theta,
                    goal_x, goal_y,
                    lidar_copy, raw_scan_copy, persons_copy
                )
            finally:
                self._planning_running = False
        
        # Start background thread
        thread = threading.Thread(target=planning_worker, daemon=True)
        thread.start()
    
    def _do_planning(self, robot_x: float, robot_y: float, robot_theta: float,
                     goal_x: float, goal_y: float,
                     lidar_ranges: np.ndarray,
                     raw_scan_ranges: Optional[np.ndarray],
                     detected_persons: Optional[List[Dict]]):
        """Actually perform the planning (runs in background thread)"""
        if self.planner is None:
            return
        
        start_time = time.time()
        
        try:
            # Capture current snapshot for planning
            snapshot = self.snapshot_builder.capture(
                robot_x, robot_y, robot_theta,
                goal_x, goal_y,
                lidar_ranges,
                raw_scan_ranges,
                detected_persons
            )
            
            # Run planning rollout
            trajectory, reached_goal, final_dist = self.planner.plan_ahead(
                snapshot, steps=self.planning_horizon
            )
            
            # Update stats
            plan_time = time.time() - start_time
            self.planning_stats['total_plans'] += 1
            if reached_goal:
                self.planning_stats['successful_plans'] += 1
            
            # Running average of plan time
            n = self.planning_stats['total_plans']
            self.planning_stats['avg_plan_time'] = (
                (self.planning_stats['avg_plan_time'] * (n-1) + plan_time) / n
            )
            
            # Thread-safe update of planned path
            with self._planned_path_lock:
                self._planned_path = trajectory
                
        except Exception as e:
            # Silently handle planning errors in background
            pass
    
    def _run_planning(self, robot_x: float, robot_y: float, robot_theta: float,
                      goal_x: float, goal_y: float,
                      lidar_ranges: np.ndarray,
                      raw_scan_ranges: Optional[np.ndarray],
                      detected_persons: Optional[List[Dict]]):
        """Run planning rollout (now async by default)"""
        self._run_planning_async(
            robot_x, robot_y, robot_theta,
            goal_x, goal_y,
            lidar_ranges, raw_scan_ranges, detected_persons
        )
    
    def start_recovery(self, verbose: bool = True, visualize: bool = False) -> bool:
        """
        Start recovery procedure
        
        Args:
            verbose: Print progress
            visualize: Open pygame window showing recovery training
        
        Returns:
            success: Whether a recovery path was found
        """
        if self.current_snapshot is None:
            print("[ERROR] No snapshot available")
            return False
        
        self.total_recoveries += 1
        
        # Clone models for recovery
        self.retrainer.clone_for_recovery()
        
        # Train on snapshot (with optional visualization)
        success, trajectory, action_log, waypoint_pose = self.retrainer.train_on_snapshot(
            self.current_snapshot, 
            verbose=verbose,
            visualize=visualize
        )
        
        if success:
            self.recovery_mode = True
            self.recovery_trajectory = trajectory
            self.recovery_waypoint_idx = 0
            self.successful_recoveries += 1
            
            # Recovery policy is the DRL half of the blend. It sees the live
            # observation, including humans, so the blend can yield to them.
            self.gat = self.retrainer.recovery_gat
            self.actor = self.retrainer.recovery_actor
            
            self.action_replay_log = action_log
            self.action_replay_idx = 0
            self.blend_steps = 0
            self.max_blend_steps = max(len(action_log) * BLEND_STEP_FACTOR, 80)
            self.waypoint_chain = [
                (float(a), float(b)) for a, b in getattr(self.retrainer, 'waypoint_chain', [])
            ]
            self.chain_idx = 0
            if self.waypoint_chain:
                self.waypoint_x, self.waypoint_y = self.waypoint_chain[0]
            elif waypoint_pose is not None:
                self.waypoint_x, self.waypoint_y = waypoint_pose[0], waypoint_pose[1]
            else:
                self.waypoint_x = self.waypoint_y = None
            self.waypoint_theta = None
            self.positioning_active = False
            self.action_replay_active = len(self.waypoint_chain) > 0
            if self.action_replay_active:
                print("\n" + "="*60)
                print("[Recovery] GO-TO-GOAL CHAIN")
                print("="*60)
                for i, (wx, wy) in enumerate(self.waypoint_chain):
                    print(f"  {i}: ({wx:.2f}, {wy:.2f})")
                print("  Driving the chain. The policy is not used on this stretch.")
                print("="*60)
            else:
                print("\n[Recovery] WARNING: empty waypoint chain")
                self._handoff_full_drl("Switching straight to full DRL")
            
            print(f"[Recovery] Path found with {len(trajectory)} waypoints")
            self.last_attempt = self._bundle_attempt(
                True, trajectory, action_log, waypoint_pose
            )
            return True
        else:
            print("\n[Recovery] Failed to find path - returning to normal mode")
            self.last_attempt = self._bundle_attempt(
                False, trajectory, action_log, waypoint_pose
            )
            self.reset_recovery(failed=True)
            return False
    
    def _bundle_attempt(self, success, trajectory, action_log, waypoint_pose):
        """JSON-friendly record of the retrain that just finished. Snapshot is copied now."""
        snap = self.current_snapshot
        episodes = []
        for ep in self.retrainer.train_episodes:
            episodes.append(dict(ep))
        actions = []
        for entry in action_log or []:
            act = entry['action']
            actions.append({
                'action': [float(act[0]), float(act[1])],
                'x': float(entry['x']),
                'y': float(entry['y']),
                'theta': float(entry['theta']),
                'step': int(entry['step']),
                'segment': entry.get('segment', ''),
            })
        traj = [[float(p[0]), float(p[1])] for p in (trajectory or [])]
        pose = None
        if waypoint_pose is not None:
            pose = [float(waypoint_pose[0]), float(waypoint_pose[1]), float(waypoint_pose[2])]
        snapshot = None
        if snap is not None:
            snapshot = {
                'timestamp': float(snap.timestamp),
                'robot_x': float(snap.robot_x),
                'robot_y': float(snap.robot_y),
                'robot_theta': float(snap.robot_theta),
                'goal_x': float(snap.goal_x),
                'goal_y': float(snap.goal_y),
                'lidar_ranges': np.asarray(snap.lidar_ranges, dtype=float).tolist(),
                'lidar_points_world': np.asarray(snap.lidar_points_world, dtype=float).tolist(),
                'boundary_polygon': np.asarray(snap.boundary_polygon, dtype=float).tolist(),
                'blocked_regions': [
                    [float(b[0]), float(b[1]), float(b[2])] for b in snap.blocked_regions
                ],
                'n_obstacle_segments': int(len(snap.obstacle_segments)),
                'humans': snap.humans,
            }
        full_log = []
        for entry in getattr(self.retrainer, 'goto_full_log', None) or []:
            act = entry['action']
            full_log.append({
                'action': [float(act[0]), float(act[1])],
                'x': float(entry['x']),
                'y': float(entry['y']),
                'theta': float(entry['theta']),
                'step': int(entry['step']),
                'segment': entry.get('segment', ''),
            })
        return {
            'success': bool(success),
            'episodes': episodes,
            'action_log': full_log or actions,
            'waypoint_pose': pose,
            'setoff_theta': None if self.waypoint_theta is None else float(self.waypoint_theta),
            'trajectory': traj,
            'snapshot': snapshot,
            'logged_action_blend': float(LOGGED_ACTION_BLEND),
            'waypoint_reach_dist': float(WAYPOINT_REACH_DIST),
            'heading_match_rad': float(HEADING_MATCH_RAD),
        }
    
    def get_recovery_action(self, flat_state: np.ndarray) -> np.ndarray:
        """Get action during recovery mode"""
        return self.retrainer.get_action(flat_state, explore=False)
    
    def check_recovery_complete(self, robot_x: float, robot_y: float,
                               goal_x: float, goal_y: float) -> bool:
        """Check if recovery is complete (reached goal)"""
        dist = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        
        if dist < 0.75:
            print("\n" + "="*60)
            print("[RECOVERY COMPLETE] Goal reached!")
            print("="*60)
            
            # Apply light update to original model
            self._apply_light_update()
            
            # Reset to normal mode
            self.reset_recovery()
            
            return True
        
        return False
    
    def _apply_light_update(self):
        """Apply light update to original model from recovery experience"""
        print(f"\n[Light Update] Blending {self.light_update_blend*100:.1f}% of recovery weights...")
        
        gat_weights, actor_weights = self.retrainer.get_light_update_weights(
            self.light_update_blend
        )
        
        # Update original models
        self.retrainer.original_gat.load_state_dict(gat_weights)
        self.retrainer.original_actor.load_state_dict(actor_weights)
        
        # Save updated weights
        torch.save(gat_weights, self.gat_path + ".updated")
        torch.save(actor_weights, self.actor_path + ".updated")
        
        print(f"[Light Update] Updated weights saved to {self.gat_path}.updated")
        
    def reset_recovery(self, failed: bool = False):
        """Reset to normal operation mode"""
        self.recovery_mode = False
        self.current_snapshot = None
        self.recovery_trajectory = None
        self.recovery_waypoint_idx = 0
        
        # Reset waypoint blend state
        self.action_replay_log = []
        self.action_replay_idx = 0
        self.action_replay_active = False
        self.blend_steps = 0
        self.max_blend_steps = 0
        self.waypoint_x = None
        self.waypoint_y = None
        self.waypoint_theta = None
        self.waypoint_chain = []
        self.chain_idx = 0
        
        # Reset heading alignment
        self.positioning_active = False
        self.positioning_phase = None
        self._align_start_time = 0.0
        
        # Switch back to original models
        self.gat = self.retrainer.original_gat
        self.actor = self.retrainer.original_actor
        
        # Reset stuck detector
        self.stuck_detector.reset()
        
        # If recovery failed, add a cooldown to prevent immediate re-triggering
        if failed:
            self._recovery_cooldown_until = time.time() + 15.0  # 15 second cooldown
            print("[Recovery] Reset to normal mode (15s cooldown)")
        else:
            print("[Recovery] Reset to normal mode")
    
    def get_action(self, flat_state: np.ndarray) -> np.ndarray:
        """Get action (works in both normal and recovery mode)"""
        lidar_size = LIDAR_BEAMS * 4
        robot_size = 1 * 4
        human_size = MAX_HUMANS * 6
        
        lidar = torch.tensor(flat_state[:lidar_size],
                            dtype=torch.float32, device=device).view(LIDAR_BEAMS, 4)
        robot = torch.tensor(flat_state[lidar_size:lidar_size + robot_size],
                            dtype=torch.float32, device=device).view(1, 4)
        human = torch.tensor(flat_state[lidar_size + robot_size:lidar_size + robot_size + human_size],
                            dtype=torch.float32, device=device).view(MAX_HUMANS, 6)
        
        with torch.no_grad():
            z = self.gat(lidar, robot, human, self.edge_index).view(1, -1)
            action = self.actor(z).cpu().numpy().flatten()
        
        return action
    
    def _begin_setoff_yaw(self):
        """Yaw in place to the heading in the log when the twin set off."""
        self.action_replay_active = False
        self.blend_debug = None
        print("\n" + "="*60)
        print("[Recovery] MATCH SET-OFF HEADING")
        print(f"  Log heading: {math.degrees(self.waypoint_theta):.1f}°")
        print(f"  Waypoint: ({self.waypoint_x:.2f}, {self.waypoint_y:.2f})")
        print("  Yaw only, then blend logged actions with live DRL")
        print("="*60)
        self.positioning_active = True
        self.positioning_phase = 'turn'
        self._align_start_time = time.time()
    
    def _start_waypoint_blend(self):
        """Start the logged/DRL blend once the set-off heading matches."""
        self.positioning_active = False
        self.positioning_phase = None
        if not self.action_replay_log or self.waypoint_x is None:
            self._handoff_full_drl("No logged path after align — switching to full DRL")
            return
        self.action_replay_idx = 0
        self.blend_steps = 0
        self.action_replay_active = True
        self.gat = self.retrainer.recovery_gat
        self.actor = self.retrainer.recovery_actor
        print("\n" + "="*60)
        print("[Recovery] WAYPOINT BLEND MODE ACTIVATED")
        print("="*60)
        print(f"  Actions logged: {len(self.action_replay_log)}")
        print(f"  Blend: {LOGGED_ACTION_BLEND*100:.0f}% logged + "
              f"{(1.0 - LOGGED_ACTION_BLEND)*100:.0f}% live DRL")
        print("  Live DRL sees current humans and can pull the command off the log")
        print(f"  Waypoint: ({self.waypoint_x:.2f}, {self.waypoint_y:.2f})")
        first_action = self.action_replay_log[0]['action']
        print(f"  First logged action: linear={first_action[0]:.3f}, angular={first_action[1]:.3f}")
        print("="*60)
    
    def _handoff_full_drl(self, reason: str):
        """Leave recovery maneuvering and resume the original policy."""
        print(f"\n[Recovery] {reason}")
        self.action_replay_active = False
        self.blend_debug = None
        self.positioning_active = False
        self.positioning_phase = None
        self.gat = self.retrainer.original_gat
        self.actor = self.retrainer.original_actor
    
    def get_replay_action(self, robot_x: float, robot_y: float,
                          robot_theta: float = 0.0,
                          drl_action: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        Closed-loop go-to-goal along the short waypoint chain.

        The learned policy is not mixed in here. On the last run it
        commanded full yaw and zero throttle, which spun the robot.
        """
        if not self.action_replay_active or not self.waypoint_chain:
            return None, True

        while self.chain_idx < len(self.waypoint_chain) - 1:
            wx, wy = self.waypoint_chain[self.chain_idx]
            if math.hypot(robot_x - wx, robot_y - wy) < WAYPOINT_REACH_DIST:
                print(f"[Goto] Reached chain point {self.chain_idx} "
                      f"({wx:.2f}, {wy:.2f})")
                self.chain_idx += 1
                self.waypoint_x, self.waypoint_y = self.waypoint_chain[self.chain_idx]
            else:
                break

        wx, wy = self.waypoint_chain[self.chain_idx]
        self.waypoint_x, self.waypoint_y = wx, wy
        dist_wp = math.hypot(robot_x - wx, robot_y - wy)
        # Final point is the goal. Stay on go-to-goal until the outer loop
        # sees the 0.75 m goal threshold. Do not hand back to the policy.
        err = math.atan2(wy - robot_y, wx - robot_x) - robot_theta
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        w = float(np.clip(err / 0.8, -1.0, 1.0))
        if abs(err) > 2.2:
            v = 0.0
        else:
            v = float(np.clip(0.35 + 0.65 * math.cos(err), 0.2, 1.0))
        cmd = np.array([v, w], dtype=np.float32)
        self.blend_debug = {
            'logged_v': v,
            'logged_w': w,
            'drl_a0': None,
            'drl_a1': None,
            'replay_idx': int(self.chain_idx),
        }
        print(f"[GOTO {self.chain_idx}/{len(self.waypoint_chain) - 1}] "
              f"out=[{v:.2f}, {w:.2f}] wp_dist={dist_wp:.2f}m")
        return cmd, False

    def is_replay_active(self) -> bool:
        """Check if the logged/DRL blend toward the waypoint is active"""
        return self.action_replay_active
    
    def is_positioning_active(self) -> bool:
        """Check if yaw-to-log-heading is active (before the blend)"""
        return self.positioning_active
    
    def get_positioning_action(self, robot_x: float, robot_y: float, 
                               robot_theta: float, goal_x: float, goal_y: float,
                               min_obstacle_dist: float) -> Tuple[Optional[np.ndarray], bool]:
        """
        Yaw in place to the logged heading from when the twin set off.
        
        No translation. goal_x/goal_y/min_obstacle_dist are unused; kept so
        the deploy call site stays the same. When the heading matches, the
        logged/DRL blend starts.
        
        Returns:
            action: [linear, angular] in env format (0-1, -1 to 1)
            positioning_done: True when alignment is complete
        """
        if not self.positioning_active:
            return None, True
        
        if self.waypoint_theta is None:
            self._handoff_full_drl("No logged set-off heading — switching to full DRL")
            return None, True
        
        if time.time() - self._align_start_time > ALIGN_TIMEOUT_S:
            print("[Positioning] Heading align timed out — starting the blend anyway")
            self._start_waypoint_blend()
            return None, True
        
        angle_diff = self.waypoint_theta - robot_theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        if abs(angle_diff) < HEADING_MATCH_RAD:
            print("\n" + "="*60)
            print("[Positioning] COMPLETE - Heading matches the log at set-off")
            print("="*60)
            self._start_waypoint_blend()
            return None, True
        
        turn_speed = float(np.clip(angle_diff * 2.0, -1.0, 1.0))
        print(f"[Positioning TURN] angle_diff={math.degrees(angle_diff):.1f}° turn={turn_speed:.2f}")
        return np.array([0.0, turn_speed], dtype=np.float32), False
    
    def get_planned_path(self) -> List[Tuple[float, float]]:
        """
        Get the currently planned path (thread-safe)
        Returns list of (x, y) world coordinates
        """
        with self._planned_path_lock:
            return list(self._planned_path)
    
    def get_planned_path_for_visualization(self) -> Dict:
        """
        Get planned path with metadata for visualization
        """
        with self._planned_path_lock:
            path = list(self._planned_path)
        
        if not path:
            return {
                'path': [],
                'valid': False,
                'length': 0,
                'planning_stats': self.planning_stats.copy()
            }
        
        # Calculate path length
        path_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += math.sqrt(dx**2 + dy**2)
        
        return {
            'path': path,
            'valid': True,
            'length': path_length,
            'num_waypoints': len(path),
            'planning_stats': self.planning_stats.copy()
        }
    
    def force_replan(self, robot_x: float, robot_y: float, robot_theta: float,
                     goal_x: float, goal_y: float,
                     lidar_ranges: np.ndarray,
                     raw_scan_ranges: Optional[np.ndarray] = None,
                     detected_persons: Optional[List[Dict]] = None) -> List[Tuple[float, float]]:
        """
        Force immediate replanning and return path
        Useful for on-demand path visualization
        """
        self._run_planning(
            robot_x, robot_y, robot_theta,
            goal_x, goal_y, lidar_ranges,
            raw_scan_ranges, detected_persons
        )
        return self.get_planned_path()
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            'total_recoveries': self.total_recoveries,
            'successful_recoveries': self.successful_recoveries,
            'recovery_success_rate': self.successful_recoveries / max(1, self.total_recoveries),
            'recovery_mode': self.recovery_mode,
            'planning': self.planning_stats.copy()
        }
        
        # Add continuous learning stats if enabled
        if self.enable_continuous_learning and self.continuous_learner is not None:
            stats['continuous_learning'] = self.continuous_learner.get_stats()
        
        return stats
    
    def save_updated_models(self, suffix: str = "_continuous"):
        """Save models with continuous learning updates"""
        gat_path = self.gat_path.replace('.pth', f'{suffix}.pth')
        actor_path = self.actor_path.replace('.pth', f'{suffix}.pth')
        
        torch.save(self.gat.state_dict(), gat_path)
        torch.save(self.actor.state_dict(), actor_path)
        
        print(f"[Models saved] {gat_path}, {actor_path}")
        
        return gat_path, actor_path


# ==================== MAIN (for testing) ====================
if __name__ == "__main__":
    print("="*60)
    print("Stuck Recovery System with Continuous Planning")
    print("="*60)
    print("\nThis module is designed to be imported by deploy_v5_recovery.py")
    print("\nCore Components:")
    print("  - StuckDetector: Monitors robot state for stuck conditions")
    print("  - EnvironmentSnapshotBuilder: Captures LiDAR environment")
    print("  - RecoverySimEnv: Simulated environment from snapshot")
    print("  - QuickRetrainer: Clones and trains recovery model")
    
    print("\nContinuous Planning Components:")
    print("  - PlanningRollout: Runs policy ahead for path prediction")
    print("  - NavigationQualityTracker: Quality gates for updates")
    print("  - ContinuousLearner: Quality-gated micro-updates")
    
    print("\nMain Controller:")
    print("  - StuckRecoveryController: Integrates all components")
    print("    - Stuck detection and recovery")
    print("    - Continuous lookahead planning")
    print("    - Path visualization support")
    print("    - Quality-gated continuous learning")
    
    print("\nFeature Flags (in StuckRecoveryController):")
    print("  - enable_continuous_planning: Run policy ahead continuously")
    print("  - enable_continuous_learning: Quality-gated micro-updates")
    print("  - planning_interval: Time between planning runs (default: 0.5s)")
    print("  - planning_horizon: Steps to simulate ahead (default: 100)")
    
    print("\nPath Visualization API:")
    print("  - get_planned_path(): Returns List[(x, y)] of planned trajectory")
    print("  - get_planned_path_for_visualization(): Returns dict with metadata")
    print("  - force_replan(): Immediate replanning and path return")
