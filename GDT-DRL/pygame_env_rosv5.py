# pygame_env.py
import math
import random
from collections import deque
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
import subprocess
from sklearn.cluster import KMeans
import pickle
import os


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*convergence.*")
warnings.filterwarnings("ignore", message="Unknown attribute")
# Optional rendering
try:
    import pygame
except Exception:
    pygame = None

try:
    import rospkg
    import yaml
    from urdf_parser_py.urdf import URDF
    
except ImportError:
    
    print("WARNING: Husky packages not available.")

# ----- SIM CONSTANTS -----
GOAL_REACHED_DIST = 0.75
COLLISION_DIST = 0.5
TIME_DELTA = 0.1  # 10 Hz step
LIDAR_RANGE = 10.01 # actual world size is 15.0m, this may need adjustment


class HuskySpecsFromROS:
    """
    Load Clearpath Husky A200 specifications from installed ROS packages.

    Files accessed:
    - $(rospack find husky_control)/config/control.yaml → velocity limits
    - $(rospack find husky_description)/urdf/husky.urdf → physical dimensions
    """

    def __init__(self):
        rospack = rospkg.RosPack()
        control_path = rospack.get_path('husky_control')
        desc_path = rospack.get_path('husky_description')
        
        # Velocity limits
        with open(f'{control_path}/config/control.yaml') as f:
            ctrl = yaml.safe_load(f)['husky_velocity_controller']
        
        self.MAX_LINEAR_VELOCITY = float(ctrl['linear']['x']['max_velocity'])
        self.MAX_ANGULAR_VELOCITY = float(ctrl['angular']['z']['max_velocity'])
        
        # Physical dimensions
        xacro_file = f'{desc_path}/urdf/husky.urdf.xacro'
        result = subprocess.run(
            ['rosrun', 'xacro', 'xacro', xacro_file],
            capture_output=True, text=True, check=True
        )
        self._extract_urdf_specs(URDF.from_xml_string(result.stdout))
        self.COLLISION_RADIUS = max(self.LENGTH, self.WIDTH) / 2.0 + 0.05
        
        print(f"Loaded Husky: v_max={self.MAX_LINEAR_VELOCITY}m/s, "
              f"track_width={self.TRACK_WIDTH}m")

    def _extract_urdf_specs(self, robot):
        """Extract specs from URDF."""
        
        # Wheel radius
        for link in robot.links:
            if 'wheel' in link.name and link.collision:
                if hasattr(link.collision.geometry, 'radius'):
                    self.WHEEL_RADIUS = float(link.collision.geometry.radius)
                    break
        
        # Track width and wheelbase
        wheel_joints = [j for j in robot.joints if 'wheel' in j.name]
        left_joints = [j for j in wheel_joints if 'left' in j.name]
        right_joints = [j for j in wheel_joints if 'right' in j.name]
        front_joints = [j for j in wheel_joints if 'front' in j.name]
        rear_joints = [j for j in wheel_joints if 'rear' in j.name]
        
        if left_joints and right_joints:
            left_y = left_joints[0].origin.xyz[1]
            right_y = right_joints[0].origin.xyz[1]
            self.TRACK_WIDTH = abs(left_y - right_y)
        
        if front_joints and rear_joints:
            front_x = front_joints[0].origin.xyz[0]
            rear_x = rear_joints[0].origin.xyz[0]
            self.WHEELBASE = abs(front_x - rear_x)
        
        # Base dimensions
        base_link = next((link for link in robot.links if link.name == 'base_link'), None)
        if base_link:
            if base_link.collision and hasattr(base_link.collision.geometry, 'size'):
                geom = base_link.collision.geometry
                self.LENGTH = float(geom.size[0])
                self.WIDTH = float(geom.size[1])
                self.HEIGHT = float(geom.size[2])
        
        # Mass - check both base_link and inertial_link
        if base_link and base_link.inertial:
            self.MASS = float(base_link.inertial.mass)
        else:
            # Husky stores mass in separate inertial_link
            inertial_link = next((link for link in robot.links if link.name == 'inertial_link'), None)
            if inertial_link and inertial_link.inertial:
                self.MASS = float(inertial_link.inertial.mass)


HUSKY = HuskySpecsFromROS()

def _valid_spawn(x, y, obstacles, boundary_w, boundary_h):
    # At least 2.25m from all obstacles
    for obs in obstacles:
        if ((x - obs.x)**2 + (y - obs.y)**2)**0.5 < (obs.r + 2.25):
            return False
    # At least 1.5m from boundary
    if not (-boundary_w / 2 + 1.5 <= x <= boundary_w / 2 - 1.5):
        return False
    if not (-boundary_h / 2 + 1.5 <= y <= boundary_h / 2 - 1.5):
        return False
    return True


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Person:
    def __init__(self, x, y, v, omega, heading, pid, history_len=20, radius=0.5):
        self.x = x
        self.y = y
        self.v = v
        self.omega = omega
        self.heading = heading
        self.pid = pid
        self.radius = radius   # <--- ADDED
        self.hist = deque(maxlen=history_len)
        self.step_count = 0 
    # (rest unchanged)


    def step(self, dt: float):
        # Update heading with angular velocity
        self.heading += self.omega * dt
        self.heading = wrap_pi(self.heading)
        # Move forward in new heading
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt
        self.hist.append((self.x, self.y))

    def randomize_motion(self, rng):
        self.v = rng.uniform(0.7, 1.2)
        self.omega = rng.uniform(-0.3, 0.3)  # allow negative omega for left/right
        self.heading = rng.uniform(-math.pi, math.pi)



class CircleObstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def intersect_ray(self, rx, ry, rtheta, max_range):
        """Ray-circle intersection; return distance or None."""
        dx = math.cos(rtheta)
        dy = math.sin(rtheta)
        cx, cy, r = self.x, self.y, self.r
        fx = rx - cx
        fy = ry - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        disc = math.sqrt(disc)
        t1 = (-b - disc) / (2 * a)
        t2 = (-b + disc) / (2 * a)
        ts = [t for t in (t1, t2) if t >= 0]
        if not ts:
            return None
        d = min(ts)
        if d <= max_range:
            return d
        return None
    
class MotionAwareTrajectoryPredictor:
    def __init__(self, prediction_horizon=2.0, history_length=15, min_speed_threshold=0.5):
        self.prediction_horizon = prediction_horizon
        self.history_length = history_length
        self.min_speed_threshold = min_speed_threshold
        self.person_histories = {}
        self.last_cleanup_time = 0.0
        
        # Optimized GP kernel
        kernel = (ConstantKernel(1.0, constant_value_bounds="fixed") * 
                 RBF(length_scale=1.8, length_scale_bounds="fixed") + 
                 WhiteKernel(noise_level=0.05, noise_level_bounds="fixed"))
        
        self.gp_x = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, optimizer=None, normalize_y=False)
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, optimizer=None, normalize_y=False)

    def calculate_current_speed(self, history):
        """Calculate current movement speed from recent history"""
        if len(history) < 2:
            return 0.0
        
        recent = history[-3:] if len(history) >= 3 else history[-2:]
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i-1][0]  # time difference
            if dt > 0:
                dx = recent[i][1] - recent[i-1][1]  # x distance
                dy = recent[i][2] - recent[i-1][2]  # y distance
                distance = math.sqrt(dx**2 + dy**2)
                total_distance += distance
                total_time += dt
        
        return total_distance / total_time if total_time > 0 else 0.0

    def update_person_history(self, visible_people, current_time):
        """Update tracking history ONLY for visible people"""
        current_ids = set()
        
        # Only update histories for currently visible people
        for person in visible_people:
            pid = person.pid
            current_ids.add(pid)
            
            if pid not in self.person_histories:
                self.person_histories[pid] = []
            
            # Add current position to history
            self.person_histories[pid].append((current_time, person.x, person.y, 1.0))
            
            # Keep history limited
            if len(self.person_histories[pid]) > self.history_length:
                self.person_histories[pid].pop(0)

        # **NEW**: Mark non-visible people as "lost" but keep recent history
        # This allows predictions to continue briefly after losing sight
        for pid in list(self.person_histories.keys()):
            if pid not in current_ids:
                # Keep the history but don't update it
                # The prediction method will handle stale data
                pass
        
        # Cleanup very old tracks (increased threshold since updates are less frequent)
        if current_time - self.last_cleanup_time > 5.0:  # Increased from 3.0
            stale_ids = [pid for pid, hist in self.person_histories.items()
                        if not hist or current_time - hist[-1][0] > 10.0]  # Increased from 5.0
            for pid in stale_ids:
                del self.person_histories[pid]
            self.last_cleanup_time = current_time

    def predict_trajectory(self, person_id, current_time):
        """Motion-aware GP prediction - now handles intermittent observations"""
        if person_id not in self.person_histories:
            return None
        
        history = self.person_histories[person_id]
        if len(history) < 3:
            return None
        
        # Check how recent the last observation is
        last_observation_time = history[-1][0]
        time_since_last_seen = current_time - last_observation_time
        
        # If person hasn't been seen for too long, reduce prediction confidence
        if time_since_last_seen > 2.0:  # 2 seconds without observation
            return None  # Stop predicting after 2 seconds of not seeing them
        
        # Rest of the prediction method unchanged...
        current_speed = self.calculate_current_speed(history)
        if current_speed < self.min_speed_threshold:
            return None
        
        try:
            # Use recent history for prediction
            recent_history = history[-12:] if len(history) > 12 else history
            times = np.array([h[0] for h in recent_history])
            x_positions = np.array([h[1] for h in recent_history])
            y_positions = np.array([h[2] for h in recent_history])
            
            # Calculate current velocity
            if len(recent_history) >= 2:
                dt = times[-1] - times[-2]
                if dt > 0:
                    current_vx = (x_positions[-1] - x_positions[-2]) / dt
                    current_vy = (y_positions[-1] - y_positions[-2]) / dt
                else:
                    current_vx = current_vy = 0.0
            else:
                current_vx = current_vy = 0.0
            
            # Normalize time from most recent point
            times_norm = (times - times[-1]).reshape(-1, 1)
            
            # Fit GP models
            self.gp_x.fit(times_norm, x_positions)
            self.gp_y.fit(times_norm, y_positions)
            
            # Predict into future
            future_times = np.linspace(0.1, self.prediction_horizon, 10).reshape(-1, 1)
            
            # GP predictions
            x_pred_gp, x_std = self.gp_x.predict(future_times, return_std=True)
            y_pred_gp, y_std = self.gp_y.predict(future_times, return_std=True)
            
            # Velocity-biased predictions
            velocity_weight = 0.9
            current_pos_x, current_pos_y = x_positions[-1], y_positions[-1]
            x_pred_velocity = current_pos_x + current_vx * future_times.flatten()
            y_pred_velocity = current_pos_y + current_vy * future_times.flatten()
            
            # Combine GP and velocity predictions
            x_pred = (1 - velocity_weight) * x_pred_gp + velocity_weight * x_pred_velocity
            y_pred = (1 - velocity_weight) * y_pred_gp + velocity_weight * y_pred_velocity
            
            # Speed-based uncertainty
            speed_factor = min(2.0, current_speed / self.min_speed_threshold)
            uncertainty_factor = 0.3 + speed_factor * 0.4
            x_std = np.clip(x_std * uncertainty_factor, 0.1, 2.0)
            y_std = np.clip(y_std * uncertainty_factor, 0.1, 2.0)
            
            return {
                'person_id': person_id,
                'times': current_time + future_times.flatten(),
                'x_mean': x_pred,
                'y_mean': y_pred,
                'x_std': x_std,
                'y_std': y_std,
                'speed': current_speed,
                'current_pos': (current_pos_x, current_pos_y)
            }
            
        except Exception as e:
            # Fallback to linear prediction
            return self.linear_fallback(person_id, history, current_speed, current_time)
    
    def linear_fallback(self, person_id, history, current_speed, current_time):
        """Linear fallback for moving persons"""
        if len(history) < 2:
            return None
            
        recent = history[-3:] if len(history) >= 3 else history[-2:]
        
        # Calculate average velocity
        total_vx = total_vy = 0.0
        valid_intervals = 0
        
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i-1][0]
            if dt > 0:
                vx = (recent[i][1] - recent[i-1][1]) / dt
                vy = (recent[i][2] - recent[i-1][2]) / dt
                total_vx += vx
                total_vy += vy
                valid_intervals += 1
        
        if valid_intervals == 0:
            return None
            
        avg_vx = total_vx / valid_intervals
        avg_vy = total_vy / valid_intervals
        
        # Linear prediction
        future_times = np.linspace(0.1, self.prediction_horizon, 10)
        current_pos = recent[-1]
        x_pred = current_pos[1] + avg_vx * future_times
        y_pred = current_pos[2] + avg_vy * future_times
        
        # Speed-based uncertainty
        speed_factor = current_speed / self.min_speed_threshold
        base_uncertainty = 0.2 + speed_factor * 0.3
        uncertainty = base_uncertainty * (1 + future_times * 0.5)
        
        return {
            'person_id': person_id,
            'times': current_time + future_times,
            'x_mean': x_pred,
            'y_mean': y_pred,
            'x_std': uncertainty,
            'y_std': uncertainty,
            'speed': current_speed,
            'current_pos': (current_pos[1], current_pos[2])
        }
    
    def get_all_predictions(self, current_time):
        """Get predictions for all moving people"""
        predictions = {}
        for person_id in list(self.person_histories.keys()):
            pred = self.predict_trajectory(person_id, current_time)
            if pred is not None:
                predictions[person_id] = pred
        return predictions


class PygameTwinEnv:
    """
    Digital-twin environment for offline GAT+TD3 training.

    Graph nodes:
      - LiDAR: L nodes (default 20)
      - Robot: 1 node
      - Humans: up to H nodes (zero-padded if fewer visible)

    Node features (4 channels per node):
      - LiDAR_i: [dist_norm, cos(beam_angle), sin(beam_angle), 0]
      - Robot:   [dist_to_goal_x, y, v_actual, w_actual]
      - Human_k: [x_rel/15m, z_rel/15m, speed/1.5m/s, confidence]
    """

    def __init__(
        self,
        lidar_nodes=20,
        max_humans=3,
        world_size=(20, 20),
        boundary_size=(15, 15),
        num_obstacles=5,
        num_people=2,
        seed=0,
        render=False,
        ppm=50,
        ):


        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # graph layout
        self.L = int(lidar_nodes)
        self.R = 1
        self.H = int(max_humans)
        self.lidar_channels = 4  # LiDAR nodes have 4 channels
        self.robot_channels = 4  # Robot node has 4 channels  
        self.human_channels = 6  # Human nodes now have 6 channels

        # world
        self.world_w, self.world_h = world_size
        self.boundary_w, self.boundary_h = boundary_size
        self.num_obstacles = int(num_obstacles)
        self.num_people = int(num_people)
        self.render_enabled = bool(render and (pygame is not None))
        self.ppm = int(ppm)

        self.obstacles = []
        self.people = []
        self.goal = (0.0, 6.0)
        self.robot = dict(x=0.0, y=-4.0, theta=math.pi / 2)

        # lidar beams across [-pi/2, +pi/2]
        self.segment_angle = math.pi / self.L
        self.gaps = []
        start = -math.pi / 2 - 0.03
        for _ in range(self.L):
            self.gaps.append([start, start + self.segment_angle])
            start += self.segment_angle
        self.gaps[-1][-1] += 0.03

        # "camera" model (for human features)
        self.cam_fov = math.radians(120)  # +/-60 deg
        self.cam_max = 15.0

        # Use specifications from ROS packages
        self.v_max = HUSKY.MAX_LINEAR_VELOCITY
        self.w_max = HUSKY.MAX_ANGULAR_VELOCITY

        print(f"Environment initialized with Husky specs:")
        print(f"  v_max = {self.v_max} m/s")
        print(f"  w_max = {self.w_max} rad/s")
        print(f"  track_width = {HUSKY.TRACK_WIDTH} m")

        # internal state
        self.v = 0.0
        self.w = 0.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.accumulated_turn = 0.0  # Track cumulative rotation in current turn sequence
        self.turn_direction = 0

        self.trajectory_predictor = MotionAwareTrajectoryPredictor()
        self.simulation_time = 0.0

        self.cluster_explainer = NavigationClusterExplainer(
            n_clusters=6,
            embedding_dim=384,  # 20 LiDAR + 1 Robot + 3 Human = 24 nodes * 16 channels
            
        )


        # pygame
        if self.render_enabled:
            pygame.init()
            screen_width = int(self.world_w * self.ppm) + 320
            screen_height = int(self.world_h * self.ppm)
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("GAT-DT-TD3")
            self.clock = pygame.time.Clock()

    # ---------- WORLD ----------
    def _spawn_obstacles(self):
        """Spawn obstacles after robot and goal are set."""
        if not hasattr(self, 'robot') or not hasattr(self, 'goal'):
            raise RuntimeError("_spawn_obstacles requires robot and goal to be set first")
        
        x = self.robot['x']
        y = self.robot['y']
        gx, gy = self.goal
        
        # Try placing first obstacle in-line with robot-goal
        dx = gx - x
        dy = gy - y
        dist_to_goal = math.hypot(dx, dy)
        in_line_placed = False
        
        if dist_to_goal > 0.1 and self.rng.random() < 0.75: # 75% chance to place in-line
            ux = dx / dist_to_goal
            uy = dy / dist_to_goal
            
            for _ in range(20):
                t = self.rng.uniform(0.4, 0.6)
                base_x = x + t * dx
                base_y = y + t * dy
                
                perp_x = -uy
                perp_y = ux
                lateral_offset = self.rng.uniform(-0.25, 0.25)
                
                ox = base_x + lateral_offset * perp_x
                oy = base_y + lateral_offset * perp_y
                r = self.rng.uniform(0.3, 0.7)
                
                if math.hypot(ox - x, oy - y) < (r + 2.25):
                    continue
                if math.hypot(ox - gx, oy - gy) < (r + 2.25):
                    continue
                if not _valid_spawn(ox, oy, [], self.boundary_w, self.boundary_h):
                    continue
                
                self.obstacles.append(CircleObstacle(ox, oy, r))
                in_line_placed = True
                break
        
        # Spawn remaining obstacles randomly
        remaining = self.num_obstacles - (1 if in_line_placed else 0)
        for _ in range(remaining):
            for _ in range(100):
                ox = self.rng.uniform(-self.boundary_w / 2 + 2, self.boundary_w / 2 - 2)
                oy = self.rng.uniform(-self.boundary_h / 2 + 2, self.boundary_h / 2 - 2)
                r = self.rng.uniform(0.3, 0.7)
                
                if math.hypot(ox - x, oy - y) < (r + 2.25):
                    continue
                if math.hypot(ox - gx, oy - gy) < (r + 2.25):
                    continue
                
                valid = True
                for obs in self.obstacles:
                    if math.hypot(ox - obs.x, oy - obs.y) < (obs.r + r + 1.5):
                        valid = False
                        break
                if not valid:
                    continue
                
                if not _valid_spawn(ox, oy, [], self.boundary_w, self.boundary_h):
                    continue
                
                self.obstacles.append(CircleObstacle(ox, oy, r))
                break




    def _spawn_people(self):
        self.people = []
        MIN_HUMAN_DIST_FROM_ROBOT = 5  # Minimum 5m from husky
        MIN_HUMAN_DIST_FROM_OBSTACLES = 0.2  # Minimum 0.2m from obstacles
        
        for i in range(self.num_people):
            for attempt in range(100):
                x = self.rng.uniform(-self.boundary_w/2 + 1, self.boundary_w/2 - 1)
                y = self.rng.uniform(-self.boundary_h/2 + 1, self.boundary_h/2 - 1)
                v = self.rng.uniform(0.7, 1.2)
                omega = self.rng.uniform(-0.3, 0.3)
                heading = self.rng.uniform(-math.pi, math.pi)
                candidate = Person(x, y, v, omega, heading, pid=i, radius=0.5)
                
                # Check 1: Distance from robot
                robot_dist = math.hypot(candidate.x - self.robot["x"], 
                                    candidate.y - self.robot["y"])
                if robot_dist < MIN_HUMAN_DIST_FROM_ROBOT:
                    continue
                
                # Check 2: Distance from obstacles
                too_close_to_obstacle = False
                for obs in self.obstacles:
                    obs_dist = math.hypot(candidate.x - obs.x, candidate.y - obs.y)
                    if obs_dist < (obs.r + candidate.radius + MIN_HUMAN_DIST_FROM_OBSTACLES):
                        too_close_to_obstacle = True
                        break
                if too_close_to_obstacle:
                    continue
                
                # Check 3: No overlap with other humans
                overlap = False
                for q in self.people:
                    if math.hypot(candidate.x - q.x, candidate.y - q.y) < (candidate.radius + q.radius + 0.5):
                        overlap = True
                        break
                
                if not overlap:
                    candidate.hist.append((candidate.x, candidate.y))
                    self.people.append(candidate)
                    break




    def _move_people(self):
        margin = 0.1
        for i, p in enumerate(self.people):
            # Save original pos
            old_x, old_y = p.x, p.y
            p.step(TIME_DELTA)
            
            bounced = False

            # --- 1. Boundary bounce ---
            min_x = -self.boundary_w / 2 + p.radius + margin
            max_x = self.boundary_w / 2 - p.radius - margin
            min_y = -self.boundary_h / 2 + p.radius + margin
            max_y = self.boundary_h / 2 - p.radius - margin

            if not (min_x <= p.x <= max_x):
                # Reflect X heading
                p.heading = math.pi - p.heading
                p.x = max(min(p.x, max_x), min_x)
                bounced = True
            if not (min_y <= p.y <= max_y):
                p.heading = -p.heading
                p.y = max(min(p.y, max_y), min_y)
                bounced = True
            p.heading = wrap_pi(p.heading)

            # --- 2. Obstacle bounce ---
            for obs in self.obstacles:
                d = math.hypot(p.x - obs.x, p.y - obs.y)
                if d < obs.r + p.radius:
                    # Move back to pre-collision position
                    p.x, p.y = old_x, old_y
                    # Point directly away from obstacle (instant direction change)
                    angle_from_obs = math.atan2(p.y - obs.y, p.x - obs.x)
                    p.heading = angle_from_obs  # Face directly away
                    p.heading = wrap_pi(p.heading)
                    bounced = True
            # --- 2.5. Robot REAR bounce only --- (FIXED)
            robot_x, robot_y = self.robot["x"], self.robot["y"]
            robot_theta = self.robot["theta"]
            
            # Calculate person position relative to robot
            dx = p.x - robot_x
            dy = p.y - robot_y
            
            # Transform to robot frame (x=forward, y=left)
            person_x_robot = math.cos(-robot_theta) * dx - math.sin(-robot_theta) * dy
            person_y_robot = math.sin(-robot_theta) * dx + math.cos(-robot_theta) * dy
            
            # Husky dimensions
            husky_length = 0.9874
            husky_width = 0.5709
            robot_radius = max(husky_length, husky_width) / 2
            
            # Check if person is in collision range
            d_robot = math.hypot(dx, dy)
            if d_robot < robot_radius + p.radius + margin:
                # Only bounce if person is approaching from BEHIND (negative x in robot frame)
                if person_x_robot < -0.2:  # 20cm behind robot center
                    # Move back to pre-collision position
                    p.x, p.y = old_x, old_y
                    
                    # Bounce directly away from robot center
                    angle_from_robot = math.atan2(p.y - robot_y, p.x - robot_x)
                    p.heading = angle_from_robot
                    p.heading = wrap_pi(p.heading)
                    bounced = True
            # --- 3. Person-person bounce ---
            for j, q in enumerate(self.people):
                if i == j:
                    continue
                d = math.hypot(p.x - q.x, p.y - q.y)
                if d < p.radius + q.radius:
                    # Move back to pre-collision position
                    p.x, p.y = old_x, old_y
                    # Bounce off q by reflecting heading wrt centerline
                    angle_from_q = math.atan2(p.y - q.y, p.x - q.x)
                    p.heading = 2 * angle_from_q - p.heading
                    p.heading = wrap_pi(p.heading)
                    bounced = True

            if bounced:
                # To avoid stuck oscillations, jitter heading a bit
                p.heading += self.rng.uniform(-0.1, 0.1)
                p.omega = self.rng.uniform(-0.3, 0.3)
                p.v = self.rng.uniform(0.7, 1.2)



    # ---------- SENSORS ----------
    def _lidar_scan(self):
        rx, ry, rth = self.robot["x"], self.robot["y"], self.robot["theta"]
        dists = np.ones(self.L, dtype=np.float32) * LIDAR_RANGE
        for i in range(self.L):
            beam_angle = -math.pi / 2 + i * self.segment_angle + rth
            min_d = LIDAR_RANGE

            dx, dy = math.cos(beam_angle), math.sin(beam_angle)
            eps = 1e-6
            candidates = []
            # walls (AABB box)
            if abs(dx) > eps:
                t1 = ((self.boundary_w/ 2) - rx) / dx
                t2 = ((-self.boundary_w/ 2) - rx) / dx
                for t in (t1, t2):
                    if t > 0:
                        yhit = ry + t * dy
                        if -self.boundary_h / 2 <= yhit <= self.boundary_h / 2:
                            candidates.append(t)
            if abs(dy) > eps:
                t3 = ((self.boundary_h/ 2) - ry) / dy
                t4 = ((-self.boundary_h/ 2) - ry) / dy
                for t in (t3, t4):
                    if t > 0:
                        xhit = rx + t * dx
                        if -self.boundary_w / 2 <= xhit <= self.boundary_w/ 2:
                            candidates.append(t)
            if candidates:
                min_d = min(min_d, min(candidates))

            # circles
            for obs in self.obstacles:
                d = obs.intersect_ray(rx, ry, beam_angle, LIDAR_RANGE)
                if d is not None:
                    min_d = min(min_d, d)

            for person in self.people:
                # Treat each person as a circle obstacle for LiDAR
                px, py, pr = person.x, person.y, person.radius
                
                # Ray-circle intersection (same math as CircleObstacle)
                fx = rx - px
                fy = ry - py
                a = dx * dx + dy * dy
                b = 2 * (fx * dx + fy * dy)
                c = fx * fx + fy * fy - pr * pr
                
                disc = b * b - 4 * a * c
                if disc >= 0:
                    disc = math.sqrt(disc)
                    t1 = (-b - disc) / (2 * a)
                    t2 = (-b + disc) / (2 * a)
                    
                    ts = [t for t in (t1, t2) if t >= 0]
                    if ts:
                        d = min(ts)
                        if d <= LIDAR_RANGE:
                            min_d = min(min_d, d)

            dists[i] = float(min_d)
        return dists

    def _camera_detections(self):
        """Return sorted list of dicts with robot-frame person info."""
        rx, ry, rth = self.robot["x"], self.robot["y"], self.robot["theta"]
        out = []
        for p in self.people:
            dx = p.x - rx
            dy = p.y - ry
            # rotate by -rth to robot frame; x=fwd, z=left
            x_rel = math.cos(-rth) * dx - math.sin(-rth) * dy
            z_rel = math.sin(-rth) * dx + math.cos(-rth) * dy
            ang = math.atan2(z_rel, x_rel)
            dist = math.hypot(x_rel, z_rel)
            if -self.cam_fov / 2 <= ang <= self.cam_fov / 2 and 0.0 < dist <= self.cam_max:
                conf = 0.9
                out.append(
                    dict(
                        x=x_rel,
                        z=z_rel,
                        dist=dist,
                        pid=p.pid,
                        speed=p.v,
                        conf=conf,
                    )
                )
        out.sort(key=lambda d: d["dist"])  # nearest first
        return out

    def _predict_trajectories(self, horizon=20, dt=TIME_DELTA):
        # simple CV predictor over last K positions
        trajs = {}
        for p in self.people:
            if len(p.hist) < 2:
                continue
            (x1, y1), (x2, y2) = p.hist[-2], p.hist[-1]
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            xs, ys = [], []
            px, py = x2, y2
            for _ in range(horizon):
                px += vx * dt
                py += vy * dt
                xs.append(px)
                ys.append(py)
            trajs[p.pid] = (
                np.array(xs),
                np.array(ys),
                np.full(horizon, 0.3),
                np.full(horizon, 0.3),
            )
        return trajs
    
    def _get_visible_people(self):
        """Return only people currently visible to the camera"""
        rx, ry, rth = self.robot["x"], self.robot["y"], self.robot["theta"]
        visible_people = []
        
        for person in self.people:
            dx = person.x - rx
            dy = person.y - ry
            
            # Transform to robot frame
            x_rel = math.cos(-rth) * dx - math.sin(-rth) * dy
            z_rel = math.sin(-rth) * dx + math.cos(-rth) * dy
            
            # Check if within camera FOV and range
            ang = math.atan2(z_rel, x_rel)
            dist = math.hypot(x_rel, z_rel)
            
            if -self.cam_fov / 2 <= ang <= self.cam_fov / 2 and 0.0 < dist <= self.cam_max:
                visible_people.append(person)
        
        return visible_people
    

    def get_node_world_position(self, node_idx):
        """Map node index to world coordinates for attention visualization"""
        # LiDAR nodes (0 to L-1)
        if node_idx < self.L:
            # Get the center angle of this LiDAR segment
            angle = self.gaps[node_idx][0] + self.segment_angle / 2
            # Get actual beam distance from last lidar scan
            beam_dist = getattr(self, '_last_lidar_distances', [LIDAR_RANGE] * self.L)[node_idx]
            # Calculate world position
            x = self.robot['x'] + beam_dist * math.cos(angle + self.robot['theta'])
            y = self.robot['y'] + beam_dist * math.sin(angle + self.robot['theta'])
            return (x, y, 'lidar')
        
        # Robot node (index L)
        elif node_idx == self.L:
            return (self.robot['x'], self.robot['y'], 'robot')
        
        # Human nodes (L+1 to L+H)
        else:
            human_idx = node_idx - self.L - 1
            visible_people = self._get_visible_people()
            if human_idx < len(visible_people):
                p = visible_people[human_idx]
                return (p.x, p.y, 'human')
            else:
                # Zero-padded human (place at robot position with flag)
                return (self.robot['x'], self.robot['y'], 'padding')



    # ---------- REWARD ----------
    # def _avg_closest_nodes(self, lidar_dists, k=5):
    #     d = np.sort(lidar_dists)
    #     k = min(k, len(d))
    #     return float(np.mean(d[:k]))

    def _reward(self, target, collision, timeout, action, prev_dist, dist, lidar):
        if target:
            return 10.0  # Success
        
        if collision:
            return -2.0  # Collision penalty
        
        if timeout:
            return -1.0  # Timeout penalty
        
        # Progress reward
        progress = prev_dist - dist
        
        v_actual = action[0]
        w_actual = action[1]
        
        # Thresholds
        forward_threshold = 0.1   # m/s - velocity to be considered "moving forward"
        turn_velocity_threshold = 0.05  # rad/s - angular velocity to be considered "turning"
        turn_angle_threshold = np.pi / 2  # 90 degrees - free turning up to this point
        
        # Check if robot is moving forward (reset accumulator)
        if v_actual > forward_threshold and abs(w_actual) < turn_velocity_threshold:
            self.accumulated_turn = 0.0
            self.turn_direction = 0
            turn_penalty = 0.0
        
        # Check if robot is turning
        elif abs(w_actual) > turn_velocity_threshold:
            current_direction = np.sign(w_actual)
            
            # If continuing in same direction, accumulate
            if current_direction == self.turn_direction or self.turn_direction == 0:
                self.turn_direction = current_direction
                # Accumulate angular displacement this step
                angular_displacement = abs(w_actual) * TIME_DELTA
                self.accumulated_turn += angular_displacement
                
                # Only penalize turns beyond 90 degrees
                if self.accumulated_turn > turn_angle_threshold:
                    excess_turn = self.accumulated_turn - turn_angle_threshold
                    turn_penalty = -0.05 * excess_turn
                else:
                    turn_penalty = 0.0
            else:
                # Direction changed - reset and start new accumulation
                self.accumulated_turn = abs(w_actual) * TIME_DELTA
                self.turn_direction = current_direction
                turn_penalty = 0.0
        else:
            # Not moving much - no penalty
            turn_penalty = 0.0

        fwd_threshold = 0.5
        fwd_bonus = 0.0
        if v_actual > fwd_threshold:
            fwd_bonus = 0.01
            

        min_node_dist = float(np.min(lidar))

        safety_thresholds = [
        (1.0, 0.0),        # >= 1.0m: no penalty
        (0.9, -2.0/45.0),  # 0.9-1.0m
        (0.8, -4.0/45.0),  # 0.8-0.9m
        (0.7, -6.0/45.0),  # 0.7-0.8m
        (0.6, -8.0/45.0),  # 0.6-0.7m
        (0.0, -10.0/45.0)  # < 0.6m

        ]

        safety_penalty = 0.0
        for threshold, penalty in safety_thresholds:
            if min_node_dist <= threshold:
                safety_penalty = penalty
                break
    
        
        return progress + turn_penalty + safety_penalty + fwd_bonus

    # ---------- STATE (graph features) ----------
    # In pygame_env.py - update the _node_features method
    def _node_features(self, lidar, v_actual, w_actual):
        # LiDAR nodes (L x 4) - unchanged
        rx, ry, rth = self.robot["x"], self.robot["y"], self.robot["theta"]
        lidar_feats = np.zeros((self.L, 4), dtype=np.float32)
        for i, d in enumerate(lidar):
            ang = -math.pi / 2 + i * self.segment_angle
            lidar_feats[i, 0] = min(d, LIDAR_RANGE) / LIDAR_RANGE
            lidar_feats[i, 1] = math.cos(ang)
            lidar_feats[i, 2] = math.sin(ang)
            lidar_feats[i, 3] = 0.0

        gx, gy = self.goal[0], self.goal[1]

        dx_world = gx - rx
        dy_world = gy - ry

        dx_ego = dx_world * math.cos(rth) + dy_world * math.sin(rth)
        dy_ego = -dx_world * math.sin(rth) + dy_world * math.cos(rth)

        world_max = max(self.world_w, self.world_h)
        
        robot = np.array([
            dx_ego / world_max,                    # Feature 0: Goal X distance (forward/backward component)
            dy_ego / world_max,                    # Feature 1: Goal Y distance (left/right component)
            float(v_actual),                       # Feature 2: Linear velocity (UNCHANGED)
            float(w_actual),                       # Feature 3: Angular velocity (UNCHANGED)
        ], dtype=np.float32).reshape(1, 4)

        # # Robot node (1 x 4) - unchanged
        # dist = math.hypot(self.goal[0] - rx, self.goal[1] - ry)
        # beta = math.atan2(self.goal[1] - ry, self.goal[0] - rx)
        # theta = wrap_pi(beta - self.robot["theta"])
        # robot = np.array([
        #     min(dist / max(self.world_w, self.world_h), 1.0),
        #     theta / math.pi,
        #     float(v_actual),
        #     float(w_actual),
        # ], dtype=np.float32).reshape(1, 4)

        # Enhanced Human nodes (H x 6): [current_x, current_z, speed, conf, future_x, future_z]
        dets = self._camera_detections()
        predictions = self.get_trajectory_predictions()
        Hsel = dets[:self.H]
        humans = np.zeros((self.H, 6), dtype=np.float32)  # Changed from 4 to 6
        
        for i, dct in enumerate(Hsel):
            humans[i, 0] = dct["x"] / self.cam_max            # current x/forward
            humans[i, 1] = dct["z"] / self.cam_max            # current z/left
            humans[i, 2] = min(dct["speed"] / 1.5, 1.0)       # speed
            humans[i, 3] = float(dct["conf"])                 # confidence
            
            # NEW: Add predicted future position (~0.2 seconds ahead)
            if dct['pid'] in predictions:
                pred = predictions[dct['pid']]
                if len(pred['x_mean']) > 1:
                    # Get prediction at index 1 (approximately 0.2s future)
                    future_world_x = pred['x_mean'][1]
                    future_world_y = pred['y_mean'][1]
                    
                    # Transform future world position to robot-relative coordinates
                    dx = future_world_x - rx
                    dy = future_world_y - ry
                    future_x_rel = math.cos(-rth) * dx - math.sin(-rth) * dy
                    future_z_rel = math.sin(-rth) * dx + math.cos(-rth) * dy
                    
                    humans[i, 4] = future_x_rel / self.cam_max  # predicted x/forward
                    humans[i, 5] = future_z_rel / self.cam_max  # predicted z/left
            # If no prediction available, features remain 0

        return np.concatenate([lidar_feats.flatten(),robot.flatten(),humans.flatten()])

    
    def _check_collision_from_nodes(self, flat_features):
        """
        Check collision based on flattened state features.
        flat_features: flattened array of size (L*4 + R*4 + H*6)
        """
        # Calculate sizes
        lidar_size = self.L * 4  # 20 * 4 = 80
        robot_size = self.R * 4  # 1 * 4 = 4  
        human_size = self.H * 6  # 3 * 6 = 18
        
        # Extract LiDAR features and reshape
        lidar_flat = flat_features[:lidar_size]
        lidar_features = lidar_flat.reshape(self.L, 4)  # (20, 4)
        
        # Check LiDAR collision
        lidar_distances = lidar_features[:, 0] * LIDAR_RANGE
        min_lidar_dist = float(np.min(lidar_distances))
        lidar_collision = min_lidar_dist < COLLISION_DIST
        
        # Extract human features and reshape  
        human_start = lidar_size + robot_size
        human_flat = flat_features[human_start:human_start + human_size]
        human_features = human_flat.reshape(self.H, 6)  # (3, 6)
        
        # Check human collision
        human_collision = False
        for i in range(self.H):
            conf = human_features[i, 3]  # confidence at index 3
            if conf > 0:
                # Convert back from normalized coordinates
                x_rel = human_features[i, 0] * self.cam_max
                z_rel = human_features[i, 1] * self.cam_max
                human_dist = math.hypot(x_rel, z_rel)
                if human_dist < COLLISION_DIST:
                    human_collision = True
                    break
        
        return lidar_collision or human_collision




    # ---------- API ----------
    def reset(self):
        # Spawn obstacles first so we can check distances
        self.obstacles = []
        self.trajectory_predictor.person_histories.clear()
        self.simulation_time = 0.0 
        self.step_count = 0
        # Randomize husky start position within world bounds but not near obstacles
        # --- SAFER ROBOT SPAWN ---
        while True:
            x = self.rng.uniform(-self.boundary_w / 2 + 1.5, self.boundary_w / 2 - 1.5)
            y = self.rng.uniform(-self.boundary_h / 2 + 1.5, self.boundary_h / 2 - 1.5)
            if _valid_spawn(x, y, self.obstacles, self.boundary_w, self.boundary_h):
                break

        # --- SAFER GOAL SPAWN ---
        while True:
            gx = self.rng.uniform(-self.boundary_w / 2 + 2, self.boundary_w / 2 - 2)
            gy = self.rng.uniform(-self.boundary_h / 2 + 2, self.boundary_h / 2 - 2)
            if ((gx - x)**2 + (gy - y)**2)**0.5 < 5:  # Also check goal not too close to husky
                continue
            if _valid_spawn(gx, gy, self.obstacles, self.boundary_w, self.boundary_h):
                break

            

        self.robot = dict(x=x, y=y)
        self.goal = (gx, gy)
        self._spawn_obstacles()
        self._spawn_people()

        # --- FACE ROBOT TOWARD GOAL ---
        dx = gx - x
        dy = gy - y
        self.robot["theta"] = math.atan2(dy, dx)


        for p in self.people:
            p.hist.clear()
            p.hist.append((p.x, p.y))

        # Simplified state reset
        self.v = 0.0
        self.w = 0.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.accumulated_turn = 0.0  # Track cumulative rotation in current turn sequence
        self.turn_direction = 0

        # initial sensors & state
        lidar = self._lidar_scan()
        self._last_lidar_distances = lidar  # Store for visualization

        feats = self._node_features(lidar, 0.0, 0.0)
        state = feats.reshape(-1).astype(np.float32)

        self.prev_dist = math.hypot(self.goal[0] - self.robot["x"], self.goal[1] - self.robot["y"])
        self.done = False

        return state



    def step(self, action):
        """
        Action is normalized command: v_cmd_n in [0,1], w_cmd_n in [-1,1].
        We simulate Husky-like wheel-space dynamics to obtain actual (v, w), then integrate.
        """
        self.step_count += 1
        
        # Scale commands to physical limits (from ROS packages)
        v_cmd = float(action[0]) * self.v_max
        w_cmd = float(action[1]) * self.w_max

        # Simple first-order lag (approximates controller response)
        tau = 0.1  # 100ms time constant
        alpha = TIME_DELTA / (tau + TIME_DELTA)

        self.v = self.v + alpha * (v_cmd - self.v)
        self.w = self.w + alpha * (w_cmd - self.w)

        self.last_action = np.array([self.v, self.w], dtype=np.float32)

        # Integrate position using standard unicycle kinematics
        x, y, theta = self.robot["x"], self.robot["y"], self.robot["theta"]

        theta_new = wrap_pi(theta + self.w * TIME_DELTA)
        x_new = x + self.v * math.cos(theta) * TIME_DELTA
        y_new = y + self.v * math.sin(theta) * TIME_DELTA

        self.robot.update(x=x_new, y=y_new, theta=theta_new)

        

        # move people
        self._move_people()


        # sensors
        lidar = self._lidar_scan()
        self._last_lidar_distances = lidar  # Store for visualization

        

        feats = self._node_features(lidar, self.v, self.w)
        state = feats.reshape(-1).astype(np.float32)

        # collision / target based on node features
        collision = self._check_collision_from_nodes(feats)

        rx, ry = self.robot["x"], self.robot["y"]
        dist = math.hypot(self.goal[0] - rx, self.goal[1] - ry)
        target = dist < GOAL_REACHED_DIST

        # state (use actual v,w)
        

        self.simulation_time += TIME_DELTA

        visible_people = self._get_visible_people()
        self.trajectory_predictor.update_person_history(visible_people, self.simulation_time)

        timeout = self.step_count >= 500

        # reward
        reward = self._reward(target, collision, timeout, self.last_action, self.prev_dist, dist, lidar)
        self.prev_dist = dist

        done = bool(target or collision or timeout)
        self.done = done

        return state, float(reward), done, target, collision

    def get_trajectory_predictions(self):
        """Get current trajectory predictions for all moving people"""
        return self.trajectory_predictor.get_all_predictions(self.simulation_time)

    def render(self, attention_weights=None,gat_embedding=None):
        """Render the environment with pygame, optionally visualizing attention weights"""
        if not self.render_enabled:
            return

        # Process attention weights - now expects a pre-filtered attention map
        attention_map = {}
        if attention_weights is not None and isinstance(attention_weights, dict):
            # New format: pre-filtered top-k attention map from test script
            attention_map = attention_weights
        elif attention_weights is not None and len(attention_weights) > 0:
            # Old format: raw attention weights (for backward compatibility)
            try:
                edge_index, alpha = attention_weights[1]  # Layer 2 attention
                if hasattr(edge_index, 'cpu'):
                    edge_index = edge_index.cpu().numpy()
                if hasattr(alpha, 'cpu'):
                    alpha = alpha.cpu().numpy()

                # Average across heads
                alpha_avg = alpha.mean(axis=1) if len(alpha.shape) > 1 else alpha

                # Group edges by type
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

                    # Classify edges
                    if src_idx < self.L and tgt_idx < self.L:
                        edge_groups['lidar_to_lidar'].append((src_idx, tgt_idx, attention, i))
                    elif (src_idx == self.L and tgt_idx < self.L) or (tgt_idx == self.L and src_idx < self.L):
                        edge_groups['robot_to_lidar'].append((src_idx, tgt_idx, attention, i))
                    elif (src_idx == self.L and tgt_idx > self.L) or (tgt_idx == self.L and src_idx > self.L):
                        edge_groups['robot_to_human'].append((src_idx, tgt_idx, attention, i))
                    elif (src_idx < self.L and tgt_idx > self.L) or (tgt_idx < self.L and src_idx > self.L):
                        edge_groups['human_to_lidar'].append((src_idx, tgt_idx, attention, i))
                    elif src_idx > self.L and tgt_idx > self.L:
                        edge_groups['human_to_human'].append((src_idx, tgt_idx, attention, i))

                # Normalize within each group (old behavior)
                for group_name, edges in edge_groups.items():
                    if len(edges) == 0:
                        continue
                    
                    attentions = [e[2] for e in edges]
                    alpha_min = min(attentions)
                    alpha_max = max(attentions)

                    if alpha_max > alpha_min:
                        for src_idx, tgt_idx, attention, idx in edges:
                            alpha_norm = (attention - alpha_min) / (alpha_max - alpha_min)
                            attention_map[(src_idx, tgt_idx)] = alpha_norm
                    else:
                        # All same attention -> set to 0.5
                        for src_idx, tgt_idx, attention, idx in edges:
                            attention_map[(src_idx, tgt_idx)] = 0.5

            except Exception as e:
                print(f"Warning: Could not process attention weights: {e}")
                attention_map = {}

        # Handle pygame events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.render_enabled = False
                pygame.quit()
                return

        # Screen dimensions
        w, h = int(self.world_w * self.ppm), int(self.world_h * self.ppm)
        self.screen.fill((30, 30, 30))

        def wx(x):
            return int((x + self.world_w / 2) * self.ppm)

        def wy(y):
            return int((self.world_h / 2 - y) * self.ppm)

        # Draw boundary
        boundary_color = (100, 100, 100)
        boundary_width = 2
        boundary_left = wx(-self.boundary_w / 2)
        boundary_right = wx(self.boundary_w / 2)
        boundary_top = wy(self.boundary_h / 2)
        boundary_bottom = wy(-self.boundary_h / 2)
        pygame.draw.rect(self.screen, boundary_color,
                        (boundary_left, boundary_top,
                        boundary_right - boundary_left,
                        boundary_bottom - boundary_top), boundary_width)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (180, 60, 60),
                            (wx(obs.x), wy(obs.y)),
                            int(obs.r * self.ppm), 0)

        # Draw people trails and circles
        for p in self.people:
            if len(p.hist) > 1:
                pts = [(wx(x), wy(y)) for (x, y) in p.hist]
                pygame.draw.lines(self.screen, (80, 160, 220), False, pts, 2)
            pygame.draw.circle(self.screen, (70, 200, 255),
                            (wx(p.x), wy(p.y)),
                            int(p.radius * self.ppm), 0)

        # Draw trajectory predictions
        predictions = self.get_trajectory_predictions()
        for person_id, pred in predictions.items():
            pred_x_mean = pred['x_mean']
            pred_y_mean = pred['y_mean']
            if len(pred_x_mean) > 1:
                pred_points = []
                for i in range(len(pred_x_mean)):
                    px = pred_x_mean[i]
                    py = pred_y_mean[i]
                    pred_points.append((wx(px), wy(py)))
                if len(pred_points) > 1:
                    pygame.draw.lines(self.screen, (255, 255, 0), False, pred_points, 3)
                    
                    # Draw uncertainty ellipses
                    for i in range(0, len(pred['x_mean']), 2):
                        cx, cy = wx(pred['x_mean'][i]), wy(pred['y_mean'][i])
                        std_x, std_y = pred['x_std'][i] * self.ppm, pred['y_std'][i] * self.ppm
                        uncertainty_radius = int(max(std_x, std_y))
                        pygame.draw.circle(self.screen, (255, 255, 0), (cx, cy), uncertainty_radius, 1)

        # Get current state for drawing edges - USE DICTIONARY ACCESS
        rx, ry = self.robot["x"], self.robot["y"]  # Dictionary access, not object attribute
        rth = self.robot["theta"]  # Dictionary access
        robot_pos = (rx, ry)
        robot_idx = self.L

        lidar = self._lidar_scan()
        self._last_lidar_distances = lidar
        feats = self._node_features(lidar, self.last_action[0], self.last_action[1])

        # Extract LiDAR features from flattened array
        lidar_size = self.L * 4  # 20 * 4 = 80
        lidar_flat = feats[:lidar_size]
        lidar_features = lidar_flat.reshape(self.L, 4)  # (20, 4)

        # Get LiDAR node positions
        lidar_node_positions = []
        for i in range(self.L):
            beam_angle = -math.pi / 2 + i * self.segment_angle + rth
            beam_distance = lidar[i]
            node_x = rx + beam_distance * math.cos(beam_angle)
            node_y = ry + beam_distance * math.sin(beam_angle)
            lidar_node_positions.append((node_x, node_y))

        # Get human positions (camera detections in world frame)
        dets = self._camera_detections()
        human_world_positions = []
        for i, det in enumerate(dets[:self.H]):
            if det['conf'] > 0:
                human_world_x = det['x'] * math.cos(rth) - det['z'] * math.sin(rth) + rx
                human_world_y = det['x'] * math.sin(rth) + det['z'] * math.cos(rth) + ry
                human_world_positions.append((human_world_x, human_world_y))

        # --- DRAW ATTENTION EDGES ---
        # Default edge style (thin grey for non-highlighted edges)
        default_color = (100, 100, 100)  # Grey
        default_width = 1

        # 1. LiDAR-to-LiDAR edges
        for i in range(self.L):
            for j in range(i + 1, self.L):
                start_pos = (wx(lidar_node_positions[i][0]), wy(lidar_node_positions[i][1]))
                end_pos = (wx(lidar_node_positions[j][0]), wy(lidar_node_positions[j][1]))
                
                attention_ij = attention_map.get((i, j), None)
                attention_ji = attention_map.get((j, i), None)
                attention = max(attention_ij, attention_ji) if (attention_ij is not None or attention_ji is not None) else None
                
                if attention is not None:
                    gray = (100, 100, 100)
                    target_color = (255, 215, 0)  # Yellow for LiDAR-LiDAR
                    edge_color = (
                        int(gray[0] + (target_color[0] - gray[0]) * attention),
                        int(gray[1] + (target_color[1] - gray[1]) * attention),
                        int(gray[2] + (target_color[2] - gray[2]) * attention)
                    )
                    edge_width = max(1, int(1 + 3 * attention))
                else:
                    edge_color = default_color
                    edge_width = default_width
                
                pygame.draw.line(self.screen, edge_color, start_pos, end_pos, edge_width)

        # 2. Robot-to-LiDAR edges
        for i, lidar_pos in enumerate(lidar_node_positions):
            attention = attention_map.get((robot_idx, i), 0) or attention_map.get((i, robot_idx), 0)
            
            if attention > 0:
                gray = (100, 100, 100)
                target_color = (0, 255, 255)  # Cyan for Robot-LiDAR
                beam_color = (
                    int(gray[0] + (target_color[0] - gray[0]) * attention),
                    int(gray[1] + (target_color[1] - gray[1]) * attention),
                    int(gray[2] + (target_color[2] - gray[2]) * attention)
                )
                beam_width = max(1, int(1 + 3 * attention))
            else:
                beam_color = default_color
                beam_width = default_width
            
            pygame.draw.line(self.screen, beam_color,
                            (wx(robot_pos[0]), wy(robot_pos[1])),
                            (wx(lidar_pos[0]), wy(lidar_pos[1])), beam_width)

        # 3. Robot-to-Human edges
        for i, (human_world_x, human_world_y) in enumerate(human_world_positions):
            human_idx = self.L + 1 + i
            attention = attention_map.get((robot_idx, human_idx), 0) or attention_map.get((human_idx, robot_idx), 0)
            
            if attention > 0:
                gray = (100, 100, 100)
                target_color = (255, 100, 0)  # Red-Orange for Robot-Human
                edge_color = (
                    int(gray[0] + (target_color[0] - gray[0]) * attention),
                    int(gray[1] + (target_color[1] - gray[1]) * attention),
                    int(gray[2] + (target_color[2] - gray[2]) * attention)
                )
                edge_width = max(1, int(1 + 3 * attention))
            else:
                edge_color = default_color
                edge_width = default_width
            
            pygame.draw.line(self.screen, edge_color,
                            (wx(robot_pos[0]), wy(robot_pos[1])),
                            (wx(human_world_x), wy(human_world_y)), edge_width)

        # 4. Human-to-LiDAR edges
        for human_i, (human_world_x, human_world_y) in enumerate(human_world_positions):
            human_idx = self.L + 1 + human_i
            for lidar_idx, lidar_pos in enumerate(lidar_node_positions):
                attention_hl = attention_map.get((human_idx, lidar_idx), 0)
                attention_lh = attention_map.get((lidar_idx, human_idx), 0)
                attention = max(attention_hl, attention_lh)
                
                if attention > 0:
                    gray = (100, 100, 100)
                    target_color = (0, 255, 0)  # Green for Human-LiDAR
                    edge_color = (
                        int(gray[0] + (target_color[0] - gray[0]) * attention),
                        int(gray[1] + (target_color[1] - gray[1]) * attention),
                        int(gray[2] + (target_color[2] - gray[2]) * attention)
                    )
                    edge_width = max(1, int(1 + 3 * attention))
                else:
                    edge_color = default_color
                    edge_width = default_width
                
                pygame.draw.line(self.screen, edge_color,
                                (wx(human_world_x), wy(human_world_y)),
                                (wx(lidar_pos[0]), wy(lidar_pos[1])), edge_width)

        # 5. Human-to-Human edges
        for human_i in range(len(human_world_positions)):
            for human_j in range(human_i + 1, len(human_world_positions)):
                human_i_world_x, human_i_world_y = human_world_positions[human_i]
                human_j_world_x, human_j_world_y = human_world_positions[human_j]
                human_i_idx = self.L + 1 + human_i
                human_j_idx = self.L + 1 + human_j
                
                attention_ij = attention_map.get((human_i_idx, human_j_idx), 0)
                attention_ji = attention_map.get((human_j_idx, human_i_idx), 0)
                attention = max(attention_ij, attention_ji)
                
                if attention > 0:
                    gray = (100, 100, 100)
                    target_color = (255, 0, 255)  # Magenta for Human-Human
                    edge_color = (
                        int(gray[0] + (target_color[0] - gray[0]) * attention),
                        int(gray[1] + (target_color[1] - gray[1]) * attention),
                        int(gray[2] + (target_color[2] - gray[2]) * attention)
                    )
                    edge_width = max(1, int(1 + 3 * attention))
                else:
                    edge_color = default_color
                    edge_width = default_width
                
                pygame.draw.line(self.screen, edge_color,
                                (wx(human_i_world_x), wy(human_i_world_y)),
                                (wx(human_j_world_x), wy(human_j_world_y)), edge_width)

        # --- DRAW ROBOT ---
        half_length = HUSKY.LENGTH / 2
        half_width = HUSKY.WIDTH / 2
        corners = [(-half_length, -half_width), (half_length, -half_width), 
                (half_length, half_width), (-half_length, half_width)]
        
        world_corners = []
        for lx, ly in corners:
            wx_corner = rx + lx * math.cos(rth) - ly * math.sin(rth)
            wy_corner = ry + lx * math.sin(rth) + ly * math.cos(rth)
            world_corners.append((wx(wx_corner), wy(wy_corner)))
        
        pygame.draw.polygon(self.screen, (200, 200, 200), world_corners)
        pygame.draw.polygon(self.screen, (150, 150, 150), world_corners, 3)
        
        # Wheels
        wheel_positions = [
            (HUSKY.WHEELBASE * 0.45, HUSKY.TRACK_WIDTH / 2),
            (HUSKY.WHEELBASE * 0.45, -HUSKY.TRACK_WIDTH / 2),
            (-HUSKY.WHEELBASE * 0.45, HUSKY.TRACK_WIDTH / 2),
            (-HUSKY.WHEELBASE * 0.45, -HUSKY.TRACK_WIDTH / 2)
        ]
        wheel_radius_pixels = int(HUSKY.WHEEL_RADIUS * self.ppm)
        
        for wheel_lx, wheel_ly in wheel_positions:
            wheel_wx = rx + wheel_lx * math.cos(rth) - wheel_ly * math.sin(rth)
            wheel_wy = ry + wheel_lx * math.sin(rth) + wheel_ly * math.cos(rth)
            pygame.draw.circle(self.screen, (50, 50, 50),
                            (wx(wheel_wx), wy(wheel_wy)), wheel_radius_pixels)
        
        # Sensor payload
        sensor_length = HUSKY.LENGTH * 0.4
        sensor_width = HUSKY.WIDTH * 0.6
        sensor_corners = [
            (-sensor_length/2, -sensor_width/2), (sensor_length/2, -sensor_width/2),
            (sensor_length/2, sensor_width/2), (-sensor_length/2, sensor_width/2)
        ]
        sensor_world_corners = []
        for lx, ly in sensor_corners:
            wx_sensor = rx + lx * math.cos(rth) - ly * math.sin(rth)
            wy_sensor = ry + lx * math.sin(rth) + ly * math.cos(rth)
            sensor_world_corners.append((wx(wx_sensor), wy(wy_sensor)))
        pygame.draw.polygon(self.screen, (120, 120, 120), sensor_world_corners)
        
        # Robot center and heading arrow
        pygame.draw.circle(self.screen, (255, 255, 255), (wx(rx), wy(ry)), 4, 0)
        
        arrow_length = HUSKY.LENGTH * 0.8
        arrow_end_x = rx + arrow_length * math.cos(rth)
        arrow_end_y = ry + arrow_length * math.sin(rth)
        pygame.draw.line(self.screen, (255, 255, 0),
                        (wx(rx), wy(ry)), (wx(arrow_end_x), wy(arrow_end_y)), 4)
        
        arrow_size = HUSKY.LENGTH * 0.15
        left_arrow_x = arrow_end_x - arrow_size * math.cos(rth - math.pi/6)
        left_arrow_y = arrow_end_y - arrow_size * math.sin(rth - math.pi/6)
        right_arrow_x = arrow_end_x - arrow_size * math.cos(rth + math.pi/6)
        right_arrow_y = arrow_end_y - arrow_size * math.sin(rth + math.pi/6)
        pygame.draw.polygon(self.screen, (255, 255, 0),
                        [(wx(arrow_end_x), wy(arrow_end_y)),
                            (wx(left_arrow_x), wy(left_arrow_y)),
                            (wx(right_arrow_x), wy(right_arrow_y))])

        # Draw goal
        goal_x, goal_y = wx(self.goal[0]), wy(self.goal[1])
        star_radius = int(0.25 * self.ppm)
        inner_radius = int(star_radius * 0.4)

        star_points = []
        for i in range(10):  # 5-pointed star (10 points total)
            angle = math.pi / 2 + (i * math.pi / 5)  # Start at top
            radius = star_radius if i % 2 == 0 else inner_radius
            px = goal_x + int(radius * math.cos(angle))
            py = goal_y - int(radius * math.sin(angle))
            star_points.append((px, py))

        pygame.draw.polygon(self.screen, (90, 255, 90), star_points, 0)  # Filled star
        pygame.draw.polygon(self.screen, (60, 200, 60), star_points, 2)  # Outline

        # Draw LiDAR nodes
        for i, (node_x, node_y) in enumerate(lidar_node_positions):
            dist_norm = lidar_features[i, 0]
            color_intensity = int(255 * (1 - dist_norm))
            node_color = (211,211,211)
            pygame.draw.circle(self.screen, node_color, (wx(node_x), wy(node_y)), 6, 0)
            # Optional beam line
            pygame.draw.line(self.screen, (80, 80, 80),
                            (wx(rx), wy(ry)), (wx(node_x), wy(node_y)), 1)

        # Draw human nodes
        for i, (human_world_x, human_world_y) in enumerate(human_world_positions):
            pygame.draw.circle(self.screen, (255, 150, 0),
                            (wx(human_world_x), wy(human_world_y)), 10, 0)
            
        # --- DRAW ATTENTION LEGEND --- (Move to right side)
        # Calculate position on right side of screen
        screen_width = self.screen.get_width()
        legend_width = 220
        legend_x = screen_width - legend_width - 20  # 20px margin from right
        legend_y = 20  # pixels from top edge
        legend_line_height = 25
        legend_line_width = 40
        legend_text_offset = 50

        # Initialize font if not already done
        if not hasattr(self, 'legend_font'):
            try:
                self.legend_font = pygame.font.SysFont('Arial', 16)
            except:
                self.legend_font = pygame.font.Font(None, 20)

        # Legend title
        title_surface = self.legend_font.render("Attention Categories:", True, (255, 255, 255))
        self.screen.blit(title_surface, (legend_x, legend_y))

        # Legend entries (category name, color)
        legend_entries = [
            ("LiDAR-to-LiDAR", (255, 215, 0)),      # Yellow
            ("Robot-to-LiDAR", (0, 255, 255)),       # Cyan
            ("Robot-to-Human", (255, 100, 0)),       # Red-Orange
            ("Human-to-LiDAR", (0, 255, 0)),         # Green
            ("Human-to-Human", (255, 0, 255))        # Magenta
        ]

        current_y = legend_y + legend_line_height + 5

        for label, color in legend_entries:
            # Draw colored line sample
            line_start = (legend_x, current_y + 8)
            line_end = (legend_x + legend_line_width, current_y + 8)
            pygame.draw.line(self.screen, color, line_start, line_end, 3)
            
            # Draw label text
            text_surface = self.legend_font.render(label, True, (200, 200, 200))
            self.screen.blit(text_surface, (legend_x + legend_text_offset, current_y))
            
            current_y += legend_line_height


        # for label, color in legend_entries:
        #     # Draw colored line sample
        #     line_start = (legend_x, current_y + 8)
        #     line_end = (legend_x + legend_line_width, current_y + 8)
        #     pygame.draw.line(self.screen, color, line_start, line_end, 3)
            
        #     # Draw label text
        #     text_surface = self.legend_font.render(label, True, (200, 200, 200))
        #     self.screen.blit(text_surface, (legend_x + legend_text_offset, current_y))
            
        #     current_y += legend_line_height

        if gat_embedding is not None:
                self.cluster_explainer.update(gat_embedding)

        panel_y_position = 190
            
            # Draw cluster explanation panel
        draw_cluster_explanation_panel(self.screen, self.cluster_explainer, 
                              panel_x=None,  # Auto-position on right
                              panel_y=panel_y_position)



        pygame.display.flip()



# Add cluster labels as a constant
CLUSTER_LABELS = [
    "Clear Forward Motion",
    "Human Avoidance",
    "Static Obstacle Avoidance",
    "Narrow Space Maneuvering/Cautious",
    "Goal Approach"
]

class NavigationClusterExplainer:
    """Real-time clustering and explanation of GAT embeddings"""
    
    def __init__(self, n_clusters=6, embedding_dim=384):
        """
        Args:
            n_clusters: Number of semantic clusters (default 8)
            embedding_dim: Dimension of GAT output (24 nodes * 16 channels = 384)
        """
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        
        # Unlimited buffer for full episode collection
        self.embedding_buffer = []
        self.metadata_buffer = []
        
        # Cluster model
        self.kmeans = None
        self.is_fitted = False
        self.cluster_labels = CLUSTER_LABELS.copy()
        
        # Current state
        self.current_cluster = None
        self.cluster_confidences = np.zeros(n_clusters)
        
    def add_embedding(self, embedding, metadata):
        """Add GAT embedding and state metadata to buffer"""
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {len(embedding)}")
        
        self.embedding_buffer.append(embedding.copy())
        self.metadata_buffer.append(metadata.copy())
    
    def fit_clusters(self):
        """Fit K-means on all collected embeddings"""
        if len(self.embedding_buffer) < self.n_clusters:
            print(f"⚠️ Not enough data: {len(self.embedding_buffer)} embeddings, need at least {self.n_clusters}")
            return False
        
        embeddings = np.array(self.embedding_buffer)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(embeddings)
        self.is_fitted = True
        
        # Assign semantic labels based on metadata
        self._assign_semantic_labels()
        
        print(f"✅ Cluster model fitted on {len(embeddings)} embeddings")
        return True
    
    def _assign_semantic_labels(self):
        """Analyze cluster metadata and assign semantic labels with fallback for duplicates"""
        CLUSTER_LABELS = [
            "Clear Forward Motion",
            "Human Avoidance",
            "Static Obstacle Avoidance",
            "Narrow Space Maneuvering/Cautious",
            "Goal Approach"
        ]

        # First pass: determine best label for each cluster
        cluster_best_labels = []
        cluster_metadata_cache = []  # Cache metadata for second pass
        
        for cluster_id in range(self.n_clusters):
            # Get all metadata for this cluster
            cluster_mask = (self.kmeans.labels_ == cluster_id)
            cluster_meta = [self.metadata_buffer[i] for i in np.where(cluster_mask)[0]]
            
            if len(cluster_meta) == 0:
                cluster_best_labels.append("Unknown")
                cluster_metadata_cache.append(None)
                continue
            
            # Calculate average characteristics
            avg_min_lidar = np.mean([m['min_lidar_dist'] for m in cluster_meta])
            avg_speed = np.mean([m['robot_speed'] for m in cluster_meta])
            avg_goal_dist = np.mean([m['goal_distance'] for m in cluster_meta])
            avg_goal_angle = np.mean([m['goal_angle'] for m in cluster_meta])
            avg_angular_vel = np.mean([m['angular_velocity'] for m in cluster_meta])
            avg_forward_clear = np.mean([m['forward_clearance'] for m in cluster_meta])
            closest_human = np.min([m['closest_human'] for m in cluster_meta])
            
            # Cache for later
            metadata = {
                'avg_min_lidar': avg_min_lidar,
                'avg_speed': avg_speed,
                'avg_goal_dist': avg_goal_dist,
                'avg_goal_angle': avg_goal_angle,
                'avg_angular_vel': avg_angular_vel,
                'avg_forward_clear': avg_forward_clear,
                'closest_human': closest_human
            }
            cluster_metadata_cache.append(metadata)
            
            # Priority-based label assignment (most specific first)
            if avg_goal_dist < 2.0 and avg_goal_angle < 0.35:
                label = "Goal Approach"
            
            elif closest_human < 3.0 and closest_human > 0.55:
                label = "Human Avoidance"
            
            elif avg_min_lidar < 2.5 and avg_angular_vel > 0.3 and avg_speed < 0.4:
                label = "Narrow Space Maneuvering/Cautious"
            
            elif avg_min_lidar > 1.5 and avg_min_lidar < 3.5 and avg_speed > 0.4:
                label = "Static Obstacle Avoidance"
            
            elif avg_forward_clear > 2.5 and avg_speed > 0.5:
                label = "Clear Forward Motion"
            
            else:
                label = "Goal Seeking"  # Default fallback
            
            cluster_best_labels.append(label)
        
        # Second pass: assign labels ensuring uniqueness (UNINDENT THIS SECTION)
        self.cluster_id_to_label = {}
        used_labels = set()
        
        for cluster_id in range(self.n_clusters):
            best_label = cluster_best_labels[cluster_id]
            
            if best_label not in used_labels:
                # First choice available
                self.cluster_id_to_label[cluster_id] = best_label
                used_labels.add(best_label)
            else:
                # Try ALL labels in priority order (not just ones this cluster matches)
                assigned = False
                for label in CLUSTER_LABELS:
                    if label not in used_labels:
                        self.cluster_id_to_label[cluster_id] = label
                        used_labels.add(label)
                        assigned = True
                        break
                
                # If all main labels taken, use Goal Seeking
                if not assigned:
                    if "Goal Seeking" not in used_labels:
                        self.cluster_id_to_label[cluster_id] = "Goal Seeking"
                        used_labels.add("Goal Seeking")
                    else:
                        # Multiple Goal Seeking needed
                        suffix = 2
                        while f"Goal Seeking ({suffix})" in used_labels:
                            suffix += 1
                        self.cluster_id_to_label[cluster_id] = f"Goal Seeking ({suffix})"
                        used_labels.add(f"Goal Seeking ({suffix})")
        
        # Print results
        print("\n📊 Cluster Label Assignment:")
        for cluster_id in range(self.n_clusters):
            cluster_mask = (self.kmeans.labels_ == cluster_id)
            sample_count = np.sum(cluster_mask)
            label = self.cluster_id_to_label[cluster_id]
            best_match = cluster_best_labels[cluster_id]
            
            if label == best_match:
                print(f"  Cluster {cluster_id}: {label} ({sample_count} samples) ✓")
            else:
                print(f"  Cluster {cluster_id}: {label} ({sample_count} samples) [wanted: {best_match}]")



    
    def update(self, embedding, metadata=None):
        """Update current cluster assignment and confidences"""
        if metadata is not None:
            self.add_embedding(embedding, metadata)
        
        if not self.is_fitted:
            self.current_cluster = None
            self.cluster_confidences = np.zeros(self.n_clusters)
            return None
        
        # Get cluster assignment
        embedding_2d = embedding.reshape(1, -1)
        cluster_id = self.kmeans.predict(embedding_2d)[0]
        self.current_cluster = cluster_id
        
        # Compute softmax confidence distribution
        distances = self.kmeans.transform(embedding_2d)[0]
        temperature = 0.5
        logits = -distances / temperature
        exp_logits = np.exp(logits - np.max(logits))
        self.cluster_confidences = exp_logits / exp_logits.sum()
        
        return cluster_id
    
    def get_explanation(self):
        """Get current cluster label and confidence distribution"""
        if self.current_cluster is None:
            return None
        
        sorted_indices = np.argsort(self.cluster_confidences)[::-1]
        
        # Use assigned semantic labels
        active_label = self.cluster_id_to_label.get(self.current_cluster, "Unknown")
        
        # Create ordered label list based on cluster_id
        ordered_labels = [self.cluster_id_to_label.get(i, "Unknown") for i in range(self.n_clusters)]
        
        return {
            'active_cluster': self.current_cluster,
            'active_label': active_label,
            'active_confidence': self.cluster_confidences[self.current_cluster],
            'sorted_indices': sorted_indices,
            'confidences': self.cluster_confidences,
            'ordered_labels': ordered_labels
        }
    
    def save_model(self, filepath):
        """Save fitted cluster model and label mapping"""
        if not self.is_fitted:
            print("⚠️ Cannot save: model not fitted yet")
            return False
        
        save_data = {
            'kmeans_model': self.kmeans,
            'cluster_id_to_label': self.cluster_id_to_label,
            'n_clusters': self.n_clusters,
            'embedding_dim': self.embedding_dim,
            'total_samples': len(self.embedding_buffer)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Cluster model saved to: {filepath}")
        return True
    
    def load_model(self, filepath):
        """Load pre-fitted cluster model"""
        if not os.path.exists(filepath):
            print(f"⚠️ Cluster model not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.kmeans = save_data['kmeans_model']
        self.cluster_id_to_label = save_data['cluster_id_to_label']
        self.n_clusters = save_data['n_clusters']
        self.embedding_dim = save_data['embedding_dim']
        self.is_fitted = True
        
        print(f"✅ Cluster model loaded from: {filepath}")
        print(f"   Trained on {save_data['total_samples']} samples")
        print("\n📊 Loaded Cluster Labels:")
        for cluster_id, label in self.cluster_id_to_label.items():
            print(f"  Cluster {cluster_id}: {label}")
        
        return True



def draw_cluster_explanation_panel(screen, explainer, panel_x=None, panel_y=20):
    """
    Draw side panel showing navigation state cluster with confidence distribution
    
    Args:
        screen: pygame screen surface
        explainer: NavigationClusterExplainer instance
        panel_x: X position (if None, auto-positions on right side)
        panel_y: Top-left Y position of panel
    """
    # Initialize fonts
    try:
        font = pygame.font.SysFont('Arial', 14)
        title_font = pygame.font.SysFont('Arial', 16, bold=True)
    except:
        font = pygame.font.Font(None, 18)
        title_font = pygame.font.Font(None, 20)
    
    # Panel dimensions
    panel_width = 300  # Increased width for better spacing
    
    # Auto-position on right side if panel_x not specified
    if panel_x is None:
        screen_width = screen.get_width()
        panel_x = screen_width - panel_width - 20  # 20px margin from right edge
    
    if not explainer.is_fitted:
        text = font.render(
            f"Collecting data... {len(explainer.embedding_buffer)} samples",
            True, (200, 200, 200)
        )
        screen.blit(text, (panel_x, panel_y))
        return
    
    explanation = explainer.get_explanation()
    if explanation is None:
        return
    ordered_labels = explanation['ordered_labels']
    panel_height = 30 + 25 * len(CLUSTER_LABELS) + 20  # Title + labels + padding
    line_height = 25
    
    # Draw panel background with border
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, (40, 40, 40), panel_rect)  # Dark background
    pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)  # Border
    
    # Title
    title_surface = title_font.render("NAVIGATION STATE", True, (255, 255, 255))
    screen.blit(title_surface, (panel_x + 10, panel_y + 8))
    
    # Separator line
    pygame.draw.line(screen, (100, 100, 100), 
                     (panel_x + 10, panel_y + 30), 
                     (panel_x + panel_width - 10, panel_y + 30), 1)
    
    # Draw cluster labels with confidences
    current_y = panel_y + 40
    active_idx = explanation['active_cluster']
    
    for i in range(len(ordered_labels)):
        confidence = explanation['confidences'][i]
        label = ordered_labels[i]  # Use dynamic label
        
        # Check if this is the active cluster
        is_active = (i == active_idx)
        
        # Color coding
        if is_active:
            text_color = (100, 255, 100)  # Bright green for active
            bg_color = (60, 80, 60)  # Highlight background
            indicator = "► "
        else:
            text_color = (180, 180, 180)  # Gray for inactive
            bg_color = None
            indicator = "   "
        
        # Draw background highlight for active cluster
        if bg_color:
            highlight_rect = pygame.Rect(panel_x + 5, current_y - 2, panel_width - 10, line_height)
            pygame.draw.rect(screen, bg_color, highlight_rect)
        
        # Format label text with confidence percentage
        confidence_pct = int(confidence * 100)
        label_text = f"{indicator}{label}"
        confidence_text = f"{confidence_pct}%"
        
        # Draw label (left side)
        label_surface = font.render(label_text, True, text_color)
        screen.blit(label_surface, (panel_x + 10, current_y))
        
        # Draw mini confidence bar (middle, with more spacing)
        if i in explanation['sorted_indices'][:3]:
            bar_width = 30
            bar_height = 6
            bar_x = panel_x + panel_width - 80  # Positioned with more space from percentage
            bar_y = current_y + 8
            
            # Background bar
            pygame.draw.rect(screen, (60, 60, 60), 
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Filled portion based on confidence
            filled_width = int(bar_width * confidence)
            if filled_width > 0:
                bar_color = (100, 255, 100) if is_active else (150, 150, 150)
                pygame.draw.rect(screen, bar_color,
                               (bar_x, bar_y, filled_width, bar_height))
        
        # Draw confidence percentage (right side, more spacing from bar)
        conf_surface = font.render(confidence_text, True, text_color)
        conf_x = panel_x + panel_width - 40  # More space from right edge
        screen.blit(conf_surface, (conf_x, current_y))
        
        current_y += line_height


