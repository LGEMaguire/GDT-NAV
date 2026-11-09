# test_gat_td3.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import argparse
import time
from collections import deque
from pygame_env_rosv5 import PygameTwinEnv, TIME_DELTA
from graph_utils import create_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RealTimeRunner:
    """Ensures consistent real-time step duration"""
    def __init__(self, target_step_duration=TIME_DELTA):
        self.target_step_duration = target_step_duration
        self.last_step_time = None
        self.step_count = 0
        self.total_sleep_time = 0.0

    def sleep_to_sync(self):
        """Sleep to ensure real-time step duration"""
        current_time = time.time()
        if self.last_step_time is not None:
            elapsed = current_time - self.last_step_time
            sleep_time = self.target_step_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.total_sleep_time += sleep_time
        self.last_step_time = time.time()
        self.step_count += 1

    def get_stats(self):
        """Return timing statistics"""
        if self.step_count > 0:
            avg_sleep = self.total_sleep_time / self.step_count
            return {
                'steps': self.step_count,
                'total_sleep': self.total_sleep_time,
                'avg_sleep_per_step': avg_sleep,
                'target_step_duration': self.target_step_duration
            }
        return {}


def filter_top_attention_weights(attention_weights, top_k=3):
    """
    Filter attention weights to highlight only top-k edges per GAT layer category.
    Returns a structured format compatible with pygame_env_rosv5.py rendering.
    
    Args:
        attention_weights: List of tuples [(edge_index, alpha), ...] from GAT layers
        top_k: Number of highest-weight edges to highlight per category (default: 3)
    
    Returns:
        filtered_data: Dictionary with edge categories and their top-k edges
    """
    # Use layer 2 attention (index 1) as in the original code
    if len(attention_weights) < 2:
        return None
    
    edge_index, alpha = attention_weights[1]  # Layer 2
    
    # Convert to numpy
    if hasattr(edge_index, 'cpu'):
        edge_index = edge_index.cpu().numpy()
    if hasattr(alpha, 'cpu'):
        alpha = alpha.cpu().numpy()
    
    # Average across heads if multi-head
    alpha_avg = alpha.mean(axis=1) if len(alpha.shape) > 1 else alpha
    
    # Determine node counts from the agent (20 LiDAR, 1 Robot, 3 Humans)
    L = 20  # LiDAR nodes
    R = 1   # Robot node (index L)
    H = 3   # Human nodes (indices L+1 to L+H)
    
    # Categorize edges by type
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
        
        # Classify edge by source and target indices
        if src_idx < L and tgt_idx < L:
            edge_groups['lidar_to_lidar'].append((src_idx, tgt_idx, attention, i))
        elif src_idx == L and tgt_idx < L:
            edge_groups['robot_to_lidar'].append((src_idx, tgt_idx, attention, i))
        elif tgt_idx == L and src_idx < L:
            edge_groups['robot_to_lidar'].append((src_idx, tgt_idx, attention, i))
        elif src_idx == L and tgt_idx > L:
            edge_groups['robot_to_human'].append((src_idx, tgt_idx, attention, i))
        elif tgt_idx == L and src_idx > L:
            edge_groups['robot_to_human'].append((src_idx, tgt_idx, attention, i))
        elif src_idx < L and tgt_idx > L:
            edge_groups['human_to_lidar'].append((src_idx, tgt_idx, attention, i))
        elif tgt_idx < L and src_idx > L:
            edge_groups['human_to_lidar'].append((src_idx, tgt_idx, attention, i))
        elif src_idx > L and tgt_idx > L:
            edge_groups['human_to_human'].append((src_idx, tgt_idx, attention, i))
    
    # Process each category: find top-k edges
    filtered_attention_map = {}
    
    for group_name, edges in edge_groups.items():
        if len(edges) == 0:
            continue
        
        # Extract attention values
        attentions = [e[2] for e in edges]
        
        # Check if all values are the same (no variance)
        if len(set(attentions)) == 1:
            # All same -> keep all as default (grey, thin)
            continue
        
        # Sort edges by attention (descending)
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        
        # Take top-k
        k = min(top_k, len(sorted_edges))
        top_edges = sorted_edges[:k]
        
        # Get min/max of top-k for normalization
        top_attentions = [e[2] for e in top_edges]
        alpha_min = min(top_attentions)
        alpha_max = max(top_attentions)
        
        # Normalize top-k edges to [0, 1]
        for src_idx, tgt_idx, attention, idx in top_edges:
            if alpha_max > alpha_min:
                alpha_norm = (attention - alpha_min) / (alpha_max - alpha_min)
            else:
                alpha_norm = 1.0  # Single top edge
            
            # Store bidirectional (for symmetric lookup)
            filtered_attention_map[(src_idx, tgt_idx)] = alpha_norm
            filtered_attention_map[(tgt_idx, src_idx)] = alpha_norm
    
    return filtered_attention_map


# Copy the model classes from your training script
class GATNetwork(nn.Module):
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

    def forward(self, lidar_nodes, robot_nodes, human_nodes, edge_index, return_attention=False):
        # Project to uniform dimensions
        lidar_proj = self.lidar_proj(lidar_nodes)
        robot_proj = self.robot_proj(robot_nodes)
        human_proj = self.human_proj(human_nodes)

        # Concatenate projected features
        x = torch.cat([lidar_proj, robot_proj, human_proj], dim=0)
        x_init = x.clone()

        # Apply GAT layers with optional attention extraction
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
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        return self.tanh(self.l3(s))


class GATTD3Tester:
    def __init__(self, lidar_nodes=20, robot_nodes=1, human_nodes=3,
                 lidar_channels=4, robot_channels=4, human_channels=6,
                 gat_out_channels=16, num_heads=4, action_dim=2, max_action=1.0):
        self.L = lidar_nodes
        self.R = robot_nodes
        self.H = human_nodes
        self.total_nodes = self.L + self.R + self.H
        self.lidar_channels = lidar_channels
        self.robot_channels = robot_channels
        self.human_channels = human_channels

        # Create GAT network
        self.gat = GATNetwork(
            lidar_channels, robot_channels, human_channels,
            gat_out_channels, num_heads, lidar_nodes, robot_nodes, human_nodes
        ).to(device)

        # Create Actor network
        input_dim = gat_out_channels * self.total_nodes
        self.actor = Actor(input_dim, action_dim).to(device)
        self.max_action = max_action
        self.edge_index = create_edge_index(self.L, self.R, self.H).to(device)

    def _flat_to_nodes(self, flat_state):
        """Convert flattened mixed-dimension state to list of node tensors"""
        # Calculate expected sizes
        lidar_size = self.L * self.lidar_channels  # 20 * 4 = 80
        robot_size = self.R * self.robot_channels  # 1 * 4 = 4
        human_size = self.H * self.human_channels  # 3 * 6 = 18

        # Split flattened state
        lidar_flat = flat_state[:lidar_size]
        robot_flat = flat_state[lidar_size:lidar_size + robot_size]
        human_flat = flat_state[lidar_size + robot_size:lidar_size + robot_size + human_size]

        # Reshape to node format
        lidar_nodes = torch.tensor(lidar_flat, dtype=torch.float32, device=device).view(self.L, self.lidar_channels)
        robot_nodes = torch.tensor(robot_flat, dtype=torch.float32, device=device).view(self.R, self.robot_channels)
        human_nodes = torch.tensor(human_flat, dtype=torch.float32, device=device).view(self.H, self.human_channels)

        return lidar_nodes, robot_nodes, human_nodes

    @torch.no_grad()
    def act(self, flat_state):
        """Get action from trained policy (no exploration noise)"""
        # Convert flattened state back to node features
        lidar_nodes, robot_nodes, human_nodes = self._flat_to_nodes(flat_state)
        z = self.gat(lidar_nodes, robot_nodes, human_nodes, self.edge_index).view(1, -1)
        return self.actor(z).cpu().numpy().flatten()

    @torch.no_grad()
    def act_with_attention(self, flat_state):
        """Get action from trained policy and return attention weights"""
        # Convert flattened state back to node features
        lidar_nodes, robot_nodes, human_nodes = self._flat_to_nodes(flat_state)

        # Get embeddings with attention weights
        z, attention_weights = self.gat(lidar_nodes, robot_nodes, human_nodes,
                                        self.edge_index, return_attention=True)

        # Get action
        action = self.actor(z.view(1, -1)).cpu().numpy().flatten()

        return action, attention_weights

    def load_models(self, model_dir, model_name="TD3_gat_humans"):
        """Load trained models"""
        gat_path = os.path.join(model_dir, f"{model_name}_gat.pth")
        actor_path = os.path.join(model_dir, f"{model_name}_actor.pth")

        if not os.path.exists(gat_path):
            raise FileNotFoundError(f"GAT model not found: {gat_path}")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor model not found: {actor_path}")

        # Load state dictionaries
        self.gat.load_state_dict(torch.load(gat_path, map_location=device))
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

        # Set to evaluation mode
        self.gat.eval()
        self.actor.eval()

        print(f"✅ Models loaded successfully from {model_dir}")

def extract_metadata_from_state(flat_state, env):
    """Extract interpretable metadata from flattened state"""
    # Parse state (80 lidar + 4 robot + 18 human = 102 total)
    lidar_features = flat_state[:80].reshape(20, 4)
    robot_features = flat_state[80:84]
    human_features = flat_state[84:102].reshape(3, 6)
    
    # Extract key metrics
    lidar_distances = lidar_features[:, 0] * 10.01  # Denormalize
    goal_x = robot_features[0]
    goal_y = robot_features[1]
    linear_vel = robot_features[2]
    angular_vel = robot_features[3]
    
    # Human distances (x, z positions)
    human_distances = []
    for i in range(3):
        x_rel, z_rel = human_features[i, 0], human_features[i, 1]
        if human_features[i, 3] > 0:  # confidence > 0
            dist = np.sqrt(x_rel**2 + z_rel**2)
            human_distances.append(dist)
    
    closest_human = min(human_distances) if human_distances else 999.0
    
    metadata = {
        'min_lidar_dist': float(np.min(lidar_distances)),
        'robot_speed': abs(float(linear_vel)),
        'goal_distance': float(np.sqrt(goal_x**2 + goal_y**2)),
        'goal_angle': abs(float(np.arctan2(goal_y, goal_x))),
        'closest_human': float(closest_human),
        'angular_velocity': abs(float(angular_vel)),
        'forward_clearance': float(np.mean(lidar_distances[8:13])),  # Front beams
        'lidar_variance': float(np.std(lidar_distances))
    }
    
    return metadata


def test_agent(model_dir="pytorch_models", model_name="TD3_gat_humans",
               n_episodes=50, max_ep_len=500, render=False, save_results=True,
               lidar_nodes=20, human_nodes=3, seed=42, real_time=True, top_k_edges=3, collect_cluster_data=False, cluster_model_path="cluster_model.pkl"):
    """
    Test the trained GAT+TD3 agent over multiple episodes

    Args:
        model_dir: Directory containing saved models
        model_name: Base name of saved models
        n_episodes: Number of test episodes
        max_ep_len: Maximum steps per episode
        render: Whether to render the environment
        save_results: Whether to save detailed results
        lidar_nodes: Number of LiDAR nodes
        human_nodes: Number of human nodes
        seed: Random seed for reproducible testing
        real_time: Whether to sync to real-time (10 Hz)
        top_k_edges: Number of top attention edges to highlight (default: 3)
    """

    # Create environment and agent
    env = PygameTwinEnv(
        lidar_nodes=lidar_nodes,
        max_humans=human_nodes,
        seed=seed,
        render=render
    )

    # Load or initialize cluster model
    if collect_cluster_data:
        print("📊 COLLECTING CLUSTER DATA MODE")
        print("   Gathering embeddings across all episodes...")
    elif os.path.exists(cluster_model_path):
        env.cluster_explainer.load_model(cluster_model_path)
        print("✅ Using pre-trained cluster model")
    else:
        print(f"⚠️ No cluster model found at {cluster_model_path}")
        print("   Run with --collect-cluster-data first")

    agent = GATTD3Tester(
        lidar_nodes=lidar_nodes,
        robot_nodes=1,
        human_nodes=human_nodes,
        lidar_channels=4,
        robot_channels=4,
        human_channels=6
    )

    # Load trained models
    try:
        agent.load_models(model_dir, model_name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    # Initialize real-time runner
    if real_time:
        rt_runner = RealTimeRunner(target_step_duration=TIME_DELTA)
        print(f"🕐 Real-time mode enabled (10 Hz, {TIME_DELTA}s per step)")
    else:
        print("⚡ Maximum speed mode (no real-time sync)")

    print(f"🎨 Attention visualization: Highlighting top-{top_k_edges} edges per layer")

    # Test metrics
    episode_rewards = []
    episode_lengths = []
    successes = []
    collisions = []
    timeouts = []

    # Detailed episode data
    episode_data = []

    print(f"\n🚀 Starting {n_episodes} test episodes...")
    print("=" * 60)

    total_test_start_time = time.time()

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        episode_start_time = time.time()
        step_times = []

        while not done and episode_length < max_ep_len:
            step_start_time = time.time()

            # Get action from trained policy with attention weights
            action, attention_weights = agent.act_with_attention(state)

            lidar_nodes, robot_nodes, human_nodes = agent._flat_to_nodes(state)
            gat_embedding, _ = agent.gat(lidar_nodes, robot_nodes, human_nodes,
                             agent.edge_index, return_attention=True)
            gat_embedding_flat = gat_embedding.detach().cpu().numpy().flatten()  # Shape: (384,)

            metadata = extract_metadata_from_state(state, env)

            env.cluster_explainer.update(gat_embedding_flat, metadata)

            # Filter attention weights to highlight only top-k edges
            filtered_attention = filter_top_attention_weights(attention_weights, top_k=top_k_edges)

            # Convert action to environment format
            action_env = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float32)

            # Step environment
            next_state, reward, done, target_reached, collision = env.step(action_env)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                env.render(attention_weights=filtered_attention, 
                        gat_embedding=gat_embedding_flat)


            # Real-time synchronization
            if real_time:
                rt_runner.sleep_to_sync()

            step_end_time = time.time()
            step_times.append(step_end_time - step_start_time)

        episode_time = time.time() - episode_start_time
        avg_step_time = np.mean(step_times) if step_times else 0.0

        # Determine episode outcome
        if target_reached:
            outcome = "SUCCESS"
            successes.append(True)
            collisions.append(False)
            timeouts.append(False)
        elif collision:  # Collision (based on your reward of -50)
            outcome = "COLLISION"
            successes.append(False)
            collisions.append(True)
            timeouts.append(False)
        else:  # Timeout
            outcome = "TIMEOUT"
            successes.append(False)
            collisions.append(False)
            timeouts.append(True)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Store detailed episode data
        episode_data.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'outcome': outcome,
            'time': episode_time,
            'avg_step_time': avg_step_time
        })

        # Print episode summary with timing info
        timing_info = f"({avg_step_time*1000:.1f}ms/step)" if real_time else ""
        print(f"Episode {episode + 1:2d}: {outcome:9s} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Steps: {episode_length:3d} | "
              f"Time: {episode_time:.1f}s {timing_info}")

    total_test_time = time.time() - total_test_start_time

    # Calculate statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    success_rate = np.mean(successes) * 100
    collision_rate = np.mean(collisions) * 100
    timeout_rate = np.mean(timeouts) * 100

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Episodes Tested: {n_episodes}")
    print(f"Success Rate: {success_rate:.1f}% ({sum(successes)}/{n_episodes})")
    print(f"Collision Rate: {collision_rate:.1f}% ({sum(collisions)}/{n_episodes})")
    print(f"Timeout Rate: {timeout_rate:.1f}% ({sum(timeouts)}/{n_episodes})")
    print()
    print(f"Average Reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}")
    print(f"Average Length: {episode_lengths.mean():.1f} ± {episode_lengths.std():.1f} steps")
    print(f"Total Test Time: {total_test_time:.1f}s")
    print()
    print(f"Best Episode: {episode_rewards.max():.1f} reward")
    print(f"Worst Episode: {episode_rewards.min():.1f} reward")

    if sum(successes) > 0:
        successful_episodes = [episode_lengths[i] for i in range(n_episodes) if successes[i]]
        print(f"Avg Success Length: {np.mean(successful_episodes):.1f} steps")

    # Real-time statistics
    if real_time:
        rt_stats = rt_runner.get_stats()
        if rt_stats:
            print(f"\n⏱️ Real-Time Statistics:")
            print(f"Target Step Time: {rt_stats['target_step_duration']*1000:.1f}ms")
            print(f"Avg Sleep Time: {rt_stats['avg_sleep_per_step']*1000:.1f}ms/step")
            print(f"Total Steps: {rt_stats['steps']}")

    # Save results to file
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_realtime" if real_time else "_maxspeed"
        results_file = f"test_results_{model_name}_{n_episodes}eps{mode_suffix}_{timestamp}.txt"

        with open(results_file, 'w') as f:
            f.write("GAT+TD3 Test Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Episodes: {n_episodes}\n")
            f.write(f"Max Episode Length: {max_ep_len}\n")
            f.write(f"Test Seed: {seed}\n")
            f.write(f"Real-time Mode: {'Yes' if real_time else 'No'}\n")
            f.write(f"Top-K Attention Edges: {top_k_edges}\n")
            f.write(f"Total Test Time: {total_test_time:.1f}s\n\n")

            f.write("Summary Statistics:\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Collision Rate: {collision_rate:.1f}%\n")
            f.write(f"Timeout Rate: {timeout_rate:.1f}%\n")
            f.write(f"Average Reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}\n")
            f.write(f"Average Length: {episode_lengths.mean():.1f} ± {episode_lengths.std():.1f}\n\n")

            if real_time and rt_runner.get_stats():
                rt_stats = rt_runner.get_stats()
                f.write("Real-time Statistics:\n")
                f.write(f"Target Step Duration: {rt_stats['target_step_duration']*1000:.1f}ms\n")
                f.write(f"Average Sleep Time: {rt_stats['avg_sleep_per_step']*1000:.1f}ms/step\n")
                f.write(f"Total Steps: {rt_stats['steps']}\n\n")

            f.write("Episode Details:\n")
            f.write("Episode | Outcome   | Reward  | Steps | Time   | Avg Step Time\n")
            f.write("-" * 60 + "\n")
            for data in episode_data:
                f.write(f"{data['episode']:7d} | {data['outcome']:9s} | "
                       f"{data['reward']:7.1f} | {data['length']:5d} | "
                       f"{data['time']:6.1f}s | {data['avg_step_time']*1000:8.1f}ms\n")

        print(f"\n💾 Detailed results saved to: {results_file}")

        if collect_cluster_data:
            print(f"\n📊 Fitting clusters on {len(env.cluster_explainer.embedding_buffer)} samples...")
            env.cluster_explainer.fit_clusters()
            env.cluster_explainer.save_model(cluster_model_path)

    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': episode_rewards.mean(),
        'std_reward': episode_rewards.std(),
        'avg_length': episode_lengths.mean(),
        'std_length': episode_lengths.std(),
        'total_time': total_test_time,
        'episode_data': episode_data
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained GAT+TD3 models")
    parser.add_argument("--model-dir", type=str, default="pytorch_models",
                        help="Directory containing saved models")
    parser.add_argument("--model-name", type=str, default="TD3_gat_humans",
                        help="Base name of saved models")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of test episodes")
    parser.add_argument("--max-ep-len", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible testing")
    parser.add_argument("--lidar-nodes", type=int, default=20,
                        help="Number of LiDAR nodes")
    parser.add_argument("--humans", type=int, default=3,
                        help="Number of human nodes")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    parser.add_argument("--max-speed", action="store_true",
                        help="Run at maximum speed (no real-time sync)")
    parser.add_argument("--top-k-edges", type=int, default=3,
                        help="Number of top attention edges to highlight (default: 3)")
    parser.add_argument("--collect-cluster-data", action="store_true",
                    help="Collect embedding data and fit clusters (run once)")
    parser.add_argument("--cluster-model", type=str, default="cluster_model.pkl",
                    help="Path to cluster model file")

    args = parser.parse_args()

    print("🧪 GAT+TD3 Model Testing")
    print(f"Model Directory: {args.model_dir}")
    print(f"Model Name: {args.model_name}")
    print(f"Test Episodes: {args.episodes}")
    print(f"Max Episode Length: {args.max_ep_len}")
    print(f"Random Seed: {args.seed}")
    print(f"Render: {'Yes' if args.render else 'No'}")
    print(f"Real-time: {'No (Max Speed)' if args.max_speed else 'Yes (10 Hz)'}")
    print(f"Top-K Attention Highlighting: {args.top_k_edges}")

    # Run the test
    results = test_agent(
        model_dir=args.model_dir,
        model_name=args.model_name,
        n_episodes=args.episodes,
        max_ep_len=args.max_ep_len,
        render=args.render,
        save_results=not args.no_save,
        lidar_nodes=args.lidar_nodes,
        human_nodes=args.humans,
        seed=args.seed,
        real_time=not args.max_speed,  # Default is real-time unless --max-speed is used
        top_k_edges=args.top_k_edges,
        collect_cluster_data=args.collect_cluster_data,
        cluster_model_path=args.cluster_model
    )

    if results:
        print(f"\n✅ Testing completed successfully!")
        print(f"🎯 Final Success Rate: {results['success_rate']:.1f}%")
        if 'total_time' in results:
            print(f"⏱️ Total Test Duration: {results['total_time']:.1f}s")
    else:
        print(f"\n❌ Testing failed - check model files")

