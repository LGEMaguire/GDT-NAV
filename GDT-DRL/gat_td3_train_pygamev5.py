import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

from replay_buffer import ReplayBuffer
from pygame_env_rosv5 import PygameTwinEnv, TIME_DELTA
from graph_utils import create_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Models ---------

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
        s = F.relu(self.l1(s)); s = F.relu(self.l2(s))
        return self.tanh(self.l3(s))

class Critic(nn.Module):
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
        x = F.relu(self.c1(sa)); x = F.relu(self.c2(x)); return self.c3(x)

    def Q2(self, sa):
        x = F.relu(self.c4(sa)); x = F.relu(self.c5(x)); return self.c6(x)

class GATTD3:
    def __init__(self, lidar_nodes=20, robot_nodes=1, human_nodes=3,
                 lidar_channels=4, robot_channels=4, human_channels=6,
                 gat_out_channels=16, num_heads=4, action_dim=2, max_action=1.0,
                 state_dim=102):  # Added state_dim parameter
        self.L = lidar_nodes
        self.R = robot_nodes
        self.H = human_nodes
        self.total_nodes = self.L + self.R + self.H
        self.train_step = 0

        # Create GAT with mixed dimensions
        self.gat = GATNetwork(
            lidar_channels, robot_channels, human_channels,
            gat_out_channels, num_heads, lidar_nodes, robot_nodes, human_nodes
        ).to(device)
        self.lidar_channels = lidar_channels
        self.robot_channels = robot_channels
        self.human_channels = human_channels
        self.gat_t = GATNetwork(
            lidar_channels, robot_channels, human_channels,
            gat_out_channels, num_heads, lidar_nodes, robot_nodes, human_nodes
        ).to(device)
        self.gat_t.load_state_dict(self.gat.state_dict())

        input_dim = gat_out_channels * self.total_nodes
        self.actor = Actor(input_dim, action_dim).to(device)
        self.actor_t = Actor(input_dim, action_dim).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = Critic(input_dim, action_dim).to(device)
        self.critic_t = Critic(input_dim, action_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.g_opt = torch.optim.Adam(self.gat.parameters(), lr=1e-4)
        self.max_action = max_action
        self.edge_index = create_edge_index(self.L, self.R, self.H).to(device)

        # ============ VALIDATION BUFFER & EARLY STOPPING ============
        self.val_buffer = ReplayBuffer(buffer_size=50000, random_seed=123)
        self.gat_best_val_loss = float('inf')
        self.gat_val_patience_counter = 0
        self.gat_val_patience = 1000  # Number of validation checks without improvement
        self.gat_val_min_delta = 1e-5  # Minimum improvement threshold
        self.gat_frozen = False  # Flag to stop GAT updates
        self.val_check_freq = 100  # Check validation every N training iterations
        # ============================================================

        with open("qvalues.log", "w") as f:
            f.write("step,avg_q1,avg_q2,avg_target,critic_loss\n")

        # New log file for GAT validation
        with open("gat_validation.log", "w") as f:
            f.write("train_step,val_loss,patience_counter,frozen\n")

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
        # Convert flattened state back to node features
        lidar_nodes, robot_nodes, human_nodes = self._flat_to_nodes(flat_state)
        z = self.gat(lidar_nodes, robot_nodes, human_nodes, self.edge_index).view(1, -1)
        return self.actor(z).cpu().numpy().flatten()

    def evaluate_gat_validation(self, batch_size=128, discount=0.99):
        """Compute GAT performance on held-out validation data"""
        if self.val_buffer.size() < batch_size:
            return float('inf')

        # Sample from validation buffer (not training buffer)
        s, a, r, d, s2 = [torch.tensor(x, dtype=torch.float32, device=device) 
                          for x in self.val_buffer.sample_batch(batch_size)]

        with torch.no_grad():  # No gradients for validation
            z1 = self._batch_gat_forward(s, use_target=False)
            z2 = self._batch_gat_forward(s2, use_target=True)

            # Compute how well GAT embeddings predict Q-values
            a2 = self.actor_t(z2)
            q1_val, q2_val = self.critic(z1, a)
            tq1, tq2 = self.critic_t(z2, a2)

            target = r + (1.0 - d) * discount * torch.min(tq1, tq2)
            val_loss = (F.mse_loss(q1_val, target) + F.mse_loss(q2_val, target)) / 2

        return val_loss.item()

    def train_td3(self, rb, iterations, batch_size=128, discount=0.99, tau=0.001,
                  policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if rb.size() < batch_size:
            return

        for it in range(iterations):
            s, a, r, d, s2 = [torch.tensor(x, dtype=torch.float32, device=device) for x in rb.sample_batch(batch_size)]

            # **VECTORIZED**: Process entire batch at once
            z1 = self._batch_gat_forward(s, use_target=False)

            with torch.no_grad():
                z2_t = self._batch_gat_forward(s2, use_target=True)
                a2 = self.actor_t(z2_t)
                noise = (torch.randn_like(a2) * policy_noise).clamp(-noise_clip, noise_clip)
                a2 = (a2 + noise).clamp(-self.max_action, self.max_action)
                tq1, tq2 = self.critic_t(z2_t, a2)
                y = r + (1.0 - d) * discount * torch.min(tq1, tq2)

            # Critic/GAT update
            q1, q2 = self.critic(z1, a)
            loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            self.c_opt.zero_grad()
            if not self.gat_frozen:
                self.g_opt.zero_grad()

            loss_c.backward()
            self.c_opt.step()

            # Only update GAT if not frozen
            if not self.gat_frozen:
                self.g_opt.step()

            # ============ VALIDATION CHECK FOR GAT ============
            if self.train_step % self.val_check_freq == 0 and not self.gat_frozen:
                val_loss = self.evaluate_gat_validation(batch_size, discount)

                # Check if validation loss improved
                if self.gat_best_val_loss - val_loss > self.gat_val_min_delta:
                    self.gat_best_val_loss = val_loss
                    self.gat_val_patience_counter = 0
                else:
                    self.gat_val_patience_counter += 1

                # Log validation metrics
                with open("gat_validation.log", "a") as f:
                    f.write(f"{self.train_step},{val_loss:.6f},{self.gat_val_patience_counter},{int(self.gat_frozen)}\n")

                # Freeze GAT if patience exhausted
                if self.gat_val_patience_counter >= self.gat_val_patience:
                    self.gat_frozen = True
                    print(f"\n{'='*70}")
                    print(f"[GAT EARLY STOPPING] GAT training frozen at step {self.train_step}")
                    print(f"[GAT EARLY STOPPING] Best validation loss: {self.gat_best_val_loss:.6f}")
                    print(f"[GAT EARLY STOPPING] GAT will remain frozen, TD3 continues training")
                    print(f"{'='*70}\n")

                    # Log to file as well
                    with open("gat_validation.log", "a") as f:
                        f.write(f"# GAT FROZEN at step {self.train_step}, best_val_loss={self.gat_best_val_loss:.6f}\n")
            # ==================================================

            if self.train_step % 100 == 0:
                avg_q1 = q1.mean().item()
                avg_q2 = q2.mean().item()
                avg_target = y.mean().item()
                critic_loss = loss_c.item()
                with open("qvalues.log", "a") as f:
                    f.write(f"{self.train_step},{avg_q1:.4f},{avg_q2:.4f},{avg_target:.4f},{critic_loss:.4f}\n")

            self.train_step += 1

            # Delayed actor update
            if it % policy_freq == 0:
                a_cur = self.actor(z1.detach())
                sa = torch.cat([z1.detach(), a_cur], dim=1)
                loss_a = -self.critic.Q1(sa).mean()

                self.a_opt.zero_grad()
                loss_a.backward()
                self.a_opt.step()

                # Target network updates
                with torch.no_grad():
                    for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)
                    for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)
                    # Only update GAT target if GAT is not frozen
                    for p, tp in zip(self.gat.parameters(), self.gat_t.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)

    def _batch_gat_forward(self, batch_states, use_target=False):
        """Vectorized GAT forward pass for entire batch"""
        B = batch_states.shape[0]

        # Vectorized state unpacking
        lidar_size = self.L * self.lidar_channels
        robot_size = self.R * self.robot_channels
        human_size = self.H * self.human_channels

        # Reshape all at once (much faster)
        batch_lidar = batch_states[:, :lidar_size].view(B, self.L, self.lidar_channels)
        batch_robot = batch_states[:, lidar_size:lidar_size + robot_size].view(B, self.R, self.robot_channels)
        batch_human = batch_states[:, lidar_size + robot_size:lidar_size + robot_size + human_size].view(B, self.H, self.human_channels)

        # Choose network
        gat_net = self.gat_t if use_target else self.gat

        # Process entire batch through GAT
        batch_outputs = []
        for i in range(B):
            z_i = gat_net(batch_lidar[i], batch_robot[i], batch_human[i], self.edge_index)
            batch_outputs.append(z_i.view(1, -1))

        return torch.cat(batch_outputs, dim=0)

    def save(self, name, outdir="pytorch_models"):
        os.makedirs(outdir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(outdir, f"{name}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(outdir, f"{name}_critic.pth"))
        torch.save(self.gat.state_dict(), os.path.join(outdir, f"{name}_gat.pth"))
        torch.save(self.gat_t.state_dict(), os.path.join(outdir, f"{name}_gat_target.pth"))
        torch.save(self.actor_t.state_dict(), os.path.join(outdir, f"{name}_actor_target.pth"))
        torch.save(self.critic_t.state_dict(), os.path.join(outdir, f"{name}_critic_target.pth"))

# --------- Training Loop ---------

def train(
    seed=0,
    lidar_nodes=20,
    robot_nodes=1,
    human_nodes=3,
    in_channels=4,
    max_steps=int(5e6),
    eval_every=int(1e5),
    max_ep_len=500,
    batch_size=128,
    expl_noise_start=0.1,
    expl_noise_min=0.02,
    expl_decay_steps=100000,
    log_path="rewards.log",
    val_split=0.15,  # NEW: Percentage of data for validation
    render=False
):
    np.random.seed(seed); torch.manual_seed(seed)

    env = PygameTwinEnv(lidar_nodes=lidar_nodes, max_humans=human_nodes, seed=seed, render=render)

    # Calculate state dimension
    state_dim = lidar_nodes * 4 + robot_nodes * 4 + human_nodes * 6
    action_dim = 2

    agent = GATTD3(
        lidar_nodes=lidar_nodes, robot_nodes=robot_nodes, human_nodes=human_nodes,
        lidar_channels=4, robot_channels=4, human_channels=6,
        state_dim=state_dim, action_dim=action_dim
    )

    # Training buffer
    rb = ReplayBuffer(int(1e6), random_seed=seed)


    if os.path.exists(log_path):
        os.remove(log_path)

    def log_reward(r, t):
        with open(log_path, "a") as f:
            f.write(f"{t},{r}\n")

    t = 0; ep_steps = 0; ep = 0; ep_ret = 0.0
    noise = expl_noise_start
    state = env.reset()

    def evaluate(n_episodes=5):
        tot = 0.0; cols = 0
        for _ in range(n_episodes):
            s = env.reset(); done = False; steps = 0
            while not done and steps < max_ep_len:
                a = agent.act(s)
                a_in = np.array([(a[0]+1)/2, a[1]], dtype=np.float32)
                s, r, done, _, _ = env.step(a_in)
                tot += r; steps += 1
                if r < -90: cols += 1
        return tot / n_episodes, cols / n_episodes

    print(f"[INFO] Validation split: {val_split*100:.1f}% of experiences go to validation buffer")
    print(f"[INFO] GAT will be frozen after {agent.gat_val_patience} validation checks without improvement\n")

    while t < max_steps:
        if env.done or ep_steps >= max_ep_len:
            print(f"[Episode {ep}] steps={ep_steps} return={ep_ret:.2f} train_buffer={rb.size()} val_buffer={agent.val_buffer.size()} GAT_frozen={agent.gat_frozen}")

            if t > 0 and rb.size() >= batch_size:
                agent.train_td3(rb, iterations=4, batch_size=batch_size)

            if (t // eval_every) != ((t - ep_steps) // eval_every):
                avg_r, avg_c = evaluate()
                print(f"[Eval @ {t}] avg_reward={avg_r:.3f}, avg_col={avg_c:.3f}")
                agent.save("TD3_gat_humans")

            ep += 1
            state = env.reset(); ep_steps = 0; ep_ret = 0.0

        # exploration anneal
        if rb.size() >= batch_size:
            noise = max(expl_noise_min, expl_noise_start - (expl_noise_start - expl_noise_min) * (t / float(expl_decay_steps)))
        else:
            noise = expl_noise_start

        a = agent.act(state)
        a = (a + np.random.normal(0, noise, size=2)).clip(-1, 1)
        a_in = np.array([(a[0]+1)/2, a[1]], dtype=np.float32)

        next_state, reward, done, target, collision = env.step(a_in)
        log_reward(reward, t)
        done_bool = 0.0 if (ep_steps + 1) == max_ep_len else float(done)

        # ============ SPLIT DATA BETWEEN TRAIN AND VALIDATION ============
        if np.random.random() < val_split:
            # Add to validation buffer
            agent.val_buffer.add(state, a, reward, done_bool, next_state)
        else:
            # Add to training buffer
            rb.add(state, a, reward, done_bool, next_state)
        # ==================================================================

        state = next_state
        ep_steps += 1
        ep_ret += reward
        t += 1

        if render: env.render()

    avg_r, avg_c = evaluate()
    agent.save("TD3_gat_humans", outdir="models")
    return avg_r

if __name__ == "__main__":
    import argparse, sys
    try:
        import torch_geometric  # noqa: F401
    except Exception as e:
        print("[Dependency] torch_geometric not found. Please install it (and PyTorch) before running:", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train TD3+GAT in the Pygame Digital Twin with GAT Early Stopping")
    parser.add_argument("--steps", type=int, default=int(5e6), help="Total environment steps")
    parser.add_argument("--eval-every", type=int, default=int(1e5), help="Eval frequency in steps")
    parser.add_argument("--max-ep-len", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--humans", type=int, default=3, help="Human nodes capacity")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation buffer split (0.0-1.0)")
    parser.add_argument("--render", action="store_true", help="Enable Pygame render window")

    args = parser.parse_args()

    print("[Config] steps=", args.steps, " eval_every=", args.eval_every, " max_ep_len=", args.max_ep_len,
          " batch=", args.batch, " humans=", args.humans, " val_split=", args.val_split, " render=", args.render)

    avg_r = train(
        human_nodes=args.humans,
        max_steps=args.steps,
        eval_every=args.eval_every,
        max_ep_len=args.max_ep_len,
        batch_size=args.batch,
        val_split=args.val_split,
        render=args.render
    )

    print("[Done] Final average reward:", avg_r)
