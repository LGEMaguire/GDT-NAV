
import torch

def create_edge_index(lidar_nodes: int, robot_nodes: int, human_nodes: int):
    """
    Build a directed edge_index for the graph:
      - LiDAR <-> LiDAR: fully connected (excluding self-loops)
      - Robot <-> (LiDAR + Humans): bi-directional
      - Humans <-> Humans: fully connected (excluding self-loops)
      - Humans <-> LiDAR: bi-directional
    Node indexing:
      0 .. L-1                   -> LiDAR
      L .. L+R-1                 -> Robot
      L+R .. L+R+H-1             -> Humans
    """
    L, R, H = lidar_nodes, robot_nodes, human_nodes
    total = L + R + H
    edges = []

    # LiDAR <-> LiDAR
    for i in range(L):
        for j in range(L):
            if i != j:
                edges.append([i, j])

    # Humans <-> Humans
    for i in range(H):
        u = L + R + i
        for j in range(H):
            v = L + R + j
            if i != j:
                edges.append([u, v])

    # Robot <-> LiDAR and Humans
    for i in range(R):
        r = L + i
        # Robot <-> LiDAR
        for j in range(L):
            edges.append([r, j])
            edges.append([j, r])
        # Robot <-> Humans
        for k in range(H):
            h = L + R + k
            edges.append([r, h])
            edges.append([h, r])

    # Humans <-> LiDAR (optional but useful)
    for k in range(H):
        h = L + R + k
        for j in range(L):
            edges.append([h, j])
            edges.append([j, h])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()
