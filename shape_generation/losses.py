import torch
import torch.nn as nn
import numpy as np
from kabsch import kabsch_torch_batched
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("k_regular.pickle", "rb") as f:
    k_regular = pickle.load(f)


def compute_energy_loss_with_edges(denoised, ground_truth, coeff_mode, edge_mode='complete', num_vertices=20):
    """Compute energy loss using a sparse edge list representation."""
    edges = get_energy_edge_list(ground_truth, edge_mode, num_vertices)
    coeffs = 1  # placeholder constant coeffs
    dist_denoised = get_pairwise_distance_for_edges(denoised, edges)
    dist_ground_truth = get_pairwise_distance_for_edges(ground_truth, edges)
    error_energy = 0.5 * coeffs * (dist_denoised - dist_ground_truth) ** 2
    return error_energy

def get_energy_edge_list(points, edge_mode='complete', num_vertices=20):
    """Returns an edge list for the given number of vertices and edge mode."""
    batch_size, num_points, _ = points.shape
    if edge_mode == 'complete':
        return [(i, j) for i in range(num_vertices) for j in range(num_vertices) if i != j]
    elif edge_mode == 'k_regular':
        edge_list = k_regular[num_vertices][0]
        edge_list = np.tile(edge_list, (batch_size, 1, 1))
        edge_list_tensor = torch.tensor(edge_list, device=device)
        return edge_list_tensor
    else:
        raise ValueError(f"Unknown edge mode: {edge_mode}")

def get_pairwise_distance_for_edges(points, edge_indices):
    """Compute pairwise distances for a sparse edge list."""
    B, E = edge_indices.shape[:2]

    idx_a = edge_indices[:, :, 0]  # (B, E)
    idx_b = edge_indices[:, :, 1]  # (B, E)

    idx_a_exp = idx_a.unsqueeze(-1).expand(-1, -1, points.shape[2])  # (B, E, D)
    idx_b_exp = idx_b.unsqueeze(-1).expand(-1, -1, points.shape[2])  # (B, E, D)

    point_a = torch.gather(points, dim=1, index=idx_a_exp)  # (B, E, D)
    point_b = torch.gather(points, dim=1, index=idx_b_exp)  # (B, E, D)

    diffs = point_a - point_b
    distances = torch.norm(diffs, dim=-1)
    return distances

def compute_energy_loss(denoised, ground_truth, coeff_mode, edge_mode='complete', num_vertices=20):
    """Compute energy loss using a dense adjacency matrix representation."""
    edges = get_energy_edges(ground_truth, edge_mode, num_vertices)
    coeffs = get_energy_coeffs(ground_truth, coeff_mode)
    dist_denoised = get_pairwise_distances(denoised)
    dist_ground_truth = get_pairwise_distances(ground_truth)
    error_energy = 0.5 * edges * coeffs * (dist_denoised - dist_ground_truth) ** 2
    return error_energy

def get_pairwise_distances(points):
    """Compute all-pairs pairwise distances."""
    diffs = points.unsqueeze(2) - points.unsqueeze(1)  # (B, N, N, D)
    distances = torch.norm(diffs, dim=-1)  # (B, N, N)
    return distances

def get_energy_edges(points, edge_mode='complete', num_vertices=20):
    """Return a binary adjacency matrix for the given edge mode."""
    batch_size, num_points, _ = points.shape
    if edge_mode == 'complete':
        return torch.ones(batch_size, num_points, num_points, device=device) - torch.eye(num_points, device=device)
    elif edge_mode == 'k_regular':
        num_k_regular = len(k_regular[num_vertices])
        random_batch_indices = torch.randint(0, num_k_regular, (batch_size,))
        edges = k_regular[num_vertices][random_batch_indices]
        return edges.to(device)
    elif edge_mode == 'hull':
        raise NotImplementedError("Hull edges not implemented yet")

def get_energy_coeffs(points, coeff_mode):
    """Return per-edge spring coefficients."""
    if coeff_mode == "constant":
        return torch.ones_like(get_energy_edges(points))
    elif coeff_mode == "exp_dist":
        distances = get_pairwise_distances(points)
        coeffs = torch.exp(-distances)
        return 2.6 * coeffs

def compute_loss(denoised, data, loss_mode, coeff_mode, criterion=nn.MSELoss(), base_loss_mode=None, num_vertices=20, edge_mode='complete'):
    if loss_mode == "mse":
        loss = criterion(denoised, data)

    elif loss_mode == "energy":
        loss = compute_energy_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode, edge_mode, num_vertices).mean()

    elif loss_mode == "energy_sparse":
        loss = compute_energy_loss_with_edges(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode, edge_mode, num_vertices).mean()

    elif loss_mode == "hybrid":
        energy = compute_energy_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode).mean((1, 2))  # (B,)
        mse_loss = (denoised - data).pow(2).mean((1))  # (B,)
        loss = energy.detach() * mse_loss
        loss = loss.mean()

    elif loss_mode == "fape":
        loss = compute_fape_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), criterion)

    elif loss_mode == "hybrid_bond":
        denoised = denoised.view(-1, num_vertices, 2)
        data = data.view(-1, num_vertices, 2)
        energy = compute_energy_loss(denoised, data, coeff_mode).mean((2))  # (B, N)

        if base_loss_mode == "mse":
            base_loss = (denoised - data).pow(2).mean(2)
        elif base_loss_mode == "mae":
            base_loss = (denoised - data).abs().mean(2)
        elif base_loss_mode == "huber":
            base_loss = nn.SmoothL1Loss()(denoised, data).mean(2)

        full_loss = energy.detach() * base_loss  # (B, N)
        loss = full_loss.mean()

    return loss

def compute_fape_loss(denoised, ground_truth, criterion=nn.MSELoss()):
    ground_truth_rotated, _, _ = kabsch_torch_batched(ground_truth.detach(), denoised.detach())
    fape_loss = criterion(ground_truth_rotated, denoised)
    return fape_loss
