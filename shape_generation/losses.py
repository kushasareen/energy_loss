import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from kabsch import kabsch_torch_batched
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# times = []
# time_index = 0
# TODO load sparse edges
import pickle
with open("k_regular.pickle" , "rb") as f:
    k_regular = pickle.load(f)

#### #REMEMBER TO COMMENT OUT
# with open("/home/mila/k/kusha.sareen/scratch/e3_diffusion/shapes/k_regular_timing.pickle", "rb") as f:
#     k_regular_edges = pickle.load(f)


def compute_energy_loss_with_edges(denoised, ground_truth, coeff_mode, edge_mode='complete', num_vertices=20):
    """"""
    edges = get_energy_edge_list(ground_truth, edge_mode, num_vertices)
    # torch.cuda.synchronize()
    # time_start = time.time()
    coeffs = 1 # placeholder constant coeffs
    dist_denoised = get_pairwise_distance_foqr_edges(denoised, edges)
    dist_ground_truth = get_pairwise_distance_for_edges(ground_truth, edges)
    error_energy = 0.5 * coeffs * (dist_denoised - dist_ground_truth) ** 2
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"Sparse energy loss computation time: {time_end - time_start} seconds")
    # global time_index
    # time_index += 1
    # if time_index >= 3:
    #     times.append(time_end - time_start)
    #     print(f"Average time for energy loss computation: {np.mean(times)} seconds")
    #     standard_error = np.std(times) / np.sqrt(len(times))
    #     print(f"Standard error: {standard_error} seconds")

    # breakpoint()
    return error_energy

def get_energy_edge_list(points, edge_mode='complete', num_vertices=20):
    """
    Returns a list of edges for the given number of vertices and edge mode.
    """
    batch_size, num_points, _ = points.shape
    if edge_mode == 'complete':
        return [(i, j) for i in range(num_vertices) for j in range(num_vertices) if i != j]
    elif edge_mode == 'k_regular':
        # ideally want to sample a random graph, instead just do this for now
        edge_list = k_regular_edges[num_vertices][0]
        # repeat for batch size
        edge_list = np.tile(edge_list, (batch_size, 1, 1))
        edge_list_tensor = torch.tensor(edge_list, device=device)
        return edge_list_tensor
    else:
        raise ValueError(f"Unknown edge mode: {edge_mode}")
    
def get_pairwise_distance_for_edges(points, edge_indices):
    # index points by edges
    B, E = edge_indices.shape[:2]
    batch_arange = torch.arange(B, device=device).view(B, 1).expand(B, E)

    idx_a = edge_indices[:, :, 0]  # (B, E)
    idx_b = edge_indices[:, :, 1]  # (B, E)

    # Expand to match gather requirements
    idx_a_exp = idx_a.unsqueeze(-1).expand(-1, -1, points.shape[2])  # (B, E, D)
    idx_b_exp = idx_b.unsqueeze(-1).expand(-1, -1, points.shape[2])  # (B, E, D)

    # Gather
    point_a = torch.gather(points, dim=1, index=idx_a_exp)  # (B, E, D)
    point_b = torch.gather(points, dim=1, index=idx_b_exp)  # (B, E, D)

    diffs = point_a - point_b
    distances = torch.norm(diffs, dim=-1)
    return distances

def compute_energy_loss(denoised, ground_truth, coeff_mode, edge_mode='complete', num_vertices=20):
    # torch.cuda.synchronize()
    # time_start = time.time()
    edges = get_energy_edges(ground_truth, edge_mode, num_vertices)
    coeffs = get_energy_coeffs(ground_truth, coeff_mode)
    dist_denoised = get_pairwise_distances(denoised)
    dist_ground_truth = get_pairwise_distances(ground_truth)
    error_energy = 0.5 * edges * coeffs * (dist_denoised - dist_ground_truth) ** 2 
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print(f"Energy loss computation time: {time_end - time_start} seconds")
    # global time_index
    # time_index += 1
    # if time_index >= 3:
    #     times.append(time_end - time_start)
    #     print(f"Average time for energy loss computation: {np.mean(times)} seconds")
    #     standard_error = np.std(times) / np.sqrt(len(times))
    #     print(f"Standard error: {standard_error} seconds")

    # breakpoint()
    return error_energy

def get_pairwise_distances(points):
    diffs = points.unsqueeze(2) - points.unsqueeze(1)  # (B, 6, 6, 2)
    distances = torch.norm(diffs, dim=-1)  # (B, 6, 6)
    return distances

def get_energy_edges(points, edge_mode='complete', num_vertices=20):
    batch_size, num_points, _ = points.shape
    if edge_mode == 'complete':
        return torch.ones(batch_size, num_points, num_points, device=device) - torch.eye(num_points, device=device)
    elif edge_mode == 'k_regular':
        num_k_regular = len(k_regular[num_vertices])
        random_batch_indices = torch.randint(0, num_k_regular, (batch_size,))
        edges = k_regular[num_vertices][random_batch_indices]
        edges = edges.to(device)
        return edges
    elif edge_mode == 'hull':
        num_hull = len(hull[num_vertices])
        random_batch_indices = torch.randint(0, num_hull, (batch_size,))
        edges = hull[num_vertices][random_batch_indices]
        edges = edges.to(device)
        raise NotImplementedError("Hull edges not implemented yet")
        return edges


def get_energy_coeffs(points, coeff_mode):
    if coeff_mode == "constant":
        return torch.ones_like(get_energy_edges(points))
    elif coeff_mode == "exp_dist":
        distances = get_pairwise_distances(points)
        coeffs = torch.exp(-distances)
        return 2.6 * coeffs

def compute_loss(denoised, data, loss_mode, coeff_mode, criterion = nn.MSELoss(), base_loss_mode = None,  num_vertices=20, edge_mode='complete'):

    if loss_mode == "mse":
        # torch.cuda.synchronize()
        # time_start = time.time()
        loss = criterion(denoised, data)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # global time_index
        # time_index += 1
        # print(f"MSE loss computation time: {time_end - time_start} seconds")
        # if time_index >= 3:
        #     times.append(time_end - time_start)
        #     print(f"Average time for MSE loss computation: {np.mean(times)} seconds")
        #     standard_error = np.std(times) / np.sqrt(len(times))
        #     print(f"Standard error: {standard_error} seconds")

        # breakpoint()

    elif loss_mode == "energy":
        loss = compute_energy_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode, edge_mode, num_vertices).mean() # avg over ,batch
    elif loss_mode == "energy_sparse":
        loss = compute_energy_loss_with_edges(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode, edge_mode, num_vertices).mean() # avg over batch
    elif loss_mode == "hybrid":
        energy = compute_energy_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), coeff_mode).mean((1,2)) # Shape: (B) # no grads through energy
        mse_loss = (denoised - data).pow(2).mean((1)) # Shape: (B)
        loss = energy.detach() * mse_loss # Shape: (B)
        loss = loss.mean() # Shape: (1)
    elif loss_mode == "fape":
        # torch.cuda.synchronize()
        # time_start = time.time()
        loss = compute_fape_loss(denoised.view(-1, num_vertices, 2), data.view(-1, num_vertices, 2), criterion) # Shape: (1) # TODO: Check
        # torch.cuda.synchronize()
        # time_end = time.time()
        # breakpoint()
        # time_index += 1
        # print(f"MSE loss computation time: {time_end - time_start} seconds")
        # if time_index >= 3:
        #     times.append(time_end - time_start)
        #     print(f"Average time for MSE loss computation: {np.mean(times)} seconds")
        #     standard_error = np.std(times) / np.sqrt(len(times))
        #     print(f"Standard error: {standard_error} seconds")

    elif loss_mode == "hybrid_bond":
        denoised = denoised.view(-1, num_vertices, 2)
        data = data.view(-1, num_vertices, 2)
        energy = compute_energy_loss(denoised, data, coeff_mode).mean((2)) # Shape: (B, N) # no grads through energy

        if base_loss_mode == "mse":
            base_loss = (denoised - data).pow(2).mean(2) # Shape: (B, N)
        elif base_loss_mode == "mae":
            base_loss = (denoised - data).abs().mean(2)
        elif base_loss_mode == "huber":
            base_loss = nn.SmoothL1Loss()(denoised, data).mean(2)

        full_loss = energy.detach() * base_loss # Shape: (B, N)
        loss = full_loss.mean() # Shape: (1)
    return loss

def compute_fape_loss(denoised, ground_truth, criterion = nn.MSELoss()):
    ground_truth_rotated, _, _ = kabsch_torch_batched(ground_truth.detach(), denoised.detach())
    fape_loss = criterion(ground_truth_rotated, denoised) # Shape: (1)

    return fape_loss