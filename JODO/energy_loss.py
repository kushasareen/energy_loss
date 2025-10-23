import torch

def compute_energy_loss(config, pred, data, node_mask, edge_mask):
    energy_weight = config.training.energy_weight
    coeff_mode = config.training.coeff_mode
    edges = get_energy_edges(data, edge_mask)
    coeffs = get_energy_coeffs(data, node_mask, mode=coeff_mode)
    energy_error = get_energy_error(pred, data, edges, coeffs)

    return energy_weight * energy_error

def get_energy_edges(data, edge_mask):
    # for now only support full edges
    n_nodes = data.size(1)
    adj_matrix = edge_mask.reshape(-1, n_nodes, n_nodes) # Shape: (B, N, N)
    return adj_matrix

def get_energy_coeffs(data, node_mask, mode='exp'):
    n_nodes = data.size(1)
    num_atoms = node_mask.sum(dim=1).squeeze()  # Shape: (B,)
    const = 1 / num_atoms # Shape [B]
    eps = 1e-5
    if mode == 'constant':
        matrix = torch.ones(data.size(0), n_nodes, n_nodes, device=data.device)  # Shape [B, N, N]
    elif mode == 'inv_dist':
        dist = torch.norm(data.unsqueeze(2) - data.unsqueeze(1), dim=-1)  # Shape: (B, N, N)
        k0 = 3.7 # manually set coeff so on the order of 1
        matrix = k0 / (dist + eps)
    elif mode == 'inv_dist2':
        dist = torch.norm(data.unsqueeze(2) - data.unsqueeze(1), dim=-1) # Shape: (B, N, N)
        k0 = 9 # manually set coeff so on the order of 1
        matrix = k0 / (dist ** 2 + eps)
    elif mode == 'exp':
        k0 = 24
        dist = torch.norm(data.unsqueeze(2) - data.unsqueeze(1), dim=-1) # Shape: (B, N, N)
        matrix = k0*torch.exp(-dist)

    # breakpoint() # TODO: Check matrix.mean() and set coeff accordingly
    return torch.einsum('b, bnm -> bnm', const, matrix)

def get_energy_error(data, pred, adj_matrix, k):
    data_diff = data.unsqueeze(2) - data.unsqueeze(1)  # Shape: (B, N, N, 3)
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)  # Shape: (B, N, N, 3)

    # Compute pairwise distances
    dist = torch.norm(data_diff, dim=-1)  # Shape: (B, N, N)
    dist_hat = torch.norm(pred_diff, dim=-1)  # Shape: (B, N, N)

    # Difference in distances
    diff = dist - dist_hat  # Shape: (B, N, N)

    energy_per_pair = 0.5 * adj_matrix * k * diff**2  # Shape: (B, N, N), element-wise multiplication

    return energy_per_pair
    