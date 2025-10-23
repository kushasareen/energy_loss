import numpy as np
import getpass
import os
import torch
import argparse
from qm9 import bond_analyze

# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#radient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


# Other utilities
def get_wandb_username(username):
    if username == 'cvignac':
        return 'cvignac'
    current_user = getpass.getuser()
    if current_user == 'victor' or current_user == 'garciasa':
        return 'vgsatorras'
    else:
        return username


if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)

def pick_best_hyperparams(args, hyperparam_file='best_hyperparams.yaml'):
    import yaml
    with open(hyperparam_file) as file:
        all_hyperparams = yaml.load(file, Loader=yaml.FullLoader)

    hyperparams = all_hyperparams[args.model][args.loss_mode]
    if args.loss_mode == "energy":
        args.lr = float(hyperparams[args.energy_coeff_mode]['lr'])
        args.energy_loss_weight = float(hyperparams[args.energy_coeff_mode]['ew'])
    else:
        args.lr = float(hyperparams['lr'])
        args.energy_loss_weight = float(hyperparams['ew'])

    print(f'Picked hyperparameters: {args.lr}, {args.energy_loss_weight}')
    return args

def get_unique_name(args):
    return f'{args.model}_{args.loss_mode}_{args.energy_coeff_mode}_s{args.seed}_fr{args.dataset_frac}_aug_{args.data_augmentation}'

def get_molecule_bonds(positions, one_hot, dataset_info, num_atoms, dataset= 'qm9', debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    bonds = np.zeros((len(x), len(x)), dtype='int')

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom_i = one_hot[i].to(torch.long).argmax().item()
            atom_j = one_hot[j].to(torch.long).argmax().item()
            atom1, atom2 = atom_decoder[atom_i], atom_decoder[atom_j]
            if dataset == 'qm9':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset == 'geom':
                order = bond_analyze.geom_predictor((atom1, atom2), dist)
            bonds[i, j] = order
            bonds[j, i] = order

    return bonds

def add_bonds(all_positions, one_hot, dataset_info, all_num_atoms, debug=False):
    import time
    # make edges a torch tensor
    all_edges = []
    print("Adding bonds...")
    start = time.time()
    idx = 0

    for positions, atom_type, num_atoms in zip(all_positions, one_hot, all_num_atoms):
        idx += 1
        print(idx)
        edges = get_molecule_bonds(positions, atom_type, dataset_info, num_atoms, debug)
        all_edges.append(edges)

    out = torch.tensor(all_edges)
    print(f"Time taken to add bonds: {time.time() - start}")
    breakpoint()

    return out

def add_k_regular(all_positions, all_n_nodes):
    x = len(all_positions[0, :, 0])
    for n_nodes in all_n_nodes:
        pass # sample graph here