import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
# from pynauty import Graph, autgrp
import igraph as ig
from collections import defaultdict
from utils import get_molecule_bonds
from rigidity.generate_dataset import random_k_regular_graph_to_adj
import math

def get_orbits(automorphisms, num_nodes): # works
    # Create a dictionary to store the orbit of each node
    orbit_map = defaultdict(set)

    # For each automorphism, update the orbit map
    for automorphism in automorphisms:
        for i, j in enumerate(automorphism):
            orbit_map[i].add(j)
            orbit_map[j].add(i)

    # Create a list of sets for unique orbits
    visited = set()
    orbits = []
    
    for node in range(num_nodes):
        if node not in visited:
            # Use a set to find all connected nodes in the orbit
            orbit = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    orbit.add(current)
                    stack.extend(orbit_map[current])
            orbits.append(orbit)

    # Map nodes to their orbit index
    orbit_indices = [-1] * num_nodes
    for index, orbit in enumerate(orbits):
        for node in orbit:
            orbit_indices[node] = index

    return orbit_indices

def build_graph_from_data_sample(sample, dataset_info=None):
    node_colors = []
    edge_colors = []
    edge_list = []
    num_atoms = sample.shape[0]
    positions = sample[:, -3:]

    atom_types = torch.from_numpy(sample[:, 0].astype(int)[:, None])
    node_colors = atom_types.squeeze().tolist()

    atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
    atom_types = torch.from_numpy(sample[:, 0].astype(int)[:, None])
    one_hot = atom_types == atomic_number_list

    # next get edges
    edges = get_molecule_bonds(positions, one_hot, dataset_info, num_atoms=num_atoms, dataset='geom')

    for i in range(edges.shape[0]): # iterate over adjacency matrix
        for j in range(i, edges.shape[1]):
            if edges[i, j] >= 1:
                edge_list.append((i, j))
                bond_distance = np.linalg.norm(positions[i] - positions[j])
                edge_colors.append(int(round(bond_distance.item(), 5) * 1e5)) # works but is adjustable?
        
    g = ig.Graph(edge_list)
    print(len(edge_colors), len(set(edge_colors)))
    
    return g, node_colors, edge_colors

def get_symmetric_k_regular_graph(sample, n, k, size, dataset_info=None):
    g, node_colors, edge_colors = build_graph_from_data_sample(sample, dataset_info)
    automorphisms = g.get_automorphisms_vf2(color = node_colors, edge_color = edge_colors)

    # first get the orbits from the automorphisms
    orbits = get_orbits(automorphisms, n) # Shape (n,), indices of the orbits of each node

    # for each orbit, add k random edges to the adjacency matrix
    adj = torch.zeros((size, size))
    for i in range(max(orbits) + 1):
        node = orbits.index(i)
        prev_neighbors = [node]
        for _ in range(k):
            possible_neighbors = [j for j in range(n) if j not in prev_neighbors]
            if len(possible_neighbors) == 0:
                break
            out = np.random.choice(possible_neighbors)
            adj[node, out] = 1
            adj[out, node] = 1
            prev_neighbors.append(out)
            

    sym_adj = torch.zeros((size, size))
    # symmetrize the adjacency matrix using the automorphisms
    for automorphism in automorphisms:
        padded_auto = automorphism + [i for i in range(n, size)]
        # inv_auto = [padded_auto.index(i) for i in range(size)]
        sym_adj += adj[padded_auto][:, padded_auto]

    sym_adj = (sym_adj > 0).float()

    # sanity check
    for automorphism in automorphisms:
        padded_auto = automorphism + [i for i in range(n, size)]
        # inv_auto = [padded_auto.index(i) for i in range(size)]
        try:
            assert (sym_adj == sym_adj[padded_auto][:, padded_auto]).all()
        except:
            print("Error")
            breakpoint()

    return sym_adj

def generate_symmetric_dataset(data_list, k = 6, dataset_info=None, split = 'train', idx = -1):
    adj_matrices = []
    
    if split == 'train':
        num_data_points = len(data_list)
        if idx >= 0:
            # split the data_list into 10 chunks, the last chunk should go to the end
            chunk_size = math.ceil(len(data_list) / 10)
            start = idx * chunk_size
            end = start + chunk_size
            if end > len(data_list):
                end = len(data_list)
            data_list = data_list[start:end]
        else:
            start = 0
            end = len(data_list)
    else:
        start = 0
        end = len(data_list)

    for idx in range(len(data_list)):
        print(f"Generating graph for idx = {idx} of {len(data_list)}")
        sample = data_list[idx]
        num_atoms = len(sample)
        try:
            adj = get_symmetric_k_regular_graph(sample, num_atoms, k, size=num_atoms, dataset_info=dataset_info) # check size
        except Exception as e:
            print(f"Error generating graph for idx = {idx}: {e}")
            # if problem just generate a random k regular graph
            adj = random_k_regular_graph_to_adj(num_atoms, k, size=num_atoms)
        adj_matrices.append(adj)

    # save the adjacency matrices in a pickle file
    if split != 'train':
        path = f'adj_matrices_{split}.pickle'
    else:
        path = f'adj_matrices_{split}_{idx}.pickle'

    with open(path, "wb") as f:
        pickle.dump(adj_matrices, f)

    return adj_matrices


def generate_k_regular_graphs(num_graphs, n_range, k):
    adj_matrices = []
    for n in n_range:
        adj_n = []
        print(f"Generating {num_graphs} k-regular graphs for n = {n} and k = {k}")
        for _ in range(num_graphs):
            adj = random_k_regular_graph_to_adj(n, k, size = n)
            adj_n.append(adj)
        adj_matrices.append(adj_n)
    return adj_matrices

if __name__ == '__main__':

    # data_dummy = pickle.load(open("data_dummy.pickle", "rb"))
    # adj_matrices = generate_symmetric_dataset(data_dummy, 6)

    adj = generate_k_regular_graphs(100, range(5, 21), 12)