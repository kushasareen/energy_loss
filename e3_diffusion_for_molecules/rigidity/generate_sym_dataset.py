import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
# from pynauty import Graph, autgrp
import igraph as ig
from collections import defaultdict

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

def build_graph_from_data_sample(dataset, idx):
    num_atoms = dataset["num_atoms"][idx].item()
    node_colors = []
    edge_colors = []
    edge_list = []

    for i in range(dataset["one_hot"][idx].shape[0]):
        atom_type = dataset["one_hot"][idx][i].to(int).argmax()
        if dataset["one_hot"][idx][i].sum() > 0:
            node_colors.append(atom_type.item())

    for i in range(dataset["edges"][idx].shape[0]): # iterate over adjacency matrix
        for j in range(i, dataset["edges"][idx].shape[1]):
            if dataset["edges"][idx][i, j] >= 1:
                edge_list.append((i, j))
                bond_distance = torch.norm(dataset['positions'][idx][i] - dataset['positions'][idx][j])
                edge_colors.append(int(round(bond_distance.item(), 5) * 1e5)) # works but is adjustable?
        
    g = ig.Graph(edge_list)
    print(len(edge_colors), len(set(edge_colors)))
    
    return g, node_colors, edge_colors

def get_symmetric_k_regular_graph(dataset, idx, n, k, size = 29):
    g, node_colors, edge_colors = build_graph_from_data_sample(dataset, idx)
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

def generate_symmetric_dataset(dataset, k = 6, size = 29):
    adj_matrices = []
    for idx in range(len(dataset["num_atoms"])):
        print(f"Generating graph for idx = {idx}")
        n = dataset["num_atoms"][idx]
        adj = get_symmetric_k_regular_graph(dataset, idx, n, k, size)
        adj_matrices.append(adj)

    adj_matrices = torch.stack(adj_matrices)
    return adj_matrices


if __name__ == '__main__':
    # test pyanuty
    # g = Graph(5)
    # g.connect_vertex(0, [1, 2, 3])
    # g.connect_vertex(2, [1, 3, 4])
    # g.connect_vertex(4, [3])

    # print(g)
    # generators, grpsize1, grpsize2, orbits, numorbits = autgrp(g)
    # print("Generators: ", generators)
    # print("Group size: ", grpsize1, grpsize2)
    # print("Orbits: ", orbits)
    # print("Number of orbits: ", numorbits)    
    # breakpoint()

    # test igraph
    # g = ig.Graph.Erdos_Renyi(n = 10, m = 20)
    # print(g.get_automorphisms_vf2())

    data_dummy = pickle.load(open("data_dummy.pickle", "rb"))
    breakpoint()
    adj_matrices = generate_symmetric_dataset(data_dummy, 6)

    