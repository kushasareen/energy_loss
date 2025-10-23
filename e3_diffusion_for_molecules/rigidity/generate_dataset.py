import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

def random_k_regular_graph_to_adj(n, k, size = 29):
    k = min(k, n-1)
    G = nx.random_regular_graph(k, n)

    adj = torch.zeros(size, size)
    for edge in G.edges():
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1

    return adj

def generate_dataset(n_range, k, size = 29, datapoints_per_n = 1000):
    dataset = {}
    for n in n_range:
        dataset[n] = []
        print(f"Generating dataset for n = {n}")
        for _ in range(datapoints_per_n):
            dataset[n].append(random_k_regular_graph_to_adj(n, k, size))

        dataset[n] = torch.stack(dataset[n])

    return dataset

def generate_dataset_for_shapes(n_range, k, datapoints_per_n = 100):
    dataset = {}
    for n in n_range:
        dataset[n] = []
        print(f"Generating dataset for n = {n}")
        for _ in range(datapoints_per_n):
            dataset[n].append(random_k_regular_graph_to_adj(n, k, size = n))

        dataset[n] = torch.stack(dataset[n])

    return dataset


if __name__ == '__main__':
    dataset = generate_dataset_for_shapes(range(2,200), 6)
    
    # dataset = torch.stack([dataset[n] for n in range(2,200)])
    # torch.save(dataset, 'k_regular_200.pt')

    with open('k_regular.pickle', 'wb') as handle:
        pickle.dump(dataset, handle)

    # dataset = torch.load('k_regular.pt')
    # breakpoint()
