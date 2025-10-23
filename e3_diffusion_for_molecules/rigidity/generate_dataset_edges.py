import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

def random_k_regular_graph_to_edges(n, k, size = 29):
    k = min(k, n-1)
    G = nx.random_regular_graph(k, n)
    edges = list(G.edges())
    return np.array(edges)

def generate_dataset(n_range, k, size = 29, datapoints_per_n = 2):
    dataset = {}
    for n in n_range:
        dataset[n] = []
        print(f"Generating dataset for n = {n}")
        for _ in range(datapoints_per_n):
            edges = random_k_regular_graph_to_edges(n, k, size)
            dataset[n].append(edges)

    return dataset

if __name__ == '__main__':
    # dataset = generate_dataset_for_shapes(range(2,200), 6)
    dataset = generate_dataset([30, 300, 3000, 30000, 300000], 6, datapoints_per_n=1)
    
    # dataset = torch.stack([dataset[n] for n in range(2,200)])
    # torch.save(dataset, 'k_regular_200.pt')

    # with open('k_regular.pickle', 'wb') as handle:
    #     pickle.dump(dataset, handle)

    with open('k_regular_timing.pickle', 'wb') as handle:
        pickle.dump(dataset, handle)

    # dataset = torch.load('k_regular.pt')
    # breakpoint()
