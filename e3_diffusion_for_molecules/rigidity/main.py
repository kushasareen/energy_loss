import networkx as nx
from graph import Graph
from rigidity_checker import GlobalRigidityChecker
import numpy as np
import matplotlib.pyplot as plt

def generate_random_k_regular_graph(n, k):
    G = nx.random_regular_graph(k, n)
    graph_dict = {}
    for node, neighbours in G.adjacency():
        graph_dict[node] = set(neighbours)

    out = Graph(graph_dict)
    return out

def check_rigidity_k_regular_graphs(n, k, dimension, number_of_graphs, num_checks = 3):
    number_global_rigid = 0
    for _ in range(number_of_graphs):
        G = generate_random_k_regular_graph(n, k)
        # create a rigidity checker object
        rigidity_checker = GlobalRigidityChecker(G.adjacency_list, dimension)
        # check rigidity
        is_global_rigid = np.array([rigidity_checker.global_rigidity_check() for _ in range(num_checks)])
        number_global_rigid += is_global_rigid.all()

    # print(f"Fraction of global rigid graphs: {number_global_rigid/number_of_graphs}")
    return number_global_rigid/number_of_graphs

def check_rigidity_vs_k(n, k_range, dimension, number_of_graphs, num_checks = 3):
    fraction_global_rigid = []
    for k in k_range:
        fraction_global_rigid.append(check_rigidity_k_regular_graphs(n, k, dimension, number_of_graphs, num_checks))

    return fraction_global_rigid

def plot_rigidity_vs_k(n_range, k_range, d_range, number_of_graphs, num_checks = 3):
    ls=['-','--','-.',':', ':']

    for dimension in d_range:
        for idx, n in enumerate(n_range):
            print(f"n = {n}, d = {dimension}")
            fraction_global_rigid = check_rigidity_vs_k(n, k_range, dimension, number_of_graphs, num_checks)
            plt.plot(k_range, fraction_global_rigid, label=f"n = {n}, d = {dimension}", linestyle=ls[idx%len(ls)])

    plt.xlabel("k")
    plt.ylabel("Fraction of global rigid graphs")
    plt.title(f"Fraction of global rigid graphs vs k")
    plt.legend()
    # plt.grid()
    plt.xticks(k_range)
    plt.savefig(f"rigidity_vs_k_fast.png")

if __name__ == '__main__':
    n_range = [8, 16, 32, 64, 128]
    plot_rigidity_vs_k(n_range, range(3, 8), [3], 1, 1)
    # print(check_rigidity_k_regular_graphs(n= 30, k= 6, dimension= 3, number_of_graphs = 1000, num_checks = 1)