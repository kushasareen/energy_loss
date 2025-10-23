"""Requires numpy and scipy"""

from graph import Graph
import numpy as np
from numpy.linalg import matrix_rank
import random
from scipy.linalg import null_space


class RigidityChecker(Graph):

    def __init__(self, graph_dict, dimension):
        """Graph must be in the correct form for Graph class"""
        super().__init__(graph_dict)
        self.dimension = dimension
        self.independent = False
        self.rigid = False

    def dimension_increase(self):
        self.dimension += 1

    def dimension_decrease(self):
        self.dimension -= 1

    def random_placement(self):
        placement = np.zeros((self.number_of_vertices(), self.dimension))

        for v in self.vertex_set():
            for k in range(self.dimension):
                placement[v][k] = random.randint(0, 100 * self.number_of_vertices())
        return placement

    def random_rigidity_matrix(self):
        rigidity_matrix = np.zeros((self.number_of_edges(), self.dimension * self.number_of_vertices()))
        placement = self.random_placement()
        for e in self.edge_list():
            for v in self.vertex_set():
                if v == min(e):
                    for k in range(self.dimension):
                        rigidity_matrix[self.edge_list().index(e)][self.dimension * v + k] = \
                            placement[v][k] - placement[max(e)][k]
                elif v == max(e):
                    for k in range(self.dimension):
                        rigidity_matrix[self.edge_list().index(e)][self.dimension * v + k] = \
                            -rigidity_matrix[self.edge_list().index(e)][self.dimension * min(e) + k]
        return rigidity_matrix

    def rank_check(self, matrix):
        if matrix_rank(matrix) == self.number_of_edges():
            self.independent = True
        else:
            self.independent = False
        if matrix_rank(matrix) == self.dimension * self.number_of_vertices() - \
                (self.dimension * (self.dimension + 1) / 2):
            self.rigid = True
        else:
            self.rigid = False

    def rigidity_check(self):
        self.rank_check(self.random_rigidity_matrix())
        print("")
        n = self.number_of_vertices()
        m = self.number_of_edges()
        if n < self.dimension + 1:
            if 2 * m == n * (n - 1):
                print(f"Graph is minimally rigid in dimension {self.dimension}.", flush=True)
            else:
                print(f"Graph is independent and flexible in dimension {self.dimension}.", flush=True)
        elif (self.independent is True) and (self.rigid is True):
            print(f"Graph is minimally rigid in dimension {self.dimension}.", flush=True)
        elif (self.independent is True) and (self.rigid is False):
            print(f"Graph is independent and flexible in dimension {self.dimension}.", flush=True)
        elif (self.independent is False) and (self.rigid is True):
            print(f"Graph is dependent and rigid in dimension {self.dimension}.", flush=True)
        elif (self.independent is False) and (self.rigid is False):
            print(f"Graph is dependent and flexible in dimension {self.dimension}. Rerun to double check.", flush=True)


class GlobalRigidityChecker(RigidityChecker):

    def __init__(self, graph_dict, dimension):
        super().__init__(graph_dict, dimension)
        self.globally_rigid = False
        self.globally_rigid_test_fail = False

    def random_stress(self, matrix):
        r2 = np.transpose(matrix)
        ns = null_space(r2)

        if ns.size != 0:
            number_stress = len(ns[0])
            vec = np.zeros((number_stress, 1))
            for s in range(number_stress):
                vec[s][0] = random.randint(0, 100 * self.number_of_edges())
            stress = ns.dot(vec)
        else:
            stress = np.zeros((self.number_of_edges(), 1))
        return stress

    def stress_matrix(self, stress):
        stress_matrix = np.zeros((self.number_of_vertices(), self.number_of_vertices()))
        for v in range(self.number_of_vertices()):
            for w in range(self.number_of_vertices()):
                for e in self.edge_list():
                    if v == w and v in e:
                        stress_matrix[v][w] = stress_matrix[v][w] + stress[self.edge_list().index(e)]
                    elif (v in e) and (w in e):
                        stress_matrix[v][w] = -stress[self.edge_list().index(e)]
        return stress_matrix

    def stress_rank_check(self, stress):
        stress_matrix = self.stress_matrix(stress)
        rank = matrix_rank(stress_matrix)
        if rank == self.number_of_vertices() - self.dimension - 1:
            self.globally_rigid = True
        elif (rank > self.number_of_vertices() - self.dimension - 1) and (rank > 0):
            self.globally_rigid_test_fail = True
        else:
            self.globally_rigid = False

    def global_rigidity_check(self):
        rigidity_matrix = self.random_rigidity_matrix()
        self.rank_check(rigidity_matrix)
        stress = self.random_stress(rigidity_matrix)
        self.stress_rank_check(stress)
        # print("")
        n = self.number_of_vertices()
        m = self.number_of_edges()
        if n < self.dimension + 1:
            if 2 * m == n * (n - 1):
                #print(f"Graph is globally rigid in dimension {self.dimension}.", flush=True)
                return True
            else:
                # print(f"Graph is not globally rigid in dimension {self.dimension}.", flush=True)
                return False
        elif self.globally_rigid_test_fail is True:
            # print(f"Test failed: rank is too high.", flush=True)
            return False
        elif (self.rigid is True) and (self.number_of_vertices() <= self.dimension + 1):
            # print(f"Graph is globally rigid in dimension {self.dimension}.", flush=True)
            return True
        elif (self.globally_rigid is True) and (self.rigid is True):
            # print(f"Graph is globally rigid in dimension {self.dimension}.", flush=True)
            return True
        else:
            # print(f"Graph is not globally rigid in dimension {self.dimension}. Rerun to double check.", flush=True)
            return False
