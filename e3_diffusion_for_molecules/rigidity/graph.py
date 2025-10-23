import numpy as np
from numpy import linalg as la


class Graph:
    """Entries should be dictionaries. Keys should be numbers 0,...,n and values should be subsets of 0,...,n.
    Graphs will be simple graphs.
    Dictionary will be made symmetric."""

    @staticmethod
    def make_dict_symmetric(graph_dict):
        """For making graphs simple"""
        for vertex in graph_dict:
            for neighbour in graph_dict[vertex]:
                graph_dict[neighbour].add(vertex)
            if vertex in graph_dict[vertex]:
                graph_dict[vertex].remove(vertex)

    def __init__(self, graph_dict):
        if graph_dict is None:
            graph_dict = {}
        Graph.make_dict_symmetric(graph_dict)
        self.adjacency_list = graph_dict

    def vertex_set(self):
        """Obtains vertices as a set."""
        vertices = set()
        for i in self.adjacency_list:
            vertices.add(i)
        return vertices

    def edge_list(self):
        """Obtains of list edges, each represented as a set."""
        edges = []
        for vertex in self.adjacency_list:
            for neighbour in self.adjacency_list[vertex]:
                if vertex < neighbour:
                    edges.append({vertex, neighbour})
        return edges

    def number_of_edges(self):
        return len(self.edge_list())

    def number_of_vertices(self):
        return len(self.vertex_set())

    def add_vertex(self):
        """Adds an extra vertex."""
        if self.vertex_set():
            max_vertex = max(self.vertex_set())
            self.adjacency_list[max_vertex + 1] = set()
        else:
            self.adjacency_list[0] = set()

    def add_edge(self, edge):
        """Adds an edge of the form {i,j}.
        Edge will be rejected if endpoints are not in the graph."""
        for i in edge:
            for j in edge:
                if i != j:
                    self.adjacency_list[i].add(j)

    def delete_edge(self, edge):
        """Deletes an edge of the form {i,j}.
        Edge will be rejected if endpoints are not in the graph."""
        for i in edge:
            for j in edge:
                if i != j:
                    self.adjacency_list[i].remove(j)

    def adjacency_matrix(self):
        """Obtains the graph's adjacency matrix."""
        n = self.number_of_vertices()
        adjacency_matrix = np.zeros((n, n))
        for vertex in self.adjacency_list:
            for neighbour in self.adjacency_list[vertex]:
                adjacency_matrix[vertex][neighbour] = 1
        return adjacency_matrix

    def laplacian(self):
        """Obtains the graph's Laplacian matrix."""
        laplacian_matrix = -self.adjacency_matrix()
        for vertex in self.adjacency_list:
            for neighbour in self.adjacency_list[vertex]:
                laplacian_matrix[vertex][vertex] += 1
        return laplacian_matrix

    def adjacency_eigenvalues(self):
        adj = self.adjacency_matrix()
        eigenvalues = list(la.eigvals(adj))
        eigenvalues.sort()
        return eigenvalues

    def laplacian_eigenvalues(self):
        lap = self.laplacian()
        eigenvalues = list(la.eigvals(lap))
        eigenvalues.sort()
        return eigenvalues

    def graph_number(self):
        """Obtains the number representation of the graph as described in:
        'The number of realizations of all Laman graphs with at most 12 vertices',
        Jose Capco, Matteo Gallet, Georg Grasegger, Christoph Koutschan, Niels Lubbes, Josef Schicho,
        10.5281/zenodo.1245517"""
        graph_number = 0
        n = self.number_of_vertices()
        for vertex in self.adjacency_list:
            count = n*(n-1)//2
            if vertex > 0:
                for i in range(vertex):
                    count -= n-i-1
            for neighbour in self.adjacency_list[vertex]:
                if vertex < neighbour:
                    graph_number += 2**(count - neighbour + vertex)
        return graph_number


class GraphWithOrientation(Graph):
    """A Graph with an added orientation to the edges, represented by dictionary.
    An edge {i,j} will either be directed from i to j (represented by the value of the key i containing j but
    the value of the key j not containing i),
    or vice versa, or non-oriented (represented by the value of the key i not containing j and vice versa)."""

    def __init__(self, graph_dict, orientation=None):
        super().__init__(graph_dict)
        if orientation is None:
            orientation = {}
            for vertex in self.adjacency_list:
                orientation[vertex] = set()
        self.orientation = orientation

    def oriented_edge_list(self):
        """Obtains of list oriented edges, each represented as a list."""
        edges = []
        for vertex in self.orientation:
            for neighbour in self.orientation[vertex]:
                edges.append([vertex, neighbour])
        return edges

    def new_edge_orientation(self, edge):
        """Give edge an orientation.
        This will overwrite any previous orientation.
        Edge should be written as [i,j]."""
        self.orientation[edge[0]].add(edge[1])

    def remove_edge_orientation(self, edge):
        """Removes orientation of edge.
        Edge should be written as {i,j}."""
        for i in edge:
            for j in edge:
                self.orientation[i].remove(j)

    def flip_edge(self, edge):
        """Flips orientation of an oriented edge.
        Edge should be written as {i,j}."""
        edge_as_list = list(edge)
        if edge_as_list in self.oriented_edge_list():
            self.orientation[edge_as_list[0]].remove(edge_as_list[1])
            self.orientation[edge_as_list[1]].add(edge_as_list[0])
        elif edge_as_list[::-1] in self.oriented_edge_list():
            self.orientation[edge_as_list[1]].remove(edge_as_list[0])
            self.orientation[edge_as_list[0]].add(edge_as_list[1])