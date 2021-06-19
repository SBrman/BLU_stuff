#! python3
"""Data structure and network representation"""

import logging
from pprint import pprint
import numpy as np

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Network:

    def __init__(self, N, A):
        self.N = set(N)
        self.A = set([tuple(element) for element in A])
        print(f'N = {self.N}')
        print(f'A = {self.A}')

    def node_link_incidence(self):
        """Returns node link incidence matrix for the network"""

        # Rows = Nodes and Columns = Links
        network = []
        for node in self.N:
            node_list = []

            for link in self.A:
                if node == link[0]:
                    node_list.append(1)
                elif node == link[1]:
                    node_list.append(-1)
                else:
                    node_list.append(0)

            network.append(node_list)

        return np.array(network)

    def node_link_incidence_to_network(self, matrix):
        """Returns the Network (G) = (NODES (N), LINKS (A)) for a given node link 
        incidence matrix"""

        links = []
        for row, node_link in enumerate(matrix):
            for column, link in enumerate(node_link):
                if link == 1 or link == -1:
                    for row_again, node_link1 in enumerate(matrix):
                        if row == row_again:
                            continue
                        if node_link1[column] == -1:
                            links.append((row+1, row_again+1))
                            break
        links = [tuple(element) for element in links]
        nodes = list(range(1, len(matrix)+1))
        return set(nodes), set(links)
                    
    def node_node_adjacency(self):
        """Returns node node adjacency matrix representation for the network"""
        network = []

        for i in self.N:
            row_list = []
            for j in self.N:
                if (i, j) in self.A:
                    row_list.append(1)
                else:
                    row_list.append(0)
            network.append(row_list)

        return np.array(network)

    def node_node_adjacency_to_network(self, matrix):
        """Returns the Network (G) = (NODES (N), LINKS (A)) for a given node 
        node adjacency matrix"""

        nodes = list(range(1, len(matrix)))
        links = []
        r = 0
        for row in matrix:
            r += 1
            for col, value in enumerate(row, 1):
                if value == 1:
                    links.append((r, col))

        links = [tuple(element) for element in links]
        return set(nodes), set(links)

    def forward_star(self):
        """Returns a list of Points"""

        links = sorted(list(self.A))
        point = [''] * (len(self.N) + 1)
        point[-1] = len(self.A) + 1

        for i, node in enumerate(self.N):
            for link in links:
                if node == link[0]:    # If there is at least one outgoing link
                    point[i] = links.index(link) + 1
                    break

        for i, elem in enumerate(point):
            if elem == '':
                point[i] = point[i+1]

        return point, links

    def forward_star_to_network(self, point_and_arcs):
        """Returns the Network (G) = (NODES (N), LINKS (A)) for a given point vector"""
        point, arcs = point_and_arcs
        nodes = set(list(range(1, len(point))))
        links = set(arcs)

        return set(nodes), set(links)


def print_matrix(matrix):
    """Prints the matrix"""
    print('--' + ' ' * (5*len(matrix[0])+2) + '--')
    for inner_list in matrix:
        print('|', end='')
        for elem in inner_list:
            print(str(elem).rjust(5), end='')
        print('|'.rjust(5))
    print('--' + ' ' * (5*len(matrix[0])+2) + '--')

def representation(N, A):
    G = Network(N, A)
    print(f'\nNetwork Type: {G.network_type()}')

    print('\nNode-Link Incidence Matrix:')
    print_matrix(G.node_link_incidence())

    print('\nNode-Node Adjacency Matrix:')
    print_matrix(G.node_node_adjacency())
    
    print(f'\nPoint for Forward Star Representation:\n{G.forward_star()}\n\n')


def main():
    # An Acyclic network
    Na1 = [1, 2, 3, 4]
    Aa1 = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]

    # Another Acyclic network
    Na2 = [1, 2, 3, 4, 5, 6]
    Aa2 = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)]

    # An Cyclic network
    Nc = [1, 2, 3, 4, 5, 6]
    Ac = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (6, 4), (5, 6)]
    
    G1 = Network(Nc, Ac)

    print('\n' + '-'*72)
    print('Node Link Incidence:')
    nli = G1.node_link_incidence()
    print_matrix(nli)

    nodes_nli, links_nli = G1.node_link_incidence_to_network(nli)
    print('\nNode Link Incidence To Network:')
    print(f'Nodes = {nodes_nli}\nLinks = {links_nli}\n')

    print('-'*72)
    nna = G1.node_node_adjacency()
    print('Node Node Adjacency:')
    print_matrix(nna)
    nodes_nna, links_nna = G1.node_node_adjacency_to_network(nna)
    print('\nNode Node Adjacency To Network:')
    print(f'Nodes = {nodes_nna}\nLinks = {links_nna}\n\n')

    print('-'*72)
    fs = G1.forward_star()
    print(f'Forward Star Representation:\nPoint = {fs[0]}\nArc = {fs[1]}')
    nodes_fs, links_fs = G1.forward_star_to_network(fs)
    print('\nForward Star Representation To Network:')
    print(f'Nodes = {nodes_fs}\nLinks = {links_fs}\n')
            
if __name__=='__main__':
    main()
