#! python3
"""Data structure and network representation"""

import logging

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class Network_representation:

    def __init__(self, N, A):
        self.N = set(N)
        self.A = set([tuple(element) for element in A])
        print(f'N = {self.N}')
        print(f'A = {self.A}')
        
        self.modifiable_N = self.N.copy()
        self.modifiable_A = self.A.copy()

    def remove_node_with_outgoing_links(self, node):
        """Removes any node with outgoing links"""
        for link in self.modifiable_A:
            if node == link[0]:
                #logging.debug(f'NOT DELETING --> N = {node}, A = {link}')
                return

        self.modifiable_N.remove(node)
        for link in self.modifiable_A.copy():
            if node in link:
                logging.debug(f'DELETING --> N = {node}, A = {link}')
                self.modifiable_A.remove(link)
        logging.debug(f'New A = {self.modifiable_A}')
        logging.debug(f'New N = {self.modifiable_N}')
        logging.debug('-'*50)

    def network_type(self):
        """Return network type"""

        logging.debug('-'*50)
        for _ in self.modifiable_N.copy():
            for node in self.modifiable_N.copy():
                self.remove_node_with_outgoing_links(node)
                
        if len(self.modifiable_A) == 0:
            return 'Acyclic Network'
        else:
            return 'Cyclic Network'
    
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
            
        return network

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

        return network
                
    def forward_star(self):
        """Returns a list of Points"""

        links = sorted(self.A)
        point = [''] * (len(self.N) + 1)
        point[-1] = len(self.A) + 1
        
        for i, node in enumerate(self.N):
            for link in links:
                if node == link[0]: # there is at least one outgoing link
                    point[i] = links.index(link)+1
                    break

        for i, elem in enumerate(point):
            if elem == '':
                point[i] = point[i+1]

        return point

    
def print_matrix(matrix):
    """Prints the matrix"""
    print('--' + ' '*(5 * len(matrix[0]) + 2) + '--')
    for inner_list in matrix:
        print('|', end='')
        for elem in inner_list:
            print(str(elem).rjust(5), end='')
        print('|'.rjust(5))
    print('--'+' '*(5*len(matrix[0])+2)+ '--')

# An Cyclic network
Nc = {1, 2, 3, 4, 5, 6}
Ac = {(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (6, 4), (5, 6)}

# An Acyclic network
Na1 = [1, 2, 3, 4]
Aa1 = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]

# Another Acyclic network
Na2 = {1, 2, 3, 4, 5, 6}
Aa2 = {(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)}

for i, (N, A) in enumerate([(Nc, Ac), (Na1, Aa1), (Na2, Aa2)], 1):

    print(f'NETWORK {i}'.center(60, '-'))
    
    G = Network_representation(N, A)

    print(f'Network Type: {G.network_type()}')
    
    print('\nNode-Link Incidence Matrix:')
    print_matrix(G.node_link_incidence())

    print('\nNode-Node Adjacency Matrix:')
    print_matrix(G.node_node_adjacency())
    
    print(f'\nPoint for Forward Star Represntation:\n{G.forward_star()}\n\n')
