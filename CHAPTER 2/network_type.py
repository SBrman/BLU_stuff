#! python3
"""Python script to determine a network type"""

import logging

#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class Network:
    def __init__(self, N, A):
        self.N = N
        self.A = A

    def remove_node_with_outgoing_links(self, node):
        """Removes any node with outgoing links"""
        for link in self.A:
            if node == link[0]:
                #logging.debug(f'NOT DELETING --> N = {node}, A = {link}')
                return

        self.N.remove(node)
        for link in self.A.copy():
            if node in link:
                logging.debug(f'DELETING --> N = {node}, A = {link}')
                self.A.remove(link)
        logging.debug(f'New A = {self.A}')
        logging.debug(f'New N = {self.N}')
        logging.debug('-'*50)

    def network_type(self):
        """Return network type"""

        for _ in self.N.copy():
            for node in self.N.copy():
                self.remove_node_with_outgoing_links(node)
                
        if len(self.A) == 0:
            return 'Acyclic Network'
        else:
            return 'Cyclic Network'

def main():

    N = {1, 2, 3, 4, 5, 6}
    Aa = {(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)} # Acyclic
    Ac = {(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (6, 4), (5, 6)} # Cyclic

    Nn = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    An = {(1, 2), (1, 3), (1, 4), (2, 10), (3, 10), (3, 5), (4, 5), (5, 10), (6,7), (7,8), (8,9), (7,10), (8,10)}
    
    G = Network(Nn, An)
    print(G.network_type())

if __name__ == '__main__':
    main()
