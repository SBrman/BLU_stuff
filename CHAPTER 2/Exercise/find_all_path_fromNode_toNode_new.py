#! python 3

"""Chapter 2 -- exercise 14"""
from math import inf
import logging

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Network:

    def __init__(self, N, A):

        self.N = {i: node for i, node in enumerate(N, 1)}
        self.A = {tuple((i, j)) for i, j in A}

        self.queue = []
        self.paths = set()
    
    def forward_star(self, node_index):
        """Returns the Forward star for the node"""
        return self.__star(node_index, link_inner_index=0)

    def __star(self, node_index, link_inner_index):
        """Returns the forward star or reverse star for the node"""
        for link in self.A:
            if self.N[node_index] == link[link_inner_index]:
                yield link
    
    def get_all_paths(self, start=1, end=None):
        if end == None:
            end = self.N[max(self.N.keys())]

        print(end)

        b_v = {i: [] for i, node in enumerate(self.N, 1)}
        f_v = {i: [] for i, node in enumerate(self.N, 1)}
        
        for node in self.N:
            for (i, j) in self.forward_star(node):
                b_v[j].append(i)
                f_v[i].append(j)
        print(b_v, '\n', f_v)

        # BFS search
        queue = [[start]]

        while queue:
            temp_path = queue.pop(0)
            logging.info(f'Popping {temp_path}')

            if temp_path[-1] == end:
                self.paths.add(tuple(temp_path))
                logging.info(f'Adding {self.paths}')
            else:
                for node in f_v[temp_path[-1]]:
                    temp_path.append(node)
                    logging.info(f'Modifying temp_path = {temp_path}')
                    queue.append(temp_path.copy())
                    logging.info(f'Modifying queue = {queue}')
                    temp_path.remove(node)
        
        return self.paths, f_v
    
def main():
        
    N = [1, 2, 3, 4]
    A = [[1, 2],[1, 3],[2, 3],[2, 4],[3, 4]]

    Nc = [1, 2, 3, 4, 5, 6]
    Ac = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)]

    G = Network(Nc, Ac)

    # Enter start and end nodes in the following method
    all_paths, f_vector = G.get_all_paths(start=1, end=4)
    
    [print(f'Path-{i}: {path}') for i, path in enumerate(all_paths, 1)]

    return all_paths, f_vector

if __name__ == '__main__':
    a, f = main()
