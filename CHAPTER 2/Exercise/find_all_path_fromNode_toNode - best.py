#! python 3

"""Chapter 2 -- exercise 14"""
from math import inf

class Network:

    def __init__(self, N, A):

        self.N = {i: node for i, node in enumerate(N)}
        self.A = {tuple((i, j)) for i, j in A}
        self.h = set()
    
    def forward_star(self, node_index):
        """Returns the Forward star for the node"""
        return self.__star(node_index, link_inner_index=0)

    def __star(self, node_index, link_inner_index):
        """Returns the forward star or reverse star for the node"""
        for link in self.A:
            if list(self.N)[node_index] == link[link_inner_index]:
                yield link
    
    def find_all_paths(self, start=1, end=None):
        """Returns all the possible paths from startNode to endNode"""
        if end == None:
            end = self.N[max(self.N.keys())]

        b_v = {i: [] for i, node in enumerate(self.N, 1)}

        for node in self.N:
            for (i, j) in self.forward_star(node):
                b_v[j].append(i)

        # BFS search
        queue = [[end]]

        while queue:
            temp_path = queue.pop(0)
            if temp_path[-1] == start:
                self.h.add(tuple(temp_path[::-1]))
            else:
                for node in b_v[temp_path[-1]]:
                    temp_path.append(node)
                    queue.append(temp_path.copy())
                    temp_path.remove(node)
        
        return self.h, b_v
    
def main():
        
    N = [1, 2, 3, 4]
    A = [[1, 2],[1, 3],[2, 3],[2, 4],[3, 4]]

    Nc = [1, 2, 3, 4, 5, 6]
    Ac = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)]

    G = Network(Nc, Ac)

    # Enter start and end nodes in the following method
    all_paths, backnode_vector = G.find_all_paths(start=1, end=6)
    
    [print(f'Path-{i}: {path}') for i, path in enumerate(all_paths, 1)]

    return backnode_vector

if __name__ == '__main__':
    q = main()
