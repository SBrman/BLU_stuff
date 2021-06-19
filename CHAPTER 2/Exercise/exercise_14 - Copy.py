#! python 3

"""Chapter 2 -- exercise 14"""
from math import inf

class Network:

    def __init__(self, N, A):

        self.N = {i: node for i, node in enumerate(N)}
        self.A = {tuple((i, j)) for i, j in A}
        self.k_rem = []
        self.paths = set()
    
    def forward_star(self, node_index):
        """Returns the Forward star for the node"""
        return self.__star(node_index, link_inner_index=0)

    def __star(self, node_index, link_inner_index):
        """Returns the forward star or reverse star for the node"""
        for link in self.A:
            if list(self.N)[node_index] == link[link_inner_index]:
                yield link
    
    def find_all_paths(self, start=1, end=4):
        if end == None:
            end = self.N[max(self.N.keys())]

        b_v = {i: [] for i, node in enumerate(self.N, 1)}

        for node in self.N:
            for (i, j) in self.forward_star(node):
                b_v[j].append(i)

        q_r_keys = list(b_v.keys())
        q_r_keys.reverse()
        for k in q_r_keys:
            if k > end:
                continue
            
            self.k_rem.append(k)
            self.__getPath(k, b_v, start, end)
        
        return self.paths, b_v

    def __getPath(self, k, b_v, start, end):
        q_r_rev = b_v[k]
        q_r_rev.reverse()
    
        for k1 in q_r_rev:
            self.k_rem.append(k1)
            if self.k_rem[0] == end and self.k_rem[-1] == start:
                self.paths.add(tuple(self.k_rem))
                self.k_rem.pop()
                return
            self.__getPath(k1, b_v, start, end)
            self.k_rem.pop()
    
def main():
        
    N = [1, 2, 3, 4]
    A = [[1, 2],[1, 3],[2, 3],[2, 4],[3, 4]]

    G = Network(N, A)

    # Enter start and end nodes in the following method
    all_paths, backnode_vector = G.find_all_paths(start=1, end=4)
    
    [print(f'Path-{i}: {path}') for i, path in enumerate(all_paths, 1)]

if __name__ == '__main__':
    main()
