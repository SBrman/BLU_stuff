#! python 3

"""Chapter 2 -- exercise 14"""
from math import inf

        

class Network:

    def __init__(self, N, A):

        self.N = {i: node for i, node in enumerate(N)}
        self.A = {tuple((i, j)) for i, j in A}
    
    def forward_star(self, node_index):
        """Returns the Forward star for the node"""
        return self.__star(node_index, link_inner_index=0)

    def __star(self, node_index, link_inner_index):
        """Returns the forward star or reverse star for the node"""
        for link in self.A:
            if list(self.N)[node_index] == link[link_inner_index]:
                yield link
    
    def find_all_paths(self, start=1, end=None):
        if end == None:
            end = self.N[max(self.N.keys())]

        b_v = {i: [] for i, node in enumerate(self.N, 1)}

        for node in self.N:
            for (i, j) in self.forward_star(node):
                b_v[j].append(i)
        return b_v
    

N = [1, 2, 3, 4]
A = [[1, 2],[1, 3],[2, 3],[2, 4],[3, 4]]

G = Network(N, A)
q_r = G.find_all_paths()
print(q_r)
q_r_keys = list(q_r.keys())
q_r_keys.reverse()

k_rem = []
def getPath(k):
    global k_rem, paths
    
    q_r_rev = q_r[k]
    try:
        q_r_rev.reverse()
    except:
        print('problem!!!')
    
    for k1 in q_r_rev:
        k_rem.append(k1)
        if k_rem[0] == 4 and k_rem[-1] == 1:
            paths.add(tuple(k_rem))
            k_rem.pop()
            return
        getPath(k1)
        k_rem.pop()

    
paths = set()
for k in q_r_keys:
    k_rem.append(k)
    getPath(k)
    

##paths = []
##for k in q_r_keys:
##    for k1 in q_r[k]:
##        if k == 4 and k1 == 1:
##            path = [k, k1]
##            paths.append(path)
##        for k2 in q_r[k1]:
##            if k == 4 and k2 == 1:
##                path = [k, k1, k2]
##                paths.append(path)
##            for k3 in q_r[k2]:
##                if k == 4 and k3 == 1:
##                    path = [k, k1, k2, k3]
##                    paths.append(path)
                    



        

