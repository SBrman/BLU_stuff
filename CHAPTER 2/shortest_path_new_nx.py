#! python 3

"""Chapter 2 -- Shortest paths with graph in plt"""

from math import inf
import logging
import matplotlib.pyplot as plt
import networkx as nx
import time
import datetime

logging.disable(logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Shortest_path:

    def __init__(self, N, A, L, heuristic=None):

        self.N = {i: node for i, node in enumerate(N, 1)}
        self.A = set([tuple((i, j)) for i, j in A])
        self.L = L

        # Drawing the Graph
        self.DG = nx.DiGraph()
        weighted_edges = [tuple((i, j, weight)) for (i, j), weight in L.items()]
        self.DG.add_weighted_edges_from(weighted_edges)
        nx.draw_spectral(self.DG, with_labels=True, font_weight='bold')

        if heuristic == None:
            self.heuristic = {i+1: 0 for i in range(len(self.N))}
        else:
            self.heuristic = heuristic

    def forward_star(self, node):
        """Returns the Forward star for the node"""
        return self.__star(node, link_inner_index=0)

    def reverse_star(self, node):
        """Returns the Reverse star for the node"""
        return self.__star(node, link_inner_index=1)

    def __star(self, node, link_inner_index):
        """Returns the forward star or reverse star for the node"""
        for link in self.A:
            if self.N[node] == link[link_inner_index]:
                yield link

    def __bellman_principle_formula(self, node, L_r, q_r):
        temp_L_ir = {}
        
        for h, i in self.reverse_star(node):
            L_hr = L_r[h]             # node starts at 0 
            c_hi = self.L[(h, i)]
            
            temp_L_ir[h] = ((L_hr + c_hi))

        L_ir, q_ir = min(zip(temp_L_ir.values(), temp_L_ir.keys()))

        return (L_ir, q_ir)

    def acyclic(self):
        """Returns shortest path"""
        
        print('For Acyclic Network using Label Correcting Algorithm: ')

        # Step 1, Initializing by setting L_ir and q_ir
        L_r = {i: inf for i in self.N}
        q_r = {i: -1 for i in self.N}
        L_r[1] = 0
        
        # Step 2
        j = 2

        iterations = 0
        while True:
            # Step 3
            L_r[j], q_r[j] = self.__bellman_principle_formula(j, L_r, q_r)

            # Step 4
            if j == list(self.N.keys())[-1]:
                break
            j += 1
            
            iterations += 1

        logging.info(f'Total iterations - {iterations}')
        
        return (L_r, q_r)

    def cyclic(self):
        """Returns Shortest path Labels and Backnode vector for any network 
        using label correcting method"""
        print('For Cyclic Network using Label Correcting Algorithm: ')

        # For some networks this algorithm is not terminating so will end if takes more that 5 seconds
        start = time.time()
        dt = datetime.timedelta(seconds=5)
        
        # Step 1
        L_r = {i: inf for i in self.N}
        q_r = {i: -1 for i in self.N}
        L_r[1] = 0

        # Step 2, Initialize Scan Eligible List (SEL)
        sel = set([i for (r, i) in self.forward_star(1)])

        iterations = 0
        while True:
            # Step 3, Select a node from sel and remove that from sel
            node = sel.pop()
            j = {v:k for k, v in self.N.items()}[node]

            # Step 4, find L_ir and q_ir for the selected node
            temp_L_ir, temp_q_ir = self.__bellman_principle_formula(j, L_r, q_r)

            # Preparation for the 5th step
            L_ir_changed = True if L_r[j] != temp_L_ir else False

            L_r[j], q_r[j] = temp_L_ir, temp_q_ir

            # Step 5, 
            downstream_nodes = set([k for _, k in self.forward_star(j)]) if L_ir_changed else set()
            sel = sel.union(downstream_nodes)

            iterations += 1

            # Step 6,
            if not sel:
                break

            if time.time() > (start + dt.seconds):
                return (None, None)

        logging.info(f'Took {iterations} iterations')
        
        return (L_r, q_r)

    def dijkstra(self):
        """Returns Shortest path Labels and Backnode vector for Cyclic network 
        using Dijksta's Algorithm (label setting method)"""

        return self.a_star('dijkstra')

    def a_star(self, method='a_star'):
        """Returns Shortest path Labels and Backnode vector for Cyclic network 
        using A* Algorithm (One origin to one destination)"""
        
        if method=='a_star':
            g_is = self.heuristic
            for g in g_is:
                if g < 0:
                    raise Exception('Heuristic values must be non-negative')
                elif g > 0:
                    print('Using A* Algorithm: ')
                    break
            else:
                print('Using Dijksta\'s Algorithm: ')
                g_is = {i+1: 0 for i in range(len(self.N))}
        elif method=='dijkstra':
            print('Using Dijksta\'s Algorithm: ')
            g_is = {i+1: 0 for i in range(len(self.N))}

        # Step 1, Initializing Label vector
        L_r = {i: inf for i in self.N}
        L_r[1] = 0

        # Step 2, initializing Finalized set and Backnode vector
        F = set()
        q_r = {i: -1 for i in self.N}

        iterations = 0
        while True:
            # Step 3, Selecting node based on min L_ir + g_is if it's not in F
            unfinalized = {node: (L_ir + g_is[node]) for node, L_ir in L_r.items()}
            
            for node in F:
                del unfinalized[node]
            
            min_value, i = min(zip(unfinalized.values(), unfinalized.keys()))

            # Step 4, Finalize node j by adding it to F
            F.add(i)
            if len(F) == len(self.N):
                break

            # Step 5, Update the labels for the outgoing links from node j
            for (i, j) in self.forward_star(i):
                c_ij = self.L[(i, j)]
                L_ir = min(L_r[j], (L_r[i] + c_ij))

                if L_ir < L_r[j]:
                    L_r[j] = L_ir

                    # Step 6
                    q_r[j] = i
                    
            iterations += 1
            
            # Step 7
            if len(F) == len(self.N):
                break
            
        logging.info(f'Took {iterations} iterations')

        return (L_r, q_r)


def main():
    
    # An Cyclic network
    Nc = [1, 2, 3, 4, 5, 6]
    Ac = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (6, 4), (5, 6)]

    # Acyclic network 1
    Na1 = [1, 2, 3, 4]
    Aa1 = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]

    # Acyclic network 2
    Na2 = [1, 2, 3, 4, 5, 6]
    Aa2 = [(1, 2), (2, 3), (2, 4), (4, 5), (6, 3), (4, 6), (5, 6)]

    # Acyclic Network 3
    N = [1, 2, 3, 4]
    A = [[1, 2],[1, 3],[2, 3],[2, 4],[3, 4]]
    L = {   
            (1, 2): 2, 
            (1, 3): 4, 
            (2, 3): 1, 
            (2, 4): 5, 
            (3, 4): 2
        }
    
    heuristic = {   
                    1: 3, 
                    2: 2, 
                    3: 1, 
                    4: 0
                }

    # New Network
    Nn = [1, 2, 3, 4, 5, 6, 7, 8]
    An = [(1, 2), (1, 3), (2, 1), (2, 4), (2, 6), (3, 1), (3, 4), (4, 3),
         (4, 5), (4, 6), (5, 4), (5, 7), (6, 4), (6, 7), (6, 8), (7, 5),
         (7, 8), (8, 7), (8, 6)]
    Ln = {(1, 2): 3, (1, 3): 8, (2, 1): 3, (2, 4): 2, (2, 6): 9, (3, 1): 8,
         (3, 4): 4, (4, 3): 4, (4, 5): 10, (4, 6): 5, (5, 4): 10, (5, 7): 7,
         (6, 4): 5, (6, 7): 1, (6, 8): 6, (7, 5): 7, (7, 8): 11, (8, 7): 11,
         (8, 6): 6, (7, 6): 1}
    
    SP = Shortest_path(Nn, An, Ln)#, heuristic)

    for algorithm in [SP.acyclic, SP.cyclic , SP.dijkstra , SP.a_star]:
        L, q = algorithm()

        if L == None:
            print('Algorithm is taking too much time so stopping it.\n\n')
            continue
            
        print(f'L_r: {list(L.values())}\nq_r: {list(q.values())}\n')

if __name__ == '__main__':
    main()
    plt.show()
