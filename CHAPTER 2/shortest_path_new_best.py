#! python 3

"""Chapter 2 -- Shortest paths"""

from math import inf
import logging
from copy import deepcopy
from collections import OrderedDict
from time import time

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Shortest_path:

    def __init__(self, A, L, heuristic=None):
        self.A = {tuple((i, j)) for i, j in A}
        self.ordered_nodes = topological_sort(links=self.A)[1]
        self.N = {node: node for node in self.ordered_nodes.values()}
        self.L = L
        
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

        N = self.ordered_nodes

        # Step 1, Initializing by setting L_ir and q_ir
        L_r = {i: inf for i in self.N}
        q_r = {i: -1 for i in self.N}
        L_r[N[1]] = 0
        
        # Step 2
        j = 2

        iterations = 0
        while True:
            # Step 3
            L_r[N[j]], q_r[N[j]] = self.__bellman_principle_formula(N[j], L_r, q_r)

            # Step 4
            if j == len(N):
                break
            j += 1
            
            iterations += 1

        logging.debug(f'Total iterations - {iterations}')
        
        return (L_r, q_r)

    def cyclic(self):
        """Returns Shortest path Labels and Backnode vector for any network 
        using label correcting method"""
        print('For Cyclic Network using Label Correcting Algorithm: ')
        
        N = self.ordered_nodes

        # Step 1
        L_r = {i: inf for i in self.N}
        q_r = {i: -1 for i in self.N}
        L_r[N[1]] = 0

        # Step 2, Initialize Scan Eligible List (SEL)
        sel = {i for (r, i) in self.forward_star(N[1])}

        start = time()
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

            if time() - start >= 2:
                print('Stopped, taking a lot of time.')
                break

            # Step 6,
            if not sel:
                break

        logging.debug(f'Total iterations - {iterations}')
        
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
            for g in g_is.values():
                if g < 0:
                    raise Exception('Heuristic values must be non-negative')
                elif g > 0:
                    print('Using A* Algorithm: ')
                    break
            else:
                print('Can\'t use A* (Invalid heuristics). Using Dijksta\'s Algorithm: ')
                g_is = {node: 0 for node in self.N}

        elif method=='dijkstra':
            print('Using Dijksta\'s Algorithm: ')
            g_is = {node: 0 for node in self.N}

        N = self.ordered_nodes
        
        # Step 1, Initializing Label vector
        L_r = {i: inf for i in self.N}
        L_r[N[1]] = 0

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
            
        logging.debug(f'Total iterations - {iterations}')

        return (L_r, q_r)



def topological_sort(links):
    """Return network type and if the network is an acyclic network then this function
        returns nodes sorted in a topological order"""
    links = deepcopy(links)
    original_nodes = [node for link in links for node in link]
    original_nodes = {i: node for i, node in enumerate(original_nodes, 1)}

    def __get_all_degrees(nodes, links, degree_type='in'):
        if degree_type == 'in':
            pos = 1
        elif degree_type == 'out':
            pos = 0
        else:
            raise Exception('Invalid degree_type. Enter \'in\' or \'out\'.')

        degrees = {node: 0 for node in nodes.values()}
        for node in nodes:
            for link in links:
                if node == link[pos]:
                    degrees[node] += 1
        degrees = OrderedDict(sorted(degrees.items(), key=lambda x: x[1]))
        return degrees

    def get_indegrees(nodes, links):

        return __get_all_degrees(nodes, links, degree_type='in')

    def get_outdegrees(nodes, links):

        return __get_all_degrees(nodes, links, degree_type='out')
        
    def main_sorting(links):    
        ordered_nodes, i = {}, 0
        nodes = {node: node for link in links for node in link}

        while True:
            indegrees = get_indegrees(nodes, links)
            outdegrees = get_outdegrees(nodes, links)
            logging.debug(f'indegrees = {indegrees}\noutdegrees = {outdegrees}')
            
            for node, indegree in indegrees.items():
                if indegree == 0:
                    i += 1
                    node_key = list(nodes.keys())[list(nodes.values()).index(node)]
                    ordered_nodes[i] = nodes.pop(node_key)
                    for link in links.copy():
                        if node in link:
                            links.discard(link)

            if 0 not in set(indegrees.values()) and len(indegrees) > 1:
                return 'Cyclic Network', OrderedDict(original_nodes)
                
            elif len(set(indegrees.values())) == 0 and len(indegrees) <= 1:
                return 'Acyclic Network', OrderedDict(ordered_nodes)
                
    return main_sorting(links)


def main():

    # Acyclic Network
    A = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
    L = {(1, 2): 2, (1, 3): 4, (2, 3): 1, (2, 4): 5, (3, 4): 2}
    heuristic = {1: 3, 2: 2, 3: 1, 4: 0}
    
    As = [['1', '2'], ['1', '3'], ['2', '3'], ['2', '4'], ['3', '4']]
    Ls = {('1', '2'): 2, ('1', '3'): 4, ('2', '3'): 1, ('2', '4'): 5, ('3', '4'): 2}
    heuristic = {'1': 3, '2': 2, '3': 1, '4': 0}
    
    SP = Shortest_path(As, Ls, heuristic)

    for algorithm in [SP.acyclic, SP.cyclic, SP.dijkstra, SP.a_star]:
        L, q = algorithm()
        print(f'L_r: {list(L.values())}\nq_r: {list(q.values())}\n')

    return SP

if __name__ == '__main__':
    sp = main()
