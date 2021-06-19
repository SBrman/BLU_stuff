#! python3

import numpy as np
from math import inf
from sympy import *
from ast import literal_eval
import pyinputplus as pyip
import networkx as nx
import matplotlib.pyplot as plt
import time
import datetime
import logging
from itertools import combinations
from pprint import pprint

logging.disable(logging.DEBUG)
##logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Network:
    def __init__(self, N, A=None, L=None, heuristic=None, lp_func=None):
        # Nodes
        if type(N) == int:
            self.N = {key: value for key, value in enumerate(range(1, N+1), 1)}
        elif type(N) in [list, set, np.ndarray]:
            self.N = {key: value for key, value in enumerate(N, 1)}
        elif type(N) == dict:
            self.N = N
        else:
            raise Exception('Nodes can only be a list/dict/set/np.ndarray or integer.')

        # Links
        self.A = set([(i, j) for i, j in A]) if A != None else set()

        msg = 'Enter dictionary containing {link: '
        # Labels for Links
        l = 'label}'
        self.L = {} if L == None else (L if type(L) == dict else print(msg + l))

        # Labels for Links
        he = 'heuristic}'
        self.heuristic = {} if heuristic == None else (heuristic if type(heuristic) == dict\
                                                                    else print(msg + he))
        # link Performance Function
        f = 'function}'
        self.lp_func = {} if lp_func == None else (L if type(lp_func) == dict else print(msg + f))
        
    def add_nodes(self, nodes):
        """
        Adds a list of nodes to Network.N
        """
        if type(nodes) is int:
            for node in range(nodes):
                self.N[len(self.N)+1] = len(self.N)+1
        elif type(nodes) in [list, set, np.ndarray]:
            for node in nodes:
                self.N[len(self.N)+1] = node
        else:
            raise Exception('Add number of nodes or list,set or np.ndarray of nodes')

    def del_nodes(self, nodes):
        for node in nodes:
            for nk, nv in self.N.copy().items():
                if node == nv:
                    del self.N[nk]

    def add_links(self, links=None):
        if links != None: 
            for link in links:
                try:
                    if len(link) == 2:
                        self.A.add(tuple(link))
                    else:
                        print('Enter a valid link')
                except TypeError:
                    print('Enter a valid link')
        else:
            while True:
                try:
                    inp = literal_eval(input('Enter link: '))
                    if type(inp) == tuple:
                        self.A.add(inp)
                except:
                    break
                
    def del_links(self, links):
        for link in links:
            if link in self.A:
                self.A.discard(link)
            else:
                print('This link does not exist in the set of links in this Network.')
    
    def add_labels(self, labels=None, link=None):
        if labels != None:
            if type(labels) == dict:
                for link, label in labels.items():
                    self.A.add(link)
                    self.L[link] = label 
            else:
                if link == None:
                    print('Did not specify a link')
                self.A.add(link)
                self.L[link] = labels 
        else:
            inp = pyip.inputYesNo('Manually enter Labels for each Link (Enter Yes/No): ')
            if inp != 'no' or 'No':
                links = list(self.A)
                links.sort()
                for link in links:
                    labels = input(f'Enter the Labels for link {link}: ')
                    self.L[link] = labels
            else:
                print('Could not modify labels for each link, Try again.')

    def add_heuristics(self, heuristics=None, link=None):
        if heuristics != None:
            if type(heuristics) == dict:
                for link, heuristic in heuristics.items():
                    self.A.add(link)
                    self.heuristic[link] = heuristic 
            else:
                if link == None:
                    print('Did not specify a link')
                self.A.add(link)
                self.heuristic[link] = heuristics
                
        else:
            inp = pyip.inputYesNo('Manually enter Heuristics for each Link (Enter Yes/No): ')
            if inp != 'no' or 'No':
                links = list(self.A)
                links.sort()
                for link in links:
                    heuristics = input(f'Enter the Link Performance Function for link {link}: ')
                    self.heuristic[link] = heuristics
            else:
                print('Could not modify heuristics for each link, Try again.')

    def add_lpf(self, lp_funcs=None, link=None):
        if lp_funcs != None:
            if type(lp_funcs) == dict:
                for link, func in lp_funcs.items():
                    self.A.add(link)
                    self.lp_func[link] = simplify(str(func)) 
            else:
                if link == None:
                    print('Did not specify a link')
                self.A.add(link)
                self.lp_func[link] = simplify(lp_funcs)
                
        else:
            inp = pyip.inputYesNo('Manually enter Link Performance Functions for each link \n(Enter Yes/No): ')
            if inp != 'no' or 'No':
                links = list(self.A)
                links.sort()
                for link in links:
                    func_inp = input(f'Enter the Link Performance Function for link {link}: ')
                    self.lp_func[link] = simplify(func_inp)
            else:
                print('Could not modify Link Performance Functions for each link, Try again.') 

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

    def forward_star_representation(self):
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

    def draw(self):
        """Draws the Network using networkx.draw_spectral"""
        if self.L == None:
            raise Exception('Labelling for links are not given.')
            return
        DG = nx.DiGraph()
        weighted_edges = [tuple((i, j, weight)) for (i, j), weight in self.L.items()]
        DG.add_weighted_edges_from(weighted_edges)
        nx.draw_spectral(DG, with_labels=True, font_weight='bold')

        plt.show(block=False)

    def find_all_paths(self, start=1, end=None, max_node_repeat=2):
        """Returns all the possible paths from startNode to endNode"""
        self.h = set()
        
        if end == None:
            end = self.N[max(self.N.values())]

        b_v = {i: [] for i, node in enumerate(self.N.values(), 1)}

        for node in self.N:
            for (i, j) in self.forward_star(node):
                b_v[j].append(i)

        logging.info(b_v)
        
        # BFS search
        queue = [[end]]

        while queue:
            temp_path = queue.pop(0)
            if temp_path[-1] == start:
                self.h.add(tuple(temp_path[::-1]))
            else:
                for node in b_v[temp_path[-1]]:
                    logging.info(temp_path)
                    temp_path.append(node)
                    if temp_path.count(node) > max_node_repeat:
                        logging.info('Cyclic Network')
                        continue
                    queue.append(temp_path.copy())
                    temp_path.remove(node)

        alt_paths = set()
        for path in self.h:
            temp_alt_path = []
            for i in range(len(path)-1):
                temp_alt_path.append((path[i], path[i+1]))
            alt_paths.add(tuple(temp_alt_path))

        return alt_paths, self.h
        
    def all_routes(self):
        """Returns a generator of all possible combinations of routes. Example:
        >>> G = Network(4)
        >>> list(G.all_routes())
        [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        """
        return combinations(self.N.values(), 2)

    def link_path_adjacency_matrix(self, repeat_node=1, routes=None):
        """Returns Link Path Adjacency matrix, rows (list of links) and
        columns (list of paths) of the matrix.

        Usage:

        >>> matrix, link_rows, path_cols = G.link_path_adjacency_matrix()

        >>> matrix
        array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
               [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]])

        >>> link_rows
        [(1, 2), (1, 3), (2, 3), (2, 4), (3, 2), (3, 4)]

        >>> path_cols
        [(1, 3, 2), (1, 2), (1, 2, 3), (1, 3), (1, 2, 3, 4), (1, 3, 2, 4), (1, 2, 4),
        (1, 3, 4), (2, 3), (2, 4), (2, 3, 4), (3, 4), (3, 2, 4)]

        """
        paths_all = {}
        p_all = {}
        if routes == None:
            routes = self.all_routes()
            
        # G.all_routes = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        for route in routes:
            paths_all[route], p_all[route] = self.find_all_paths(*route, max_node_repeat=repeat_node)
            # example: 1st paths_i2j = paths_1to2 = {((1, 2),), ((1, 3), (3, 2))}

        path_col = []

        lp_mat = []
        link_row = list(self.A)
        link_row.sort()

        for n, link in enumerate(link_row):
            row = []
            for path, pathVal in paths_all.items():
                for p in pathVal:
                    if n == 0:
                        path_col.append(p)
                    if link in p:
                        logging.info(f'link= {link}     path={p}')
                        row.append(1)
                    else:
                        row.append(0)
            lp_mat.append(row)
            
        lp_mat = np.array(lp_mat)
        return lp_mat, link_row, path_col


class Shortest_path(Network):

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


def print_matrix(matrix):
    """Prints the matrix"""
    print(' '*5+ '+-' + ' ' * (5*len(matrix[0])+2) + '-+')
    for inner_list in matrix:
        print(' '*5+'|', end='')
        for elem in inner_list:
            print(str(elem).rjust(5), end='')
        print('|'.rjust(5))
    print(' '*5+'+-' + ' ' * (5*len(matrix[0])+2) + '-+') 


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
    L = {(1, 2): 2, (1, 3): 4, (2, 3): 1, (2, 4): 5, (3, 4): 2}
    heuristic = {1: 3, 2: 2, 3: 1, 4: 0}

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
    #main()
    G = Network(4)
    G.add_links({(2, 4), (1, 2), (3, 4), (2, 3), (3, 2), (1, 3)})
##    G.add_labels({(2, 4): 1, (1, 2): 2, (3, 4): 3, (2, 3): 4, (3, 2): 5, (1, 3): 6})
    G.add_lpf({(1, 2): '10*x', (1, 3): 'x + 50', (2, 3): 'x + 10', (3, 2): 'x + 10',
               (2, 4): 'x + 50', (3, 4): '10*x'})
    lpa_matrix, link_row, path_col = G.link_path_adjacency_matrix()

    h_vector = np.array([0, 0, 0, 0, 10, 10, 10, 10, 0, 30, 30, 0, 0])
    x_vector = lpa_matrix.dot(h_vector)

    t_vector = {}
    x_vector_dict = dict(zip(link_row, x_vector))
    x = Symbol('x') 
    for link, lp_func in G.lp_func.items():
        d = {x: x_vector_dict[link]}
        t_vector[link] = lp_func.subs(d)
        
    c_vector = lpa_matrix.transpose().dot(np.array(list(t_vector.values())))
    c_vector_dict = dict(zip(path_col, c_vector))
    
    print('\u0394 =')
    print_matrix(lpa_matrix)
    print('\nDot product of \u0394 and h gives X. So,')
    print(f'    X = \u0394 . h\n==> X = ', end='')
    pprint(x_vector_dict)
    print('\nSubstituting the X values in each link\'s link performance function ==>')
    print('    t = ', end='')
    pprint(t_vector)
    print('\nDot product of \u0394^T and t gives c. So,')
    print(f'    c = \u0394^T . t\n==> c = ')
    pprint(c_vector_dict)
    
