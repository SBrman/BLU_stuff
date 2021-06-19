#! python3

import numpy as np
from math import inf
from sympy import simplify, Symbol
from ast import literal_eval
import pyinputplus as pyip
import logging
from pprint import pprint
from itertools import product, combinations

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Network:
    def __init__(self, N, A=None, L=None, heuristic=None, lp_func=None):
        # Nodes
        self.N = {key: value for key, value in enumerate(range(1, N+1), 1)}

        # Links
        self.A = set([(i, j) for i, j in A]) if A != None else set()

        # link Performance Function
        f = 'function}'
        self.lp_func = {} if lp_func == None else (L if type(lp_func) == dict else print(msg + f))
        
    def add_nodes(self, nodes):
        """
        Adds a list of nodes to Network.N
        """
        if type(nodes) is int:
            for node in range(1, nodes+1):
                self.N[len(self.N)+node] = len(self.N) + node
        elif type(nodes) in [list, set, np.ndarray]:
            for i, node in enumerate(nodes, 1):
                self.N[len(self.N)+i] = node
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

    def find_all_paths(self, start=1, end=None, max_node_repeat=1):
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
                new_path = tuple(temp_path[::-1])
                logging.info(f'Adding to path: {new_path}')
                self.h.add(new_path)
            else:
                for node in b_v[temp_path[-1]]:
                    temp_path.append(node)
                    logging.info(temp_path)
                    if temp_path.count(node) > max_node_repeat:
                        logging.info('Cyclic Network, so discard this path.')
                        temp_path.pop()
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
        array([[1, 0, 0, 1, 0, 0],
               [0, 1, 1, 0, 0, 0],
               [1, 0, 0, 0, 0, 1],
               [0, 1, 0, 1, 1, 0],
               [0, 1, 0, 0, 0, 0],
               [1, 0, 1, 0, 0, 1]])
        >>> link_rows
        [(1, 2), (1, 3), (2, 3), (2, 4), (3, 2), (3, 4)]
        >>> path_cols
        [(1, 3, 2), (1, 2), (1, 2, 3), (1, 3), (1, 2, 3, 4), (1, 3, 2, 4),
        (1, 2, 4), (1, 3, 4), (2, 3), (2, 4), (2, 3, 4), (3, 4), (3, 2, 4)]
        """
        paths_all = {}
        p_all = {}
        if routes == None:
            routes = self.all_routes()
        else:
            routes = product(*routes)
            
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
            for pathVal in paths_all.values():
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


def get_h(paths, d):
        divider = {}        # Assuming all the link flows are divided equally between links
        for path in paths:
            for k in d:
                if (path[0][0], path[-1][-1]) == k:
                    if k not in divider:
                        divider.setdefault(k, 1)
                    else: 
                        divider[k] += 1

        h_values = {}
        for path in paths:
            for k in d:
                if (path[0][0], path[-1][-1]) == k:
                    h_values[path] = d[k] / divider[k]

        return h_values

def print_matrix(matrix, indent=8, distance=5):
    """Prints the matrix"""
    print(' '*indent+ '+- ' + ' '*(distance*(len(matrix[0]))) + ' -+')
    for inner_list in matrix:
        print(' '*indent+'|', end='')
        for elem in inner_list:
            print(str(elem).rjust(distance), end='')
        print('|'.rjust(distance))
    print(' '*indent+ '+- ' + ' '*(distance*(len(matrix[0]))) + ' -+')


def get_xtc_and_lpaMatrix(G, path_flows, routes, h_vector_dict=None):
    """Returns x\u20D7, t\u20D7, c\u20D7 and the link path adjacency matrix
    Example:
    >>> x, t, c, lpa_mat, lpa_mat_link_row, lpa_mat_path_column =
                            get_xtc_and_lpaMatrix(Graph, path_flows, routes)
    >>> lpa_mat
    array([[1, 0, 0, 1, 0, 0],
           [0, 1, 1, 0, 0, 0],
           [1, 0, 0, 0, 0, 1],
           [0, 1, 0, 1, 1, 0],
           [0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 1]])
    >>> lpa_mat_link_row
    [(1, 2), (1, 3), (2, 3), (2, 4), (3, 2), (3, 4)]
    >>> lpa_mat_path_column
    [(1, 2, 3, 4), (1, 3, 2, 4), (1, 3, 4), (1, 2, 4), (2, 4), (2, 3, 4)]
    >>> x
    {(1, 2): 20.0, (1, 3): 20.0, (2, 3): 40.0, (2, 4): 50.0, (3, 2): 10.0,
    (3, 4): 50.0}
    >>> t
    {(1, 2): 200.0, (1, 3): 70.0, (2, 3): 50.0, (3, 2): 20.0, (2, 4): 100.0,
    (3, 4): 500.0}
    >>> c
    {(1, 2, 3, 4): 750.0, (1, 3, 2, 4): 190.0, (1, 3, 4): 570.0,
    (1, 2, 4): 300.0, (2, 4): 100.0, (2, 3, 4): 550.0}
    """
    # Start of Calculations:
    lpa_matrix, link_row, path_col = G.link_path_adjacency_matrix(routes=routes)
    alt_path_col = [tuple([path[0][0]] + list(np.array(path)[:,1])) for path in path_col]
    print('\u0394 =')
    print_matrix(lpa_matrix)

    # Getting link flows for each links as h_vector
    if h_vector_dict == None:
        h_vector_dict = get_h(path_col, path_flows)
        
    h_vector = [h_vector_dict[path] for path in path_col]
        
    # Dot product of delta(lpa_matrix) and h_vector
    print('\nDot product of \u0394 and h gives X. So,')
    print(f'    X = \u0394 . h\n==> X = ')
    x_vector = lpa_matrix.dot(h_vector)
    x_vector_dict = dict(zip(link_row, x_vector))
    pprint(x_vector_dict, indent=8)

    # Getting t_vector from the link performance functions
    print('\nSubstituting the X values in each link\'s link performance function ==>')
    print('    t = ',)
    t_vector_dict = {}
    x = Symbol('x') 
    for link, lp_func in G.lp_func.items():
        values_of_x = {x: x_vector_dict[link]}
        t_vector_dict[link] = float(lp_func.subs(values_of_x))
    pprint(t_vector_dict, indent=8)
    t_vector = [t_vector_dict[link] for link in link_row]

    # Getting the c_vector
    print('\nDot product of \u0394^T and t gives c.')
    print(f'    c = \u0394^T . t\n==> c = ')
    c_vector = lpa_matrix.transpose().dot(np.array(t_vector))
    
    c_vector_dict = dict(zip(alt_path_col, c_vector))
    pprint(c_vector_dict, indent=8)

    return x_vector_dict, t_vector_dict, c_vector_dict, lpa_matrix, link_row, alt_path_col

if __name__ == '__main__':

    # Define the Network
    Graph = Network(4)          # Network of 4 nodes
    Graph.add_links({(2, 4), (1, 2), (3, 4), (2, 3), (3, 2), (1, 3)})
    Graph.add_lpf({(1, 2): '10*x', (1, 3): 'x + 50', (2, 3): 'x + 10', (3, 2): 'x + 10',
               (2, 4): 'x + 50', (3, 4): '10*x'})

    # Routes and the given path_flows
    routes = [(1, 2), (4,)]     #[(origin nodes), (destination nodes)]
    path_flows = {(1,4): 40, (2,4): 60}

    # Get Everything
    x, t, c, lpa_mat, link_row, path_column = get_xtc_and_lpaMatrix(Graph, path_flows, routes)
