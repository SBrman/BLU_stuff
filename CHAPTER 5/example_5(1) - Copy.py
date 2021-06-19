#! python3

import numpy as np
from sympy import *
from ast import literal_eval
import pyinputplus as pyip

class Network:
    def __init__(self, N, A=None, lp_func=None):

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
        if A != None:
            self.A = set([(i, j) for i, j in A])
        else:
            self.A = set()

        # link Performance Function
        if lp_func == None:
            self.lp_func = {}
        elif type(lp_func) == dict:
            self.lp_func = lp_func
        else:
            raise Exception('Enter dictionary containing {link: function}')

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

    def add_links(self, links):
        for link in links:
            try:
                if len(link) == 2:
                    self.A.add(tuple(link))
                else:
                    print('Enter a valid link')
            except TypeError:
                print('Enter a valid link')

    def del_links(self, links):
        for link in links:
            if link in self.A:
                self.A.discard(link)
            else:
                print('This link does not exist in the set of links in this Network.')

    def add_lpf(self, link=None, lp_funcs=None):
        if lp_funcs != None:
            if type(lp_funcs) == dict:
                for link, func in lp_funcs.items():
                    self.A.add(link)
                    self.lp_func[link] = simplify(func) 
            else:
                if link == None:
                    print('Did not specify a link')
                self.A.add(link)
                self.lp_func[link] = simplify(lp_funcs)
                
        else:
            inp = pyip.inputYesNo('Manually enter Link Performance Functions for each link: ')
            if inp != 'no' or 'No':
                links = list(self.A)
                links.sort()
                for link in links:
                    func_inp = input(f'Enter the Link Performance Function for link {link}: ')
                    self.lp_func[link] = simplify(func_inp)
            else:
                print('Could not modify Link Performance Functions for each link, Try again.') 
