#! python 3

"""USAGE --
get_shortest_path.get_sp()"""

from math import inf
import logging
from data_structures import Network, print_matrix, representation
from shortest_path_new_nx import Shortest_path
import pyinputplus as pyip
import ast
from pprint import pprint

WIDTH = 45

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def get_sp():
    """Takes inputs for Nodes, Links, Costs/Labels and Heuristics and algorithm type \
and returns L_r and q_r."""
    
    # Nodes
    N = list(range(1, int(input('Number of nodes = '))+1 ))
    print(f'N = {N}')
    print('-'*WIDTH)

    # Links
    print('Enter links by which method?')
    lc = pyip.inputMenu(['Set of all Links', 'Links one by one'], numbered=True)
    if lc == 'Links one by one':
        A = set()
        while True:
            print('Enter Link(i,j):')
            fromInp = pyip.inputInt('From Node: ', blank=True)
            toInp = pyip.inputInt('To Node: ', blank=True)

            if fromInp == '' or toInp == '':
                break

            print(f'Entering Link {(fromInp, toInp)}\n')
            A.add((fromInp, toInp))
    else:
        print('A = ', end='')
        A = ast.literal_eval(input())
        print()

    print(f'A = {A}') 
    print('-'*WIDTH)
    

    # Label or Cost values for each link 
    print('Enter Costs by which method?')
    lcc = pyip.inputMenu(['Set of all Costs', 'Costs one by one'], numbered=True)
    if lcc == 'Set of all Costs':
        print('L = ', end='')
        L = ast.literal_eval(input())
        print()

    else:
        L = {}
        Arcs = list(A)
        Arcs.sort()
        print('Type of Cost or label?')
        costUnit = pyip.inputMenu(['Travel Time', 'Distance',
                                   'Other unit (integer)'], numbered=True)
            
        for link in Arcs:
            if costUnit == 'Travel Time':
                tt = input(f'Travel time cost (Enter h,m) for Link {link} = ')
                tt = ast.literal_eval(tt)
                inp = tt[0] + (tt[1]/60)
            elif costUnit == 'Distance':
                inp = float(input(f'Distance cost for Link {link} = '))
            else:
                inp = int(input(f'Cost for Link {link} = '))
            L[link] = inp

    print(f'L = {L}')
    print('-'*WIDTH)

    # Heuristic values for each link
    H = {}
    print('Will you use A* Algorithm? (Enter Yes/no)')
    H_use = pyip.inputYesNo()
    if H_use.lower() in ['yes', 'y']:
        print('Enter the Heuristic values: ')
        for node in N:
            inp = int(input(f'Heuristic for Node {node} = '))
            if inp == None:
                break
            H[node] = inp

        print(f'H = {H}')

    if len(H) == 0:
        H = None
    print('-'*WIDTH)

    SP = Shortest_path(N, A, L, H)

    algorithms = {
                    'Acyclic': SP.acyclic,
                    'Label Correcting': SP.cyclic,
                    "Label Setting (Dijkstra's Algorithm)": SP.dijkstra,
                    'A* Algorithm': SP.a_star
                  }

    representation(N, A)

    while True:
        # Which algorithm to use
        print('Which algorithm to use?')
        alg_choice = pyip.inputMenu(['Acyclic', 'Label Correcting',
                                     'Label Setting (Dijkstra\'s Algorithm)',
                                     'A* Algorithm'], numbered=True)
        print('-'*WIDTH)    
        
        L, q = algorithms[alg_choice]()
        print('L_r = ', end='')
        pprint(list(L.values()))
        print('q_r = ', end='')
        pprint(list(q.values()))
        again = pyip.inputYesNo('Try another algorithm? (yes/no)\n')

        if again.lower() not in ['yes', 'y']:
            break

def main():
    while True:
        get_sp()
        print('-'*WIDTH)
        if not input('Try again with another Network? (y/n)\n').lower() in ['yes', 'y']:
            break
if __name__== "__main__":
    main()
