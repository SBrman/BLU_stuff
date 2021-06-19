#! python3
"""Data structure and network representation"""

from data_structures import representation

# Another Acyclic network
N1 = list(range(1, 9))
A1 = [(1, 2), (1, 4), (2, 5), (3, 5), (3, 6), (4, 5), (4, 7), (6, 8), (7, 5), (8, 3), (8, 5)]

#N = list(range(1, int(input('Number of nodes:'))))
##A = []
##while True:
##    inp = input('Enter Link(i,j)-->')
##    if not inp:
##        break
##    inp = inp.split(',')
##    link = (int(inp[0]), (int(inp[1])) )
##    A.append(link)
    
representation(N1, A1)
