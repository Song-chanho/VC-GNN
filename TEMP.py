import sys, os, argparse, time, datetime
import numpy as np
import random
import networkx as nx
from concorde.tsp import TSPSolver
from redirector import Redirector
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

n=10
Ma = np.zeros((n, n))
for i in range(n):
    Ma[i,i] = 0
    for j in range(i+1,n):
        Ma[i,j] = Ma[j,i] = int(np.random.rand() < 0.5)

for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
    print(i,j)