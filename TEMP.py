
import sys, os, argparse, time, datetime
import numpy as np
import random
import networkx as nx
from concorde.tsp import TSPSolver
from redirector import Redirector
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import gurobipy as gp
from gurobipy import GRB

def solve_vertex_cover(Ma):
    """
    Solves the Vertex Cover problem exactly using Integer Linear Programming (ILP).
    Args:
        Ma (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        list: List of vertices in the Vertex Cover.
    """
    n = Ma.shape[0]

    # Create a new Gurobi model
    model = gp.Model("Vertex_Cover")

    # Suppress Gurobi output (optional)
    model.Params.OutputFlag = 0

    # Decision variables: x_i = 0 or 1 (binary variables)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective function: Minimize the total number of vertices in the vertex cover
    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MINIMIZE)

    # Constraints: For each edge (i, j), at least one endpoint must be in the vertex cover
    for i in range(n):
        for j in range(i + 1, n):
            if Ma[i, j] == 1:  # If there is an edge between i and j
                model.addConstr(x[i] + x[j] >= 1, name=f"edge_{i}_{j}")

    # Solve the ILP
    model.optimize()

    # Check if the optimization was successful
    if model.status == GRB.OPTIMAL:
        # Extract the solution: vertices in the vertex cover
        vertex_cover = [i for i in range(n) if round(x[i].X) == 1]
        return vertex_cover
    else:
        print("No optimal solution found!")
        return None
#end

# Example adjacency matrix for a simple graph
Ma5 = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0]
])

# Solve the Vertex Cover problem
vertex_cover = solve_vertex_cover(Ma5
)

print("Vertex Cover:", vertex_cover)
