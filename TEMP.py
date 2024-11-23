import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

def ilp_vertex_cover(Ma):
    """
    Solves the Vertex Cover problem exactly using Integer Linear Programming (ILP).
    Args:
        Ma (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        list: List of vertices in the Vertex Cover.
    """
    n = Ma.shape[0]
    
    problem = LpProblem("Vertex_Cover", LpMinimize)

    # decision variables. ex) if x_i = 1, vertex i is in the vertex cover.
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(n)}

    # objective function
    problem += lpSum([x[i] for i in range(n)])

    # constraint: each edge is covered by at least one vertex
    for i in range(n):
        for j in range(i + 1, n):
            if Ma[i, j] == 1:
                problem += x[i] + x[j] >= 1

    problem.solve()

    return [i for i in range(n) if x[i].varValue == 1]

# Test graph (Adjacency Matrix)
# Graph:
# 0 -- 1
# |    
# 2 -- 3
Ma = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 0]
])

# Run the function and print the result
vertex_cover = ilp_vertex_cover(Ma)
print("Vertex Cover:", vertex_cover)