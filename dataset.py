
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

    model = gp.Model("Vertex_Cover")
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
    vertex_cover = [i for i in range(n) if round(x[i].X) == 1]
    return vertex_cover
#end

def create_graph(n, connectivity, distances='euc_2D', metric=True):

    # Init adjacency and weight matrices
    Ma = np.zeros((n,n))
    #Mw = np.zeros((n,n))

    # Define adjacencies
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            rng = np.random.default_rng()
            Ma[i,j] = Ma[j,i] = int(rng.uniform(0, 1.0) < connectivity)
        #end
    #end
    # Solve
    vertex_cover = solve_vertex_cover(Ma)
    # if route is None:
    #     raise Exception('Unsolvable')
    # #end

    #node means coordinate, it doesn't influence VC
    return np.triu(Ma), vertex_cover
#end

def create_dataset(path, nmin, nmax, conn_min=0.2, conn_max=0.8, samples=1000, distances='euc_2D', metric=True):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    start_time = time.time()

    for i in range(samples):

        n = random.randint(nmin,nmax)

        # Create graph
        Ma, vertex_cover = create_graph(n, np.random.uniform(conn_min,conn_max), distances=distances, metric=metric)

        # Write graph to file
        write_graph(Ma, filepath="{}/{}.graph".format(path,i), vertex_cover=vertex_cover)

        # Report progress
        if (i-1) % (samples//20) == 0:
            elapsed_time = time.time() - start_time
            remaining_time = (samples-i)*elapsed_time/(i+1)
            print('Dataset creation {}% Complete. Remaining time at this rate: {}'.format(int(100*i/samples), str(datetime.timedelta(seconds=remaining_time))), flush=True)
        #end
    #end
#end

def write_graph(Ma, filepath, vertex_cover=None):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : Vertex Cover\n')
        out.write('DIMENSION: {n}\n'.format(n = n))


        out.write('EDGE_DATA_SECTION:\n')
        for i in range(Ma.shape[0]):  
            for j in range(i + 1, Ma.shape[1]): 
                if Ma[i, j] == 1:
                    out.write("{} {}\n".format(i, j))
        # out.write('EDGE_DATA_SECTION:\n')
        # for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
        #     out.write("{} {}\n".format(i,j))
        # #end
        out.write('-1\n')
        # Vertex Degree 저장
        vertex_degree = np.sum(Ma, axis=1) + np.sum(Ma, axis=0)# 행과 열 합산
        out.write("VERTEX_DEGREE:\n")
        out.write(" ".join(map(str, vertex_degree)) + "\n")

        out.write("VERTEX_COVER:\n")
        out.write(" ".join(map(str, vertex_cover)) + "\n")

        out.write('EOF\n')
    #end
#end

# def write_graph(Ma, Mw, filepath, route=None, int_weights=False, bins=10**6):
#     with open(filepath,"w") as out:

#         n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
#         out.write('TYPE : TSP\n')

#         out.write('DIMENSION: {n}\n'.format(n = n))

#         out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
#         out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
#         out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
#         # List edges in the (generally not complete) graph
#         out.write('EDGE_DATA_SECTION:\n')
#         for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
#             out.write("{} {}\n".format(i,j))
#         #end
#         out.write('-1\n')

#         # Write edge weights as a complete matrix
#         out.write('EDGE_WEIGHT_SECTION:\n')
#         for i in range(n):
#             for j in range(n):
#                 if Ma[i,j] == 1:
#                     out.write(str( int(bins*Mw[i,j]) if int_weights else Mw[i,j]))
#                 else:
#                     out.write(str(n*bins+1 if int_weights else 0))
#                 #end
#                 out.write(' ')
#             #end
#             out.write('\n')
#         #end

#         if route is not None:
#             # Write route
#             out.write('TOUR_SECTION:\n')
#             out.write('{}\n'.format(' '.join([str(x) for x in route])))
#         #end

#         out.write('EOF\n')
#     #end
# #end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('-distances', default='euc_2D', help='What type of distances? (euc_2D or random)')
    parser.add_argument('--metric', const=False, default=True, action='store_const', help='Create metric instances?')
    parser.add_argument('-samples', default=2**8, type=int, help='How many samples?')
    parser.add_argument('-path', help='Save path', required=True)
    parser.add_argument('-nmin', default=10, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=20, type=int, help='Max. number of vertices')
    parser.add_argument('-cmin', default=0.8, type=float, help='Min. connectivity')
    parser.add_argument('-cmax', default=0.2, type=float, help='Max. connectivity')
    parser.add_argument('-bins', default=10**6, help='Quantize edge weights in how many bins?')

    # Parse arguments from command line
    args = parser.parse_args()

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['path'],
        vars(args)['nmin'], vars(args)['nmax'],
        vars(args)['cmin'], vars(args)['cmax'],
        samples=vars(args)['samples'],
        #distribution=vars(args)['distribution']
    )
#end
