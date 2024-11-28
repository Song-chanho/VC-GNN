
import os, sys
import random
import numpy as np
from functools import reduce

class InstanceLoader(object):

    def __init__(self,path):
        self.path = path
        self.filenames = [ path + '/' + x for x in os.listdir(path) ]
        random.shuffle(self.filenames)
        self.reset()
    #end

    def get_instances(self, n_instances):
        for i in range(n_instances):
            # Read graph from file
            Ma, vertex_cover, vertex_degrees = read_graph(self.filenames[self.index])
            # Yield two copies of the same instance
            yield Ma, vertex_cover, vertex_degrees
            yield Ma, vertex_cover, vertex_degrees

            self.index += 1
        #end
    #end

    def create_batch(instances, dev=0.02, training_mode='relational', target_cost=None):

        # n_instances: number of instances
        n_instances = len(instances)
        
        # n_vertices[i]: number of vertices in the i-th instance
        # note that instance = Ma, vertex_cover
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)

        # Compute matrices M, W, CV, CE
        # and vectors edges_mask and vertex_cover_exists

        #EV : Edge - vertex adjacency matrix
        EV              = np.zeros((total_edges,total_vertices))
        D               = np.zeros((total_vertices,1))
        #W               = np.zeros((total_edges,1))
        ######################################################################## C is defined in each instance #######################
        C               = np.zeros((total_vertices,1))
        # C               = np.zeros((total_vertices,1))
        #C               = np.zeros((total_edges,1))

        # Even index instances are UNSAT, odd are SAT
        vertex_cover_exists = np.array([ i%2 for i in range(n_instances) ])

        for (i,(Ma, vertex_cover, vertex_degrees)) in enumerate(instances):
            # Get the number of vertices (n) and edges (m) in this graph
            n, m = n_vertices[i], n_edges[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Populate EV
            for e,(x,y) in enumerate(edges):
                EV[m_acc+e,n_acc+x] = 1
                EV[m_acc+e,n_acc+y] = 1
            #end
            # Compute the cost of the optimal vertex_cover
            cost = len(vertex_cover) / n
            # cost = sum([ Mw[min(x,y),max(x,y)] for (x,y) in zip(vertex_cover,vertex_cover[1:]+vertex_cover[1:]) ]) / n

            # if target_cost is None:
            #     C[i] = (1-dev)*cost if i%2 == 0 else (1+dev)*cost
            # else:
            #     C[i] = target_cost
            # #end      
            if target_cost is None:
                C[n_acc:n_acc+n,0] = (1-dev)*cost if i%2 == 0 else (1+dev)*cost
            else:
                C[n_acc:n_acc+n,0] = target_cost
            #end            

            # if target_cost is None:
            #     C[m_acc:m_acc+m,0] = (1-dev)*cost if i%2 == 0 else (1+dev)*cost
            # else:
            #     C[m_acc:m_acc+m,0] = target_cost
            # #end
            for v, degree in enumerate(vertex_degrees):
                D[n_acc + v, 0] = degree
        #end
        return EV, C, D, vertex_cover_exists, n_vertices, n_edges
    #end

    def get_batches(self, batch_size, dev):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            yield InstanceLoader.create_batch(instances, dev=dev)
        #end
    #end

    def reset(self):
        random.shuffle(self.filenames)
        self.index = 0
    #end
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        # Mw = np.zeros((n,n),dtype=float)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end

        # # Parse edge weights
        # while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
        # for i in range(n):
        #     Mw[i,:] = [ float(x) for x in f.readline().split() ]
        # #end
        while 'VERTEX_DEGREE' not in line: line = f.readline();
        vertex_degrees = [float(x) for x in f.readline().split()]
        while 'VERTEX_COVER' not in line: line = f.readline();
        vertex_cover = [ int(x) for x in f.readline().split() ]

    #end
    return Ma, vertex_cover, vertex_degrees
#end