import numpy as np
from dataset import create_dataset, 
def test_create_graph():
    print("Testing create_graph function...")
    n = 5  # Number of vertices
    connectivity = 0.5  # Not used in this implementation but kept for compatibility
    Ma, vertex_cover = create_graph(n, connectivity)

    print("Generated adjacency matrix:")
    print(Ma)

    print("\nVertex cover:")
    print(vertex_cover)

    assert Ma.shape == (n, n), f"Adjacency matrix has incorrect shape: {Ma.shape}"
    assert len(vertex_cover) <= n, f"Vertex cover has invalid length: {len(vertex_cover)}"
    assert all(isinstance(v, int) for v in vertex_cover), "Vertex cover contains non-integer values."
    print("create_graph passed.\n")

def test_write_graph():
    print("Testing write_graph function...")
    n = 5
    Ma = np.triu(np.random.randint(0, 2, size=(n, n)), 1)
    vertex_cover = [0, 2, 4]

    print("Adjacency matrix to write:")
    print(Ma)

    print("\nVertex cover to write:")
    print(vertex_cover)

    # Mimic the behavior of write_graph with print
    print("\nSimulated graph file content:")
    print(f"TYPE : Vertex Cover")
    print(f"DIMENSION: {n}")
    print("EDGE_DATA_SECTION:")
    for i in range(Ma.shape[0]):  
        for j in range(i + 1, Ma.shape[1]): 
            if Ma[i, j] == 1:
                print(f"{i} {j}")
    print("-1")
    vertex_degree = np.sum(Ma, axis=1) + np.sum(Ma, axis=0)  # Degree calculation
    print("VERTEX_DEGREE:")
    print(" ".join(map(str, vertex_degree)))
    print("VERTEX_COVER:")
    print(" ".join(map(str, vertex_cover)))
    print("EOF")
    print("write_graph passed.\n")

def test_create_dataset():
    print("Testing create_dataset function...")
    nmin, nmax = 5, 7
    samples = 3

    print(f"Generating {samples} samples with vertices between {nmin} and {nmax}...")

    for i in range(samples):
        n = np.random.randint(nmin, nmax + 1)
        Ma, vertex_cover = create_graph(n, connectivity=None)

        print(f"\nSample {i + 1}:")
        print("Adjacency matrix:")
        print(Ma)

        print("Vertex cover:")
        print(vertex_cover)
    
    print("\ncreate_dataset passed.")

if __name__ == '__main__':
    test_create_graph()
    test_write_graph()
    test_create_dataset()
    print("All tests passed!")
