import os
import numpy as np
from instance_loader import InstanceLoader, read_graph

data_dir = "data"
loader = InstanceLoader(data_dir)


# 테스트: create_batch
print("=== Test: create_batch ===")
batch = list(loader.get_batches(batch_size=2, dev=0.02))
for i, (EV, C, vertex_cover_exists, n_vertices, n_edges) in enumerate(batch):
    print(f"Batch {i + 1}:")
    print("Edge-Vertex Incidence Matrix (EV):")
    print(EV)
    print("Cost Vector (C):")
    print(C)
    print("Vertex Cover Exists:", vertex_cover_exists)
    print("Number of Vertices:", n_vertices)
    print("Number of Edges:", n_edges)
    print()