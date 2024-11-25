import os
import numpy as np
from instance_loader import InstanceLoader, read_graph
from model import build_network
import tensorflow as tf

# Define hyperparameters
d = 64  # Embedding dimension
learning_rate = 1e-2
batch_size = 4
time_steps = 5
num_epochs = 5  # Number of training epochs

# Build the GNN model
GNN = build_network(d)

# Initialize a TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Generate mock data for testing
    num_graphs = batch_size
    max_vertices = 10  # Maximum vertices in a graph

    # Mock vertex_cover_exists (labels for each graph)
    vertex_cover_exists = np.random.randint(0, 2, size=(num_graphs,)).astype(np.float32)

    # Mock n_vertices (number of vertices per graph)
    n_vertices = np.random.randint(1, max_vertices + 1, size=(num_graphs,)).astype(np.int32)

    # Mock EV_matrix (Edge-Vertex incidence matrix)
    total_edges = sum(n_vertices * (n_vertices - 1) // 2)  # Assume dense graphs
    total_vertices = sum(n_vertices)
    EV_matrix = np.random.randint(0, 2, size=(total_edges, total_vertices)).astype(np.float32)

    # Mock target_cost (cost per graph)
    target_cost = np.random.random(size=(num_graphs, 1)).astype(np.float32)

    # Prepare a feed dictionary
    feed_dict = {
        GNN['vertex_cover_exists']: vertex_cover_exists,
        GNN['n_vertices']: n_vertices,
        GNN['n_edges']: np.random.randint(1, total_edges + 1, size=(num_graphs,)).astype(np.int32),
        GNN['EV']: EV_matrix,
        GNN['C']: target_cost,
        GNN['time_steps']: time_steps,
    }

    # Training loop
    for epoch in range(num_epochs):
        # Run a forward pass to calculate initial loss and accuracy
        loss, acc, predictions = sess.run(
            [GNN['loss'], GNN['acc'], GNN['predictions']],
            feed_dict=feed_dict
        )

        # Print results for the current epoch
        print(f"Epoch {epoch + 1}")
        print("Loss:", loss)
        print("Accuracy:", acc)
        print("Predictions:", predictions)

        # Run the training step
        _, updated_loss = sess.run(
            [GNN['train_step'], GNN['loss']],
            feed_dict=feed_dict
        )

        # Print the updated loss after one training step
        print("Updated Loss after one training step:", updated_loss)
        print("-" * 50)  # Separator for epochs
