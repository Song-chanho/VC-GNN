# VC-GNN
Graph Neural Network architecture to solve the decision variant of the Vertex Cover problem

OBS. To run this code you must install gurobi optimizer first.

![](/figures/route-examples.png)

Upon training with -2%, +2% from the optimal cost, the model is able to achieve >90% test accuracy. It also learns to generalize for different graph distributions & larger instance sizes (with decreasing accuracy) and more relaxed deviations (with better accuracy).
