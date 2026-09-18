import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from engine.network import SimpleNetwork
from engine.gradient_check import check_gradients

np.random.seed(0)
X = np.random.randn(5, 2)
y = np.array([0, 1, 0, 1, 1])

net = SimpleNetwork(input_dim=2, hidden_dim=4, seed=1)
result = check_gradients(net, X, y)
print()
print("All gradients passed" if result else "Some gradients failed")