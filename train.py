import sys
import numpy as np
from nn import forward_pass


GATES = {
    "and": [1, 0, 0, 0],
    "or":  [1, 1, 1, 0],
    "xor": [0, 1, 1, 0],
}

gate = sys.argv[1] if len(sys.argv) > 1 else "or"
if gate not in GATES:
    sys.exit(f"unknown gate '{gate}', choose from: {', '.join(GATES)}")

np.random.seed(0)

inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])      # (4, 2)
y = np.array(GATES[gate]).reshape(-1, 1)                 # (4, 1)

learning_rate = 0.1
m = inputs.shape[0]

params = {
    "W1": np.random.randn(2, 3),
    "b1": np.zeros((1, 3)),
    "W2": np.random.randn(3, 1),
    "b2": np.zeros((1, 1)),
}

for epoch in range(1000):
    # forward pass + cost
    a, hidden_z, hidden_a = forward_pass(inputs, params)
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    cost = np.mean(loss)

    # backpropagation
    d_output_z = (a - y) / m                                 # (4, 1)
    d_W2 = hidden_a.T @ d_output_z                           # (3, 1)
    d_b2 = np.sum(d_output_z, axis=0, keepdims=True)         # (1, 1)
    d_hidden_a = d_output_z @ params["W2"].T                 # (4, 3)
    d_hidden_z = d_hidden_a * (hidden_z > 0)                 # ReLU mask
    d_W1 = inputs.T @ d_hidden_z                             # (2, 3)
    d_b1 = np.sum(d_hidden_z, axis=0, keepdims=True)         # (1, 3)

    # update parameteres
    params["W1"] -= learning_rate * d_W1
    params["b1"] -= learning_rate * d_b1
    params["W2"] -= learning_rate * d_W2
    params["b2"] -= learning_rate * d_b2

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  cost: {cost:.4f}")


a, *_ = forward_pass(inputs, params)
print("classes:", (a > 0.5).astype(int).ravel(), "targets:", y.ravel())

np.savez("model.npz", gate=gate, **params)
print(f"saved model.npz (gate: {gate})")