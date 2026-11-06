import numpy as np


def forward_pass(inputs, params):
    """Returns prediction (examples, 1), plus hidden_z and hidden_a for backprop."""
    hidden_z = inputs @ params["W1"] + params["b1"]      # (examples, 3)
    hidden_a = np.maximum(0, hidden_z)                   # ReLU
    output_z = hidden_a @ params["W2"] + params["b2"]    # (examples, 1)
    prediction = 1 / (1 + np.exp(-output_z))             # sigmoid
    return prediction, hidden_z, hidden_a


