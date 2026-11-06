import numpy as np
from nn import forward_pass

saved = np.load("model.npz")
params = {name: saved[name] for name in saved.files}

while True:
    n1 = input("First number (0 or 1): ")
    n2 = input("Second number (0 or 1): ")
    if n1 in ("0", "1") and n2 in ("0", "1"):
        break
    print("only 0 or 1 are valid inputs!")

x = np.array([[int(n1), int(n2)]])
a, *_ = forward_pass(x, params)
print("probability:", a[0, 0], "-> prediction:", int(a[0, 0] > 0.5))