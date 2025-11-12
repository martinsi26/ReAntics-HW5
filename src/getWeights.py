import numpy as np

data = np.load("weights.npz")

for key in data.files:
    arr = data[key]
    print(f"{key} = ", end="")

    if np.isscalar(arr):
        # single number
        print(f"{arr:.8e}\n")

    elif arr.ndim == 1:
        # 1D array
        print("[", ", ".join(f"{x:.8e}" for x in arr), "]\n")

    elif arr.ndim == 2:
        # 2D array
        print("[")
        for row in arr:
            print("    [", ", ".join(f"{x:.8e}" for x in row), "],")
        print("]\n")

    else:
        # higher dimensions
        print(f"# shape {arr.shape} not handled\n")
