import numpy as np
import src.py_newton
def setup_mesh(N, a, b, c, d):
    return np.meshgrid(np.linspace(a, b, N), np.linspace(c, d, N))
