#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la

if __name__ == "__main__":
    W = np.random.rand(96, 7, 7, 3)
    print("||W|| == " + str(la.norm(W)))
