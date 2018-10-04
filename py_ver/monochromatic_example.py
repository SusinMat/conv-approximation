#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la
from monochromatic import monochromatic_approx

if __name__ == "__main__":
    W = np.abs(np.random.rand(96, 7, 7, 3))
    print("||W|| == " + str(la.norm(W)))

    [Wapprox, Wmono, colors, perm] = monochromatic_approx(W, num_colors=6, even=False)
    L2_err = la.norm(W - Wapprox) / la.norm(W)

    print("||W - Wapprox|| / ||W|| == " + str(L2_err));
