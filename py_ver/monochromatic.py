#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la

def monochromatic_approx(W, args):
    colors = args["num_colors"]
    even = args["even"]
    Wapprox = W
    Wmono = W
    return [Wapprox, Wmono, colors, even]
