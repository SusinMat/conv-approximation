#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la


# This approximation clusters the first left singular vectors of each of
# the convolution kernels associated with each output feature. Filters in
# the same cluster share the same inner color component. The reconstructed
# weight matrix, Wapprox, is returned along with the the color
# transformation matrix, the monochromatic weights and the permutation of
# the weights. These matrices can be used to more efficiently compute the
# output of the convolution.
#
# args.even : True if clusters should be constrained to be equal sizes;
#             False otherwise
# args.num_colors : number of clusters (or "colors") to use

def monochromatic_approx(W, args):
    W = np.transpose(W, [0, 3, 1, 2])
    num_colors = args["num_colors"]
    even = args["even"]
    Wapprox = W
    Wmono = W
    C = []
    S = []
    approx0 = []
    for f in range(0, np.shape(W)[0]):
        folded_filter = np.squeeze(W[f, :, :]).reshape((W.shape[1], -1))
        (u, s, v) = la.svd(folded_filter, full_matrices=False)
        s = np.diag(s)
        vt = v.transpose()
        C.append(u[:, 0])
        S.append(s[0, 0] * vt[:, 0])
        chunk = u[:, 0] * s[0, 0] * v[:, 0]
        print(chunk.shape)
        approx0.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3],))
    C = np.asarray(C)
    S = np.asarray(S)
    approx0 = np.asarray(approx0)
    # print(np.shape(folded_filter))
    # print(np.shape(u))
    # print(np.shape(s))
    # print(np.shape(v))
    # print(np.shape(C))

    return [Wapprox, Wmono, num_colors, even]
