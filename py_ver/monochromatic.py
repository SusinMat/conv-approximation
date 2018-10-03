#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la
from scipy.cluster.vq import kmeans2, whiten


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
    even = False # litekmeans not implemented yet
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
        C.append(u[:, 0, np.newaxis].squeeze())
        S.append(s[0, np.newaxis, 0, np.newaxis] * vt[:, 0, np.newaxis].squeeze())
        chunk = u[:, 0, np.newaxis] * s[0, 0, np.newaxis, np.newaxis] * v[0, np.newaxis, :]
        approx0.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3]))
    C = np.asarray(C)
    S = np.asarray(S)
    approx0 = np.asarray(approx0)
    print(approx0.shape)
    print(C.shape)

    if even:
        None # implement litekmeans
    else:
        max_iter = 1000
        (codebook, label) = kmeans2(C, num_colors, minit="points",iter=max_iter)
    print(codebook)
    print(label)

    Wapprox = np.zeros(W.shape)

    return [Wapprox, Wmono, num_colors, even]
