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

def monochromatic_approx(W, num_colors=6, even=False):
    W = W.transpose([0, 3, 1, 2])
    print("W after permutation == %s" % (str(W.shape)))
    even = False # litekmeans not implemented yet
    Wmono = W
    C = []
    S = []
    approx0 = []

    for f in range(0, np.shape(W)[0]):
        folded_filter = np.squeeze(W[f, :, :]).reshape((W.shape[1], -1))
        if f == 0:
            print("folded_filter--%s" % (str(folded_filter.shape)))
        (u, s, vt) = la.svd(folded_filter, full_matrices=False)
        if f == 0:
            print("u--%s s--%s vt--%s" % (str(u.shape), str(s.shape), str(vt.shape)))
        s = np.diag(s)
        v = vt.transpose()
        C.append(u[:, 0, np.newaxis].squeeze())
        S.append(s[0, np.newaxis, 0, np.newaxis] * v[:, 0, np.newaxis].squeeze())
        chunk = u[:, 0, np.newaxis] * s[0, 0, np.newaxis, np.newaxis] * vt[0, np.newaxis, :]
        approx0.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3]))
    C = np.asarray(C)
    S = np.asarray(S)
    approx0 = np.asarray(approx0)

    # C = whiten(C)

    if even:
        None # implement litekmeans
    else:
        max_iter = 1000
        (colors, assignment) = kmeans2(C, num_colors, minit="points", iter=max_iter)

    Wapprox = np.zeros(W.shape)

    assignment = assignment.reshape((assignment.size, 1))
    print("C--" + str(C.shape))
    print("assignment--%s colors--%s" % (str(assignment.shape), str(colors.shape)))

    Wapprox = []

    for f in range(0, np.shape(W)[0]):
        if f == 0:
            print("colors[assignment[f]].transpose()--%s * S[f]--%s" % (str(colors[assignment[f]].transpose().shape), str(S[f].shape)))
        chunk = colors[assignment[f]].transpose() * S[f]
        if f == 0:
            print("chunk--%s" % (str(chunk.shape)))
        Wapprox.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3]))

    Wapprox = np.asarray(Wapprox).transpose([0, 2, 3, 1])

    Wmono = S.reshape(W.shape[0], W.shape[2], W.shape[3])

    perm = np.argsort(assignment)
    colors = colors.transpose()
    num_weights = colors.size + Wmono.size
    print("num_weights == %s" % (str(num_weights)))

    return [Wapprox, Wmono, perm, num_weights]
