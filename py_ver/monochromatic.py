#!/usr/bin/env python3

from litekmeans_module import litekmeans
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
# args.even : True if clusters must be constrained to be of equal sizes;
#             False otherwise
# args.num_colors : number of clusters (or "colors") to use

def monochromatic_approx(W, num_colors=6, even=False):
    W = W.transpose([0, 3, 1, 2]) # [filters, channels, height, width]
    print("W after permutation == %s" % (str(W.shape)))
    C = []
    S = []
    approx0 = []

    for f in range(0, np.shape(W)[0]):
        folded_filter = W[f].reshape((W.shape[1], -1)) # keep number of input channels, fold the other dimensions
        if f == 0:
            print("folded_filter--%s" % (str(folded_filter.shape)))
        (u, s, vt) = la.svd(folded_filter, full_matrices=False) # u * s * vt == original_matrix
        if f == 0:
            print("u--%s s--%s vt--%s" % (str(u.shape), str(s.shape), str(vt.shape)))
        s = np.diag(s) # s should be a diagonal matrix
        v = vt.transpose() # for parity with the original MatLab script, we transpose vt, obtaining v
        C.append(u[:, 0]) # First column of the u array
        if f == 0:
            print("s[0, np.newaxis, 0, np.newaxis]--%s * v[:, 0, np.newaxis]--%s == %s" % (str(s[0, np.newaxis, 0, np.newaxis].shape), str(v[:, 0, np.newaxis].shape), str((s[0, np.newaxis, 0, np.newaxis] * v[:, 0, np.newaxis]).shape)))
        S.append(s[0, np.newaxis, 0, np.newaxis] * v[:, 0, np.newaxis]) # First column of v multiplied by the first eigenvalue of W[f] (scalar)
        chunk = u[:, 0, np.newaxis] * s[0, 0] * vt[0, np.newaxis, :] # First column of u multiplied by first column of v multiplied by the first eigenvalue of W[f] (scalar)
        approx0.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3])) # unfold dimensions
    C = np.asarray(C)
    S = np.asarray(S).squeeze() # Remove the singleton dimension (3rd)
    approx0 = np.asarray(approx0)

    # C = whiten(C)

    if even:
        # Use litekmeans. Note that it must 'constrain clusters to be of equal sizes'
        (assignment, colors) = litekmeans(C.transpose(), num_colors)
        colors = colors.transpose()
    else:
        max_iter = 1000
        (colors, assignment) = kmeans2(C, num_colors, minit="points", iter=max_iter)

    Wapprox = np.zeros(W.shape)

    assignment = assignment.reshape((assignment.size, 1))
    print("C--" + str(C.shape))
    print("S--" + str(S.shape))
    print("assignment--%s colors--%s" % (str(assignment.shape), str(colors.shape)))

    Wapprox = []

    for f in range(0, np.shape(W)[0]):
        if f == 0:
            print("colors[assignment[f]].transpose()--%s * S[f]--%s" % (str(colors[assignment[f]].transpose().shape), str(S[f].shape)))
        chunk = colors[assignment[f]].transpose() * S[f] # Multiply the centroid to which filter f was assigned by the first eigenvalue multiplied by v[:, 0, np.newaxis]
        # Note that the centroid came from u[:, 0]
        if f == 0:
            print("chunk--%s" % (str(chunk.shape)))
        Wapprox.append(chunk.reshape(W.shape[1], W.shape[2], W.shape[3])) # [height, width, channels]

    Wapprox = np.asarray(Wapprox).transpose([0, 2, 3, 1]) # [filters, height, width, channels]


    Wmono = S.reshape(W.shape[0], W.shape[2], W.shape[3]) # [filters, height, width]

    perm = np.argsort(assignment)
    colors = colors.transpose()
    num_weights = colors.size + Wmono.size
    print("num_weights == %s" % (str(num_weights)))

    return [Wapprox, Wmono, perm, num_weights]
