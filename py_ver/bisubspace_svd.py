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

def bisubspace_svd_approx(W, iclust=2, iratio=0.4, oclust=0.4, oratio=2, conseq=0, in_s=55, out_s=51):
    Wapprox = W
    W.shape # (filters, height, width, channels)

    return [Wapprox]
