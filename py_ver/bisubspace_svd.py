#!/usr/bin/env python3

from litekmeans_module import litekmeans
import numpy as np
from numpy import linalg as la
from scipy.cluster.vq import kmeans2, whiten

def litekmeans(X, k):
    # X : d-by-n data matrix
    # k : number of seeds

    n = X.shape[1]
    last = 0

    minener = 1e20
    outiters = 30
    maxiters = 1000

    for j in range(outiers):
        print("Iter %d / %d" % (j, outiters))
        np.random.seed(seed=0)
        aux = np.random.permutation(n)
        m = X[:, aux[0:k]]
        label = constrained_assignment(X, m, n / k)[0]

        iters = 0
    return


def bisubspace_svd_approx(W, iclust=2, iratio=0.4, oclust=2, oratio=0.2, conseq=False, in_s=0, out_s=0):
    W.shape # (filters, height, width, channels)
    # W = W.transpose([0, 3, 1, 2]) # [filters, channels, height, width]
    print("iclust = %d, iratio = %f, oclust = %d, oratio = %f, conseq = %d" % (iclust, iratio, oclust, oratio, conseq))
    oclust_sz = W.shape[0] / oclust
    iclust_sz = W.shape[3] / iclust

    odegree = np.floor(W.shape[0] * oratio / oclust)
    idegree = np.floor(W.shape[3] * iratio / iclust)

    orig_ops = out_s * out_s * np.prod(W.shape)

    approx_ops = iclust * oclust * np.asarray([in_s * in_s * iclust_sz * idegree,
                                               out_s * out_s * idegree * odegree * W.shape[1] * W.shape[2],
                                               out_s * out_s * odegree * oclust_sz])
    print("Input rank : %d" % (idegree))
    print("Output rank : %d" % (odegree))
    print("Gain : %f" % (orig_ops / np.sum(approx_ops)))
    print("Tramsform 1 : %f" % (approx_ops[0] / np.sum(approx_ops)))
    print("Conv : %f" % (approx_ops[1] / np.sum(approx_ops)))
    print("Tramsform 3 : %f" % (approx_ops[2] / np.sum(approx_ops)))

    if not conseq:
        WW = np.reshape(W, (W.shape[0], np.prod(W.shape[1:4])))
        print(WW.shape)
        WW_string = "[ "
        for i in range(WW.shape[0]):
            for j in range(WW.shape[1]):
                WW_string += str(WW[i, j]) + ", "
            WW_string = WW_string[:-2]
            WW_string += "; "
        WW_string = WW_string[:-2]
        WW_string += " ] "
        print(WW_string)
        WW = W.transpose([3, 1, 2, 0])


    # Wapprox = W.transpose([0, 2, 3, 1])
    Wapprox = W
    return [Wapprox]
