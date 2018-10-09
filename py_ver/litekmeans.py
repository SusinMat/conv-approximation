#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la

def kernelizationbis(data, databis):
    ker = []
    kern = []

    L = data.shape[0]
    M = databis.shape[0]

    norms = np.sum(data ** 2, axis=1).reshape(-1, 1) * np.ones((1, M))
    normbis = np.sum(databis ** 2, axis=1).reshape(-1, 1) * np.ones((1, L))
    ker = norms + normbis.transpose() - 2 * data * databis.transpose()
    return ker

def constrained_assignment(X, C, K):
    out = []
    ener = []
    w = kernelizationbis(X.transpose(), C.transpose())
    N = w.shape[0] # number of samples
    M = w.shape[1] # number of centers

    maxvalue = np.max(w) + 1

    ds = sort(w, axis=1)
    I = argsort(w, axis=1)

    out = I[:, 0, np.newaxis]
    
    taille = []

    for m in range(M):
        taille.append(len([i for i in range(out) if out[i, 0] == m]))

    taille = np.asarray(taille)

    hmany, nextclust = (np.max(taille), np.argmax(taille))

    choices = np.ones((N, 1))

    while hmany > K:
        aux = [x for x in nextclust if x == out]

    return [out, ener]

def litekmeans(X, num_colors):
    outlabel = []
    outm = []
    return [outlabel, outm]

if __name__ == "__main__":
    data = (np.arange(4) + 1).reshape(2, 2)
    print(str(data))
    ker = kernelizationbis(data, data)
    print(str(ker))
