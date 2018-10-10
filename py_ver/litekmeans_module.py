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

    # out = I[:, 0, np.newaxis]
    out = I[:, 0]
    
    taille = []

    for m in range(M):
        taille.append(len([i for i in range(out) if out[0] == m]))

    taille = np.asarray(taille)

    hmany, nextclust = (np.max(taille), np.argmax(taille)[0]) # Note: nextclust should be an index, not a list

    choices = np.ones((N, 1))

    visited = [False for i in range(len(out))]

    while hmany > K:
        # aux = [x for x in nextclust if x in out]
        aux = np.where(out == nextclust)
        slice_ = []
        for a in aux:
            slice_.append(ds[a, choices[a] + 1 - ds[a, choices[a]]])
        slice_ = np.asarray(slice_)
        tempo = np.argsort(-slice_)

        saved = aux[tempo[0 : K]]
        out[saved] = nextclust

        visited[nextclust] = True
        for k in range(K + 1, tempo.size):
            i = 1
            while visited[I[aux[tempo[k]], i]]:
                i += 1
            out[aux[tempo[k]]] = I[aux[tempo[k]], i]
            choices[aux[tempo[k]]] = i
        for m in range(len(M)):
            taille[m] = len(np.where(out == m))
        hmany, nextclust = (np.max(taille), np.argmax(taille)[0]) # Note: nextclust should be an index, not a list

    ener = 0
    for n in range(len(N)):
        ener += w[n, out[n]]

    return [out, ener]

def litekmeans(X, k):
    outlabel = []
    outm = []

    n = X.shape[1]
    last = 0

    minener = 1e+20
    outiters = 30
    maxiters = 1000
    last = None

    for j in range(outiters):
        print("Iter %d / %d\n" % (j, outiters))
        aux = np.random.permutation(n)
        m = X[:, aux[0 : k]]
        (label, _) = constrained_assignment(X, m, n / k)
        (u, label) = np.unique(label, return_index=True, return_inverse=True) # remove empty clusters

        for iters in range(max(maxiters)):
            if last != None and last not in label:
                break
            None
            np.unique(label, )
        
    return [outlabel, outm]

if __name__ == "__main__":
    data = (np.arange(4) + 1).reshape(2, 2)
    print(str(data))
    ker = kernelizationbis(data, data)
    print(str(ker))
