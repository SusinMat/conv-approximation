#!/usr/bin/env python3

import numpy as np
import sys

global_seed = None

def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))

def print_2d_array(array):
    print(array.shape)
    array_string = "[ "
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_string += str(array[i, j]) + ", "
        array_string = array_string[:-2]
        array_string += "; "
    array_string = array_string[:-2]
    array_string += " ] "
    print(array_string)
    return

def get_rand_int(n):
    global global_seed
    m = 100003
    a = 1103515245
    c = 12345
    global_seed = (a * global_seed + c) % m
    r = global_seed % n
    return r

def random_permutation(n):
    array = np.arange(n, dtype=np.int)
    if n == 1:
        return array
    for i in range(n - 1, 0, -1):
        # j = np.random.randint(0, i)
        j = get_rand_int(i + 1)
        # print("j == " + str(j))
        aux = array[i]
        array[i] = array[j]
        array[j] = aux
    return array

def kernelizationbis(data, databis):
    L = data.shape[0]
    M = databis.shape[0]
    # print("data--" + str(data.shape))
    # print("databis--" + str(databis.shape))
    norms = np.sum(np.power(data, 2), 1, keepdims=True) * np.ones([1, M])
    normsbis = np.sum(np.power(databis, 2), 1, keepdims=True) * np.ones([1, L])
    # print("norms--" + str(norms.shape))
    # print("normsbis--" + str(normsbis.shape))
    ker = norms + normsbis.transpose() - (2.0 * np.dot(data, databis.transpose()))
    # print("ker--" + str(ker.shape))
    return ker


def constrained_assignment(X, C, K): # D?
    # assign samples to their nearest centers, with the constraint that each center receives K samples
    w = kernelizationbis(X.transpose(), C.transpose())
    K = int(K)
    # print("w == " + str(w))
    [N, M] = [w.shape[0], w.shape[1]]

    # maxvalue = np.max(w[:]) + 1
    ds = np.sort(w, 1)
    I = np.argsort(w, 1)
    # print("ds == " + str(ds))
    # print("I == " + str(I))
    # out = I[:, 0, np.newaxis]
    out = I[:, 0]
    # print("out--" + str(out.shape))
    # print(out)
    taille = []
    for m in range(M):
        found = np.where(out == m)[0]
        # print(found)
        taille.append(len(found))
    # print("taille == " + str(taille))
    nextclust = np.argmax(taille)
    hmany = taille[nextclust]
    # print("nextclust == " + str(nextclust))
    # print("hmany == %d ; nextclust == %d" % (hmany, nextclust))

    visited = np.zeros(M, dtype=np.int)
    choices = np.zeros(N, dtype=np.int)

    while hmany > K:
        aux = np.where(out == nextclust)
        aux = np.asarray(aux, dtype=np.int)
        aux = aux.flatten()
        slice_ = []
        for l in range(aux.shape[0]):
            slice_.append(ds[aux[l], choices[aux[l]] + 1] - ds[aux[l], choices[aux[l]]])
        slice_ = np.asarray(slice_)
        tempo = np.argsort(-slice_)

        # print("tempo[0:K] ==\n    " + str(tempo[0 : K]))
        saved = aux[tempo[0 : K]]
        out[saved] = nextclust

        visited[nextclust] = 1
        for k in range(K, len(tempo)):
            i = 1
            while visited[I[aux[tempo[k]], i]] != 0:
                i += 1
            out[aux[tempo[k]]] = I[aux[tempo[k]], i]
            choices[aux[tempo[k]]] = i
        for m in range(M):
            taille[m] = len(np.where(out == m)[0])
        # print("taille == " + str(taille))
        nextclust = np.argmax(taille)
        hmany = taille[nextclust]

    ener = 0
    for n in range(N):
        ener += w[n, out[n]]

    return [out, ener]


def litekmeans(X, k, seed=0):
    # X : d-by-n data matrix
    # k : number of seeds
    seed = np.random.randint(10000000)
    global global_seed

    n = X.shape[1]
    last = 0

    minener = 1e20
    outiters = 300
    maxiters = 10000

    # np.random.seed(seed=seed)
    global_seed = seed

    for j in range(outiters):
        # print("* Iter %d / %d" % (j + 1, outiters), file=sys.stderr)
        # aux = [i - 1 for i in aux]
        # aux = np.array(aux)
        # aux = random_permutation(n)
        aux = np.random.permutation(n)
        m = X[:, aux[:k]]
        [label, _] = constrained_assignment(X, m, n / k)
        assignment_distribution = np.zeros([k], dtype=np.int)
        for assignment in label:
            assignment_distribution[assignment] += 1
        # print("assignment_distribution == " + str(assignment_distribution))

        iters = 0
        while np.any(label != last) and iters < maxiters:
            [u, label] = np.unique(label, return_inverse=True)
            k = len(u)
            E = np.zeros([n, k])
            for i in range(n):
                E[i, label[i]] = 1
            diag = np.diag(np.power(sum(E, 0).transpose(), -1), k=0)
            if diag.shape != (k, k):
                print("Error: diagonal matrix is not k-by-k. k == %d diag.shape == %s" % (k, str(diag.shape)))
                exit(-1)
            m = np.dot(X, np.dot(E, diag))
            # print("m--" + str(m.shape))
            last = label
            [label, ener] = constrained_assignment(X, m, n / k)
            # print(label)
            assignment_distribution = np.zeros([k], dtype=np.int)
            for assignment in label:
                assignment_distribution[assignment] += 1
            # print("assignment_distribution == " + str(assignment_distribution))
            iters += 1

        [_, label] = np.unique(label, return_inverse=True)

        if ener < minener:
            outlabel = label
            outm = m
            minener = ener

    return [outlabel, outm]
