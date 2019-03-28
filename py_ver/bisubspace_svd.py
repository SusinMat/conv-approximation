#!/usr/bin/env python3

from litekmeans_module import litekmeans
import numpy as np
from numpy import linalg as la
import pickle
from scipy.cluster.vq import kmeans2, whiten
import sys
import tf_op

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

def kernelizationbis(data, databis):
    L = data.shape[0]
    M = databis.shape[0]
    print("data--" + str(data.shape))
    print("databis--" + str(databis.shape))
    norms = np.sum(np.power(data, 2), 1, keepdims=True) * np.ones([1, M])
    normsbis = np.sum(np.power(databis, 2), 1, keepdims=True) * np.ones([1, L])
    print("norms--" + str(norms.shape))
    print("normsbis--" + str(normsbis.shape))
    ker = norms + normsbis.transpose() - (2.0 * np.dot(data, databis.transpose()))
    print("ker--" + str(ker.shape))
    return ker

def constrained_assignment(X, C, K): # D?
    # assign samples to their nearest centers, with the constraint that each center receives K samples
    w = kernelizationbis(X.transpose(), C.transpose())
    # print("w == " + str(w))
    [N, M] = [w.shape[0], w.shape[1]]

    # maxvalue = np.max(w[:]) + 1
    ds = np.sort(w, 1)
    I = np.argsort(w, 1)
    # print("ds == " + str(ds))
    # print("I == " + str(I))
    # out = I[:, 0, np.newaxis]
    out = I[:, 0]
    print("out--" + str(out.shape))
    # print(out)
    taille = []
    for m in range(M):
        found = np.where(out == m)
        # print(found)
        found = found[0]
        taille.append(len(found))
    print("taille == " + str(taille))
    nextclust = np.argmax(taille)
    hmany = taille[nextclust]
    print("nextclust == " + str(nextclust))
    print("hmany == %d ; nextclust == %d" % (hmany, nextclust))

    visited = np.zeros([M], dtype=np.int)
    choices = np.zeros([N, 1], dtype=np.int)

    while hmany > K:
        aux = np.where(out == nextclust)
        aux = np.asarray(aux, dtype=np.int)
        slice_ = []
        for l in range(len(aux)):
            slice_.append(ds[aux[l], choices[aux[l]] + 1] - ds[aux[l], choices[aux[l]]])
        slice_ = np.asarray(slice_)
        tempo = np.argsort(-slice_)

        saved = aux[tempo[0:K]]
        out[saved] = nextclust

        visited[nextclust] = 1
        for k in range(K, len(tempo)):
            i = 1
            while visited[I[aux[tempo[k]], i]] != 0:
                i += 1
            out[aux[tempo[k]]] = I[aux[tempo[k]], i]
            choices[aux[tempo[k]]] = i
        for m in range(M):
            taille[m] = len(np.where(out == m))
        nextclust = np.argmax(taille)
        hmany = taille[nextclust]

    ener = 0
    for n in range(N):
        ener += w[n, out[n]]

    return [out, ener]

def litekmeans(X, k):
    # X : d-by-n data matrix
    # k : number of seeds

    n = X.shape[1]
    last = 0

    minener = 1e20
    outiters = 1
    maxiters = 1000

    for j in range(outiters):
        print("* Iter %d / %d" % (j + 1, outiters), file=sys.stderr)
        np.random.seed(seed=0)
        aux = [i + 1 for i in np.random.permutation(n)]
        aux = [64, 63, 43, 24, 7, 15, 45, 12, 20, 19, 46, 18, 60, 10, 29, 6, 50, 62, 23, 13, 2, 61, 41, 27, 22, 16, 25, 14, 34, 9, 31, 39, 1, 59, 35, 3, 58, 47, 17, 52, 44, 33, 36, 40, 49, 56, 28, 51, 57, 4, 54, 11, 38, 8, 32, 48, 37, 55, 5, 26, 21, 30, 42, 53]
        aux = [i - 1 for i in aux]
        aux = np.array(aux)
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

def bisubspace_svd_approx(W, iclust=2, iratio=0.4, oclust=2, oratio=0.4, conseq=False, in_s=0, out_s=0):
    W.shape # (filters, height, width, channels)
    # W = W.transpose([0, 3, 1, 2]) # [filters, channels, height, width]
    print("iclust = %d, iratio = %f, oclust = %d, oratio = %f, conseq = %d" % (iclust, iratio, oclust, oratio, conseq))
    print("in_s == %d ; out_s == %d ;" % (in_s, out_s))
    W_shape = np.asarray(W.shape, dtype=np.int)
    oclust_sz = W_shape[0] / oclust
    iclust_sz = W_shape[3] / iclust

    odegree = np.floor((W_shape[0] * oratio) / oclust)
    idegree = np.floor((W_shape[3] * iratio) / iclust)

    orig_ops = out_s * out_s * np.prod(W_shape)

    approx_ops = iclust * oclust * np.asarray([in_s * in_s * iclust_sz * idegree,
                                               out_s * out_s * idegree * odegree * W_shape[1] * W_shape[2],
                                               out_s * out_s * odegree * oclust_sz])
    # print("approx_ops == " + str(approx_ops))
    print("Input rank : %d" % (idegree))
    print("Output rank : %d" % (odegree))
    print("Gain : %f" % (orig_ops / np.sum(approx_ops)))
    print("Tramsform 1 : %f" % (approx_ops[0] / np.sum(approx_ops)))
    print("Conv : %f" % (approx_ops[1] / np.sum(approx_ops)))
    print("Tramsform 3 : %f" % (approx_ops[2] / np.sum(approx_ops)))
    print("----------------")

    if not conseq:
        WW = np.reshape(W, (W_shape[0], np.prod(W_shape[1:4])), order="F")
        print(WW.transpose())
        idx_output = litekmeans(WW.transpose(), oclust)
        # print_2d_array(WW)
        # WW = W.transpose([3, 1, 2, 0])
        # WW = WW[:, :]
        # idx_input = litekmeans(WW.transpose(), iclust)

    # Wapprox = W.transpose([0, 2, 3, 1])
    Wapprox = W
    return [Wapprox]

if __name__ == "__main__":
    op = pickle.load(open("layer.pkl", "rb"))
    W = op.inputs[1].data
    # print("[%s];" % (" ".join([("%e" % i) for i in W.flatten().tolist()])))
    input_image = op.inputs[0]
    output_image = op.outputs[0]
    in_size = input_image.shape[1]
    out_size = output_image.shape[1]
    X = np.random.randn(288 * 64).reshape([64, 288])
    C = np.random.randn(2 * 64).reshape([64, 2])

    K = 10
    number_of_seeds = 2

    print("W--" + str(W.shape))

    W_shape = np.asarray(W.shape, dtype=np.int)
    WW = np.reshape(W, (W_shape[0], np.prod(W_shape[1:4])), order="F")
    print("WW--" + str(WW.shape))
    # print(WW)
    print("||W|| = %f" % la.norm(W))
    bisubspace_svd_approx(W, in_s=in_size, out_s = out_size)

    # litekmeans(X, number_of_seeds)
    # constrained_assignment(X, C, K)
