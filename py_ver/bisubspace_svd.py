#!/usr/bin/env python3

from litekmeans_module import litekmeans
import numpy as np
from numpy import linalg as la
import pickle
from scipy.cluster.vq import kmeans2, whiten
import sys
import tf_op

seed = 0

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
    global seed
    m = 100003
    a = 1103515245
    c = 12345
    seed = (a * seed + c) % m
    # print("seed after == " + str(seed))
    r = seed % n
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

        print("tempo[0:K] ==\n    " + str(tempo[0 : K]))
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
    outiters = 30
    maxiters = 1000

    for j in range(outiters):
        print("* Iter %d / %d" % (j + 1, outiters), file=sys.stderr)
        np.random.seed(seed=0)
        # aux = [i + 1 for i in np.random.permutation(n)]
        # aux = [64, 63, 43, 24, 7, 15, 45, 12, 20, 19, 46, 18, 60, 10, 29, 6, 50, 62, 23, 13, 2, 61, 41, 27, 22, 16, 25, 14, 34, 9, 31, 39, 1, 59, 35, 3, 58, 47, 17, 52, 44, 33, 36, 40, 49, 56, 28, 51, 57, 4, 54, 11, 38, 8, 32, 48, 37, 55, 5, 26, 21, 30, 42, 53]
        # aux = [i - 1 for i in aux]
        # aux = np.array(aux)
        aux = random_permutation(n)
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

    # conseq = not conseq
    if not conseq:
        WW = np.reshape(W, (W_shape[0], np.prod(W_shape[1:4])), order="F")
        # print(WW.transpose())
        idx_output = np.asarray(litekmeans(WW.transpose(), oclust)[0])
        print("idx_output--" + str(idx_output.shape))
        # print("[" + ";".join([str(i + 1) for i in idx_output]) + "]")
        WW = W.transpose([3, 1, 2, 0])
        WW = np.reshape(WW, (WW.shape[0], np.prod(WW.shape[1:4])), order="F")
        idx_input = np.asarray(litekmeans(WW.transpose(), iclust)[0])
    else:
        # untested
        idx_input = np.zeros([iclust * int(iclust_sz)], dtype=np.int)
        idx_output = np.zeros([oclust * int(oclust_sz)], dtype=np.int)
        for i in range(iclust):
            idx_input[i * int(iclust_sz) : (i + 1) * int(iclust_sz)] = i
        for o in range(oclust):
            idx_output[o * int(oclust_sz) : (o + 1) * int(oclust_sz)] = o

    C = np.zeros([W.shape[3] // iclust, int(idegree), iclust, oclust])
    Z = np.zeros([int(odegree), W.shape[1], W.shape[2], int(idegree), iclust, oclust])
    F = np.zeros([W.shape[0] // oclust, int(odegree), iclust, oclust])

    print("C--" + str(C.shape))
    print("Z--" + str(Z.shape))
    print("F--" + str(F.shape))
    for i in range(iclust):
        for o in range(oclust):
            oidx = idx_output == o
            iidx = idx_input == i
            oidx_indices = np.nonzero(oidx)[0]
            iidx_indices = np.nonzero(iidx)[0]

            Wtmp = W[oidx_indices, :, :, :]
            Wtmp = Wtmp[:, :, :, iidx_indices]
            print("W--" + str(W.shape))
            print("Wtmp--" + str(Wtmp.shape))
            Wtmp_shape = np.asarray(Wtmp.shape)
            Wtmp_ = np.reshape(Wtmp, (Wtmp_shape[0], np.prod(Wtmp_shape[1:4])), order="F")
            (u, s, vt) = la.svd(Wtmp_, full_matrices=True)
            v = vt.transpose()
            s = np.diag(s)
            s = np.concatenate((s, np.zeros([s.shape[0], v.shape[0] - s.shape[0]])), axis=1)
            print("u--" + str(u.shape))
            print("s--" + str(s.shape))
            print("v--" + str(v.shape))
            F_ = np.matmul(u[:, 0:int(odegree)], s[0:int(odegree), 0:int(odegree)])
            print("F_--" + str(F_.shape))
            Wtmptmp = np.matmul(F_, v[:, 0:int(odegree)].transpose())
            F[:, :, i, o] = F_

            Wapprox_tmp = np.reshape(v[:, 0:int(odegree)].transpose(), [int(odegree), Wtmp.shape[1], Wtmp.shape[2], Wtmp.shape[3]], order="F")
            Wapprox_tmp = Wapprox_tmp.transpose([3, 0, 1, 2])
            print("Wapprox_tmp--" + str(Wapprox_tmp.shape))
            Wapprox_tmp_shape = np.asarray(Wapprox_tmp.shape)
            Wapprox_tmp_ = np.reshape(Wapprox_tmp, (Wapprox_tmp_shape[0], np.prod(Wapprox_tmp_shape[1:4])), order="F")
            (u, s, vt) = la.svd(Wapprox_tmp_, full_matrices=True)
            v = vt.transpose()
            s = np.diag(s)
            s = np.concatenate((s, np.zeros([s.shape[0], v.shape[0] - s.shape[0]])), axis=1)
            print("u--" + str(u.shape))
            print("s--" + str(s.shape))
            print("v--" + str(v.shape))
            C_ = np.matmul(u[:, 0:int(idegree)], s[0:int(idegree), 0:int(idegree)])
            print("C_--" + str(C_.shape))
            C[:, :, i, o] = C_
            Z_ = v[:, 0:int(idegree)]
            print("Z_--" + str(Z_.shape))
            # Wtmptmptmp_ = np.matmul(C_, Z_.transpose())
            Z[:, :, :, :, i, o] = Z_.reshape([int(odegree), W.shape[1], W.shape[2], int(idegree)], order="F")

    print("")
    Wapprox = np.zeros(W.shape)
    for i in range(iclust):
        for o in range(oclust):
            oidx = idx_output == o
            iidx = idx_input == i

            C_ = C[:, :, i, o]
            Z_ = Z[:, :, :, :, i, o]
            F_ = F[:, :, i, o]
            Z_ = Z_.transpose([3, 0, 1, 2])
            Z_ = Z_.reshape([Z_.shape[0], Z_.shape[1] * Z_.shape[2] * Z_.shape[3]], order="F").transpose()
            ZC = np.matmul(Z_, C_.transpose())
            print("ZC--" + str(ZC.shape))

            Wtmptmptmp = ZC.reshape([int(odegree), W.shape[1], W.shape[2], int(iclust_sz)], order="F")
            Wtmptmptmp = Wtmptmptmp.reshape([Wtmptmptmp.shape[0], Wtmptmptmp.shape[1] * Wtmptmptmp.shape[2] * Wtmptmptmp.shape[3]], order="F")
            print("Wtmptmptmp--" + str(Wtmptmptmp.shape))
            Wtmp = np.matmul(F_, Wtmptmptmp)
            Wtmp = Wtmp.reshape([int(oclust_sz), W.shape[1], W.shape[2], int(iclust_sz)], order="F")
            print("Wtmp--" + str(Wtmp.shape))

            oidx_indices = np.nonzero(oidx)[0]
            iidx_indices = np.nonzero(iidx)[0]
            for ii in range(len(iidx_indices)):
                for oi in range(len(oidx_indices)):
                    Wapprox[oidx_indices[oi], : , : , iidx_indices[ii]] = Wtmp[oi, :, :, ii]

    # Wapprox = W.transpose([0, 2, 3, 1])

    print("norm(W) == %e" % (la.norm(W)))
    print("norm(Wapprox) == %e" % (la.norm(Wapprox)))
    L2_err = la.norm(W - Wapprox) / la.norm(W)
    print("||W - Wapprox|| / ||W|| == " + str(L2_err))

    # print("%e" % Wapprox[2, 2, 2, 2])

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
    seed = 0
    bisubspace_svd_approx(W, in_s=in_size, out_s = out_size)

    # litekmeans(X, number_of_seeds)
    # constrained_assignment(X, C, K)
