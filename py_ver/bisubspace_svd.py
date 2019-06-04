#!/usr/bin/env python3

from litekmeans_module import litekmeans, global_seed
import numpy as np
from numpy import linalg as la
import pickle
from scipy.cluster.vq import kmeans2, whiten
import sys
import tf_op

def bisubspace_svd_approx(W, iclust=2, iratio=0.4, oclust=2, oratio=0.4, conseq=False, in_s=0, out_s=0, seed=None):
    W.shape # (filters, height, width, channels)
    # W = W.transpose([0, 3, 1, 2]) # [filters, channels, height, width]
    print("iclust = %d, iratio = %f, oclust = %d, oratio = %f, conseq = %d" % (iclust, iratio, oclust, oratio, conseq))
    print("in_s == %d ; out_s == %d ;" % (in_s, out_s))
    W_shape = np.asarray(W.shape, dtype=np.int)
    print("  W--" + str(W.shape))
    print("    " + str(W.shape[0]) + " filters")
    print("    " + str(W.shape[3]) + " channels")
    oclust_sz = W_shape[0] / oclust
    iclust_sz = W_shape[3] / iclust

    odegree = int(np.floor((W_shape[0] * oratio) / oclust))
    idegree = int(np.floor((W_shape[3] * iratio) / iclust))

    orig_ops = out_s * out_s * np.prod(W_shape)

    approx_ops = iclust * oclust * np.asarray([in_s * in_s * iclust_sz * idegree,
                                               out_s * out_s * idegree * odegree * W_shape[1] * W_shape[2],
                                               out_s * out_s * odegree * oclust_sz])
    # print("approx_ops == " + str(approx_ops))
    print("Input rank : %d" % (idegree))
    print("Output rank : %d" % (odegree))
    print("Gain : %f" % (orig_ops / np.sum(approx_ops)))
    print("Transform 1 : %f" % (approx_ops[0] / np.sum(approx_ops)))
    print("Conv : %f" % (approx_ops[1] / np.sum(approx_ops)))
    print("Transform 3 : %f" % (approx_ops[2] / np.sum(approx_ops)))
    # print("\n----------------\n")

    # conseq = not conseq
    if not conseq:
        WW = np.reshape(W, (W_shape[0], np.prod(W_shape[1:4])), order="F")
        # print(WW.transpose())
        idx_output = np.asarray(litekmeans(WW.transpose(), oclust, seed=seed)[0])
        # print("idx_output--" + str(idx_output.shape))
        # print("[" + ";".join([str(i + 1) for i in idx_output]) + "]")
        WW = W.transpose([3, 1, 2, 0])
        WW = np.reshape(WW, (WW.shape[0], np.prod(WW.shape[1:4])), order="F")
        idx_input = np.asarray(litekmeans(WW.transpose(), iclust, seed=seed)[0])
    else:
        # untested
        idx_input = np.zeros([iclust * int(iclust_sz)], dtype=np.int)
        idx_output = np.zeros([oclust * int(oclust_sz)], dtype=np.int)
        for i in range(iclust):
            idx_input[i * int(iclust_sz) : (i + 1) * int(iclust_sz)] = i
        for o in range(oclust):
            idx_output[o * int(oclust_sz) : (o + 1) * int(oclust_sz)] = o

    # print("\n----------------\n")
    C = np.zeros([W.shape[3] // iclust, idegree, iclust, oclust])
    Z = np.zeros([odegree, W.shape[1], W.shape[2], idegree, iclust, oclust])
    F = np.zeros([W.shape[0] // oclust, odegree, iclust, oclust])

    print("C--" + str(C.shape))
    print("Z--" + str(Z.shape))
    print("F--" + str(F.shape))
    for i in range(iclust):
        for o in range(oclust):
            oidx = idx_output == o
            oidx_indices = np.nonzero(oidx)[0]
            iidx = idx_input == i
            iidx_indices = np.nonzero(iidx)[0]

            Wtmp = W[oidx_indices, :, :, :]
            Wtmp = Wtmp[:, :, :, iidx_indices]
            # print("W--" + str(W.shape))
            # print("Wtmp--" + str(Wtmp.shape))
            Wtmp_shape = np.asarray(Wtmp.shape)
            Wtmp_ = np.reshape(Wtmp, (Wtmp_shape[0], np.prod(Wtmp_shape[1:4])), order="F")
            # print("Wtmp_--" + str(Wtmp_.shape))
            (u, s, vt) = la.svd(Wtmp_, full_matrices=True)
            v = vt.transpose()
            s = np.diag(s)
            s = np.concatenate((s, np.zeros([s.shape[0], v.shape[0] - s.shape[0]])), axis=1)
            # print("u--" + str(u.shape))
            # print("s--" + str(s.shape))
            # print("v--" + str(v.shape))
            F_ = np.matmul(u[:, 0:odegree], s[0:odegree, 0:odegree])
            # print("F_--" + str(F_.shape))
            Wtmptmp = np.matmul(F_, v[:, 0:odegree].transpose())
            F[:, :, i, o] = F_

            Wapprox_tmp = np.reshape(v[:, 0:odegree].transpose(), [odegree, Wtmp.shape[1], Wtmp.shape[2], Wtmp.shape[3]], order="F")
            Wapprox_tmp = Wapprox_tmp.transpose([3, 0, 1, 2])
            # print("Wapprox_tmp--" + str(Wapprox_tmp.shape))
            Wapprox_tmp_shape = np.asarray(Wapprox_tmp.shape)
            Wapprox_tmp_ = np.reshape(Wapprox_tmp, (Wapprox_tmp_shape[0], np.prod(Wapprox_tmp_shape[1:4])), order="F")
            (u, s, vt) = la.svd(Wapprox_tmp_, full_matrices=True)
            v = vt.transpose()
            s = np.diag(s)
            s = np.concatenate((s, np.zeros([s.shape[0], v.shape[0] - s.shape[0]])), axis=1)
            # print("u--" + str(u.shape))
            # print("s--" + str(s.shape))
            # print("v--" + str(v.shape))
            C_ = np.matmul(u[:, 0:idegree], s[0:idegree, 0:idegree])
            # print("C_--" + str(C_.shape))
            C[:, :, i, o] = C_
            Z_ = v[:, 0:idegree]
            # print("Z_--" + str(Z_.shape))
            Wtmptmptmp_ = np.matmul(C_, Z_.transpose())
            Z[:, :, :, :, i, o] = Z_.reshape([odegree, W.shape[1], W.shape[2], idegree], order="F")

    # print("\n----------------\n")
    Wapprox = np.zeros(W.shape)
    for i in range(iclust):
        for o in range(oclust):
            oidx = idx_output == o
            oidx_indices = np.nonzero(oidx)[0]
            iidx = idx_input == i
            iidx_indices = np.nonzero(iidx)[0]

            C_ = C[:, :, i, o]
            Z_ = Z[:, :, :, :, i, o]
            F_ = F[:, :, i, o]
            Z_ = Z_.transpose([3, 0, 1, 2])
            Z_ = Z_.reshape([Z_.shape[0], Z_.shape[1] * Z_.shape[2] * Z_.shape[3]], order="F").transpose()
            # print("Z_--" + str(Z_.shape))
            ZC = np.matmul(Z_, C_.transpose())
            # print("ZC--" + str(ZC.shape))

            Wtmptmptmp = ZC.reshape([odegree, W.shape[1], W.shape[2], int(iclust_sz)], order="F")
            Wtmptmptmp = Wtmptmptmp.reshape([Wtmptmptmp.shape[0], Wtmptmptmp.shape[1] * Wtmptmptmp.shape[2] * Wtmptmptmp.shape[3]], order="F")
            # print("F_--" + str(F_.shape))
            # print("Wtmptmptmp--" + str(Wtmptmptmp.shape))
            Wtmp = np.matmul(F_, Wtmptmptmp)
            # print("Wtmp--" + str(Wtmp.shape))
            Wtmp = Wtmp.reshape([int(oclust_sz), W.shape[1], W.shape[2], int(iclust_sz)], order="F")
            # print("Wtmp--" + str(Wtmp.shape))

            for ii in range(len(iidx_indices)):
                for oi in range(len(oidx_indices)):
                    Wapprox[oidx_indices[oi], : , : , iidx_indices[ii]] = Wtmp[oi, :, :, ii]

    # Wapprox = W.transpose([0, 2, 3, 1])

    print("norm(W) == %e" % (la.norm(W)))
    print("norm(Wapprox) == %e" % (la.norm(Wapprox)))
    L2_err = la.norm(W - Wapprox) / la.norm(W)
    print("||W - Wapprox|| / ||W|| == " + str(L2_err))
    print("\n----------------\n")

    # print("%e" % Wapprox[2, 2, 2, 2])

    return [Wapprox, C, Z, F, idx_input, idx_output]

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

    # print("W--" + str(W.shape))

    W_shape = np.asarray(W.shape, dtype=np.int)
    WW = np.reshape(W, (W_shape[0], np.prod(W_shape[1:4])), order="F")
    # print("WW--" + str(WW.shape))
    # print(WW)
    print("||W|| = %f" % la.norm(W))
    global_seed = 0
    [Wapprox, C, Z, F, idx_input, idx_output] = bisubspace_svd_approx(W, in_s=in_size, out_s=out_size)

    # litekmeans(X, number_of_seeds)
    # constrained_assignment(X, C, K)
