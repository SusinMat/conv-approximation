#!/usr/bin/env python3

# Copyright 2018 SUSIN, Matheus
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHA    LL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTER    RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import ast
import enum
from monochromatic import monochromatic_approx
from bisubspace_svd import bisubspace_svd_approx
import numpy as np
from numpy import linalg as la
import os
import pickle
import re
import sys
import xorapu

from tf_op import Tensor, Op

global_seed = 0

def approximate(op, num_colors=4, even=False, strategy="", seed=None):
    global global_seed
    if seed is None:
        seed = global_seed
    else:
        global_seed = seed
    input_image = op.inputs[0]
    output_image = op.outputs[0]
    in_size = input_image.shape[1]
    out_size = output_image.shape[1]
    W = op.inputs[1].data
    bias = op.inputs[2].data
    if strategy == "monochromatic":
        return monochromatic_approx(W, num_colors=num_colors, even=even)
    elif strategy == "bisubspace_svd":
        return bisubspace_svd_approx(W, in_s=in_size, out_s=out_size, seed=seed)
    else:
        print("Error: approximation strategy '" + strategy + "' not supported.")
        exit(1)

if __name__ == "__main__":
    op = pickle.load(open("layer.pkl", "rb"))
    W = op.inputs[1].data
    # [Wapprox, Wmono, colors, perm, num_weights] = approximate(op)
    Wapprox = (approximate(op, strategy="bisubspace_svd"))[0]
    print(Wapprox.shape)

    L2_err = la.norm(W - Wapprox) / la.norm(W)

    print("||W - Wapprox|| / ||W|| == " + str(L2_err))
