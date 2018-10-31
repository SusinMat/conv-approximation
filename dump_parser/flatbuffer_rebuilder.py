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
import magic # package is called 'python-libmagic' in pip, and it also requires libmagic-dev to be installed in the system
import numpy as np
import os
import pickle
import re
import sys
import tensorflow as tf
import tempfile, subprocess

from tf_op import Tensor, Op, remove_successive_duplicates

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild a CNN (Convolutional Neural Network) given a .pkl generated by dump_parser.py")

    parser.add_argument("input_path", help="path to the output file of the input file, which must be a pickle dump generated by dump_parser.py")

    args = parser.parse_args()

    input_path = args.input_path

    mimetype = magic.Magic().from_file(input_path)

    basename = os.path.basename(input_path)
    (filename, file_extension) = os.path.splitext(basename)

    ops = []

    if mimetype == "application/octet-stream" and (file_extension in [".pkl", ".pickle"]):
        ops = pickle.load(open(input_path, "rb"))
    else:
        print("Input file: " + input_path + " is of unsupported type " + mimetype)
        exit(0)

    op_names = [op.name for op in ops]
    tensors = []
    for i in range(len(ops)):
        op = ops[i]
        op_name = op.name
        op_inputs = [tensor for tensor in op.inputs]
        op_outputs = [tensor for tensor in op.outputs]
        print("Name: %s, inputs: %s, outputs: %s, options: %s" % (op_name, str([tensor.index for tensor in op_inputs]), str([tensor.index for tensor in op_outputs]), op.options))
        for tensor in op_inputs:
            print(" * --> input " + str(tensor.index) + " : s=" + str(tensor.shape) + " <--")
            tensors.append(tensor)
        for tensor in op_outputs:
            print(" * <-- output " + str(tensor.index) + " : s=" + str(tensor.shape) + " -->")
            tensors.append(tensor)
    tensors.sort(key=lambda item: item.index)
    tensors = remove_successive_duplicates(tensors)
    # print([tensor.index for tensor in tensors])

    tf_tensors = []
    for tensor in tensors:
        data_type = None
        tf_tensor = None
        if tensor.type_name == "FLOAT32":
            data_type = tf.float32
        elif tensor.type_name == "INT32":
            data_type = tf.int32
        elif tensor.type_name == "UINT8":
            data_type = tf.uint8
        else:
            print("Unsupported data type " + tensor.type_name)
            exit(1)

        if type(tensor.data) == np.ndarray:
            # tf_tensor = tf.convert_to_tensor(tensor.data, dtype=data_type)
            tf_tensor = tf.constant(tensor.data, dtype=data_type)
        elif type(tensor.data).__name__ == "NoneType":
            tf_tensor = tf.placeholder(dtype=data_type, shape=tensor.shape)
        else:
            print("Tensor's 'data' member is of unsupported type " + type(tensor.data))
            exit(1)

        tf_tensors.append(tf_tensor)
