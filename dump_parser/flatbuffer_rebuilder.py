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

use_slim_depthwise = 0
use_layers_conv    = 1

def read_tensor_from_image_file(file_name, input_height=224, input_width=224, input_mean=-127, input_std=127):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def fix_dictionary_enum(old_dict):
    new_dict = {}
    enum_pattern = re.compile(r"\w+\(\d+\)")
    for key in old_dict.keys():
        value = old_dict[key]
        if type(value) == str:
            if re.match(enum_pattern, value):
                value = re.sub(r"(?P<string>\w+)\(\d+\)", r"\g<string>", value)
            value = value.lower()
        new_dict[key] = value

    return new_dict

def type_name_to_tf(type_name):
    if type_name == "FLOAT32":
        return tf.float32
    elif type_name == "INT32":
        return tf.int32
    elif type_name == "UINT8":
        return tf.uint8
    else:
        print("Unsupported type: " + type_name)
        exit(1)

def activation_function_to_tf(func_name):
    func_name = func_name.upper()
    if func_name == "NONE":
        return None
    elif func_name == "RELU6":
        return tf.nn.relu6
    elif func_name == "RELU":
        return tf.nn.relu
    elif func_name == "TANH":
        return tf.nn.tanh
    else:
        print("Unsupported type: " + func_name)
        exit(1)

def tensor_has_no_data(tensor):
    if tensor.data is None:
        return True
    else:
        return False

def op_to_tf(op, input_value):
    result = None
    subgraph = []
    if op.name == "Conv2D":
        weight_as_tensor = tf.constant_initializer(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
        bias_as_tensor = tf.constant_initializer(op.inputs[2].data, dtype=type_name_to_tf(op.inputs[2].type_name))
        weight_data = op.inputs[1].data
        weight_as_array = weight_data.transpose(1, 2, 3, 0)
        if use_layers_conv:
            result = tf.layers.conv2d(input_value,
                    op.inputs[1].shape[0],
                    op.inputs[1].shape[1:3],
                    kernel_initializer=weight_as_tensor,
                    bias_initializer=bias_as_tensor,
                    use_bias=False,
                    strides=[op.options["stride_h"], op.options["stride_w"]],
                    padding=op.options["padding"],
                    activation=activation_function_to_tf(op.options["fused_activation_function"])
                   )
            subgraph.append(result)
        else:
            result = tf.nn.conv2d(input_value,
                    weight_as_array,
                    [1, op.options["stride_h"], op.options["stride_w"], 1],
                    padding=op.options["padding"].upper()
                   )
            subgraph.append(result)
            result = tf.nn.bias_add(result, op.inputs[2].data)
            subgraph.append(result)
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)

    elif op.name == "DepthwiseConv2D":
        weight_as_tensor = tf.constant_initializer(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
        bias_as_tensor = tf.constant_initializer(op.inputs[2].data, dtype=type_name_to_tf(op.inputs[2].type_name))
        weight_data = op.inputs[1].data
        weight_as_array = weight_data.transpose(1, 2, 3, 0)
        if use_slim_depthwise:
            result = tf.contrib.slim.separable_convolution2d(input_value,
                    None, # Makes the separable_convolution2d depthwise (as used @mobilenet)
                    op.inputs[1].shape[1:3],
                    weights_initializer=weight_as_tensor,
                    biases_initializer=bias_as_tensor,
                    depth_multiplier=op.options["depth_multiplier"],
                    stride=[op.options["stride_h"], op.options["stride_w"]],
                    padding=op.options["padding"].upper(),
                    activation_fn=activation_function_to_tf(op.options["fused_activation_function"])
                   )
            subgraph.append(result)
        else:
            result = tf.nn.depthwise_conv2d(input_value,
                    weight_as_array,
                    [1, op.options["stride_h"], op.options["stride_w"], 1],
                    padding=op.options["padding"].upper()
                   )
            subgraph.append(result)
            result = tf.nn.bias_add(result, op.inputs[2].data)
            subgraph.append(result)
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)

    elif op.name == "Pool2D":
        result = tf.contrib.slim.avg_pool2d(input_value,
                                            [op.options["filter_height"], op.options["filter_width"]],
                                            stride=[op.options["stride_h"], op.options["stride_w"]],
                                            padding=op.options["padding"].upper())
        subgraph.append(result)

    elif op.name == "Squeeze":
        result = tf.squeeze(input_value,
                            axis=op.options["squeeze_dims"]
                           )
        subgraph.append(result)

    elif op.name == "Reshape":
        result = tf.reshape(input_value, op.options["new_shape"])

    elif op.name == "Softmax":
        if abs(op.options["beta"] - 1.0) > 0.0001:
            beta = tf.constant(op.options["beta"])
            input_value = beta * input_value
        result = tf.nn.softmax(input_value)
        subgraph.append(result)

    else:
        print("Unsupported operation: " + op.name)
        exit(1)

    if result == None:
        print("Error: result unassigned. Op name: " + op.name)
        exit(1)
    return subgraph


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

    for i in range(len(ops)):
        ops[i].options = fix_dictionary_enum(ops[i].options)

    op_names = [op.name for op in ops]
    tensors = [item for sublist in [op.inputs + op.outputs for op in ops] for item in sublist]

    # for i in range(len(ops)):
    #     op = ops[i]
    #     op_name = op.name
    #     op_inputs = [tensor for tensor in op.inputs]
    #     op_outputs = [tensor for tensor in op.outputs]
    #     print("Name: %s, inputs: %s, outputs: %s, options: %s" % (op_name, str([tensor.index for tensor in op_inputs]), str([tensor.index for tensor in op_outputs]), op.options))
    #     for tensor in op_inputs:
    #         print(" * --> input " + str(tensor.index) + " : s=" + str(tensor.shape) + " <--")
    #     for tensor in op_outputs:
    #         print(" * <-- output " + str(tensor.index) + " : s=" + str(tensor.shape) + " -->")

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
        elif tensor_has_no_data(tensor):
            tf_tensor = tf.placeholder(dtype=data_type, shape=tensor.shape)
        else:
            print("Tensor's 'data' member is of unsupported type " + type(tensor.data))
            exit(1)
        tf_tensors.append(tf_tensor)

    input_image = read_tensor_from_image_file("grace_hopper.bmp")
    input_image = read_tensor_from_image_file("scabbard.bmp")
    input_image = read_tensor_from_image_file("dingo.bmp")
    input_image = read_tensor_from_image_file("tench.bmp")
    input_image = read_tensor_from_image_file("llama.bmp")
    input_image = read_tensor_from_image_file("zebra.bmp")
    image = input_image.reshape([1, 224, 224, 3])

    op = ops[0]
    input_placeholder = None

    for tensor in ops[0].inputs:
        if tensor_has_no_data(tensor):
            input_placeholder = tf_tensors[tensor.index]
    if input_placeholder == None:
        print("Error: could not find input tensor.")
        exit(1)

    graph = [input_placeholder]
    for op in ops:
        subgraph = op_to_tf(op, graph[-1])
        graph += subgraph
        for tensor in subgraph:
            print(tensor)
        print("----------------")

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    # tf.tables_initializer().run(session=sess)
    out_tensor = sess.run(graph[-1], {input_placeholder : image})

    sorted_out_tensor = np.flipud(np.sort(out_tensor[0]))
    indexes = np.argsort(-out_tensor[0])
    print("Top 5:")
    for i in range(5):
        print("%03d : %05.2f%%" % (indexes[i], sorted_out_tensor[i] * 100))
    # print(sess.run(evaluated_tensors[2], {input_placeholder : image}))
