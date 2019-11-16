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
from protopyte import approximate
import re
import sys
import tensorflow as tf
import tempfile, subprocess
import xorapu

from tf_op import Tensor, Op, remove_successive_duplicates

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress message about lack of AVX2 and FMA support in the TensorFlow binary

pooling_types = ["AVG", "MAX"]

use_layers_conv            = 0
use_slim_depthwise         = 0
use_slim_pool              = 1
use_layers_fully_connected = 0
use_custom_implementation  = 0

def new_tensor_indexes(count, tensor_indexes):
    new_indexes = [len(tensor_indexes) + i for i in range(count)]
    tensor_indexes += new_indexes
    return (new_indexes, tensor_indexes)

def cluster_mapping_to_index_mapping(idx):
        new_idx = []
        c_count = max(idx) + 1
        for i in range(c_count):
            new_idx.append(np.where(idx == i)[0].tolist())
        return new_idx

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
            value = value.upper()
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

def activation_function_to_int(func_name):
    if func_name == "NONE":
        return 0
    elif func_name == "RELU":
        return 1
    elif func_name == "RELU6":
        return 3
    else:
        print("Unsupported type: " + func_name)
        exit(1)

def padding_type_to_int(padding):
    if padding == "UNKNOWN":
        return 0
    elif padding == "SAME":
        return 1
    elif padding == "VALID":
        return 2
    else:
        print("Unsupported padding: %s" % (padding))
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
        if len(op.inputs) >= 3:
            bias_as_tensor = tf.constant_initializer(op.inputs[2].data, dtype=type_name_to_tf(op.inputs[2].type_name))
        weight_data = op.inputs[1].data
        weight_as_array = weight_data.transpose(1, 2, 3, 0)
        if use_layers_conv:
            if len(op.inputs) >= 3:
                result = tf.layers.conv2d(input_value,
                        op.inputs[1].shape[0],
                        op.inputs[1].shape[1:3],
                        kernel_initializer=weight_as_tensor,
                        bias_initializer=bias_as_tensor,
                        strides=[op.options["stride_h"], op.options["stride_w"]],
                        padding=op.options["padding"],
                        activation=activation_function_to_tf(op.options["fused_activation_function"])
                       )
            else:
                result = tf.layers.conv2d(input_value,
                        op.inputs[1].shape[0],
                        op.inputs[1].shape[1:3],
                        kernel_initializer=weight_as_tensor,
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
                    padding=op.options["padding"]
                   )
            subgraph.append(result)
            if len(op.inputs) >= 3:
                result = tf.nn.bias_add(result, op.inputs[2].data)
                subgraph.append(result)
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)
        # print("Conv2D output:", result.shape)

    elif op.name == "DepthwiseConv2D":
        weight_as_tensor = tf.constant_initializer(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
        if len(op.inputs) >= 3:
            bias_as_tensor = tf.constant_initializer(op.inputs[2].data, dtype=type_name_to_tf(op.inputs[2].type_name))
        weight_data = op.inputs[1].data
        weight_as_array = weight_data.transpose(1, 2, 3, 0)
        if use_slim_depthwise:
            if len(op.inputs) >= 3:
                result = tf.contrib.slim.separable_convolution2d(input_value,
                        None, # Makes the separable_convolution2d depthwise (as used @mobilenet)
                        op.inputs[1].shape[1:3],
                        weights_initializer=weight_as_tensor,
                        biases_initializer=bias_as_tensor,
                        depth_multiplier=op.options["depth_multiplier"],
                        stride=[op.options["stride_h"], op.options["stride_w"]],
                        padding=op.options["padding"],
                        activation_fn=activation_function_to_tf(op.options["fused_activation_function"])
                       )
            else:
                result = tf.contrib.slim.separable_convolution2d(input_value,
                        None, # Makes the separable_convolution2d depthwise (as used @mobilenet)
                        op.inputs[1].shape[1:3],
                        weights_initializer=weight_as_tensor,
                        use_bias=False,
                        depth_multiplier=op.options["depth_multiplier"],
                        stride=[op.options["stride_h"], op.options["stride_w"]],
                        padding=op.options["padding"],
                        activation_fn=activation_function_to_tf(op.options["fused_activation_function"])
                       )
            subgraph.append(result)
        else:
            result = tf.nn.depthwise_conv2d(input_value,
                    weight_as_array,
                    [1, op.options["stride_h"], op.options["stride_w"], 1],
                    padding=op.options["padding"]
                   )
            subgraph.append(result)
            if len(op.inputs) >= 3:
                result = tf.nn.bias_add(result, op.inputs[2].data)
                subgraph.append(result)
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)
        # print("DepthwiseConv2D output:", result.shape)

    elif op.name == "Pool2D":
        if use_slim_pool:
            if op.options["pooling_type"] == "AVG":
                result = tf.contrib.slim.avg_pool2d(input_value,
                        [op.options["filter_height"], op.options["filter_width"]],
                        stride=[op.options["stride_h"], op.options["stride_w"]],
                        padding=op.options["padding"]
                       )
            elif op.options["pooling_type"] == "MAX":
                result = tf.contrib.slim.max_pool2d(input_value,
                        [op.options["filter_height"], op.options["filter_width"]],
                        stride=[op.options["stride_h"], op.options["stride_w"]],
                        padding=op.options["padding"]
                       )
            else:
                print("Unsupported pooling type: " + op.options["pooling_type"])
                exit(1)
            subgraph.append(result)
        else:
            result = tf.nn.pool(input_value,
                    window_shape=[op.options["filter_height"], op.options["filter_width"]],
                    pooling_type=op.options["pooling_type"],
                    padding=op.options["padding"],
                    strides=[op.options["stride_h"], op.options["stride_w"]]
                   )
            subgraph.append(result)
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)

    elif op.name == "Squeeze":
        result = tf.squeeze(input_value,
                            axis=op.options["squeeze_dims"]
                           )
        subgraph.append(result)

    elif op.name == "Reshape":
        result = tf.reshape(input_value, op.options["new_shape"])
        subgraph.append(result)

    elif op.name == "Softmax":
        if abs(op.options["beta"] - 1.0) > 0.0001:
            beta = tf.constant(op.options["beta"])
            input_value = beta * input_value
        result = tf.nn.softmax(input_value)
        subgraph.append(result)

    elif op.name == "Concatenation":
        # print("Concatenation input:", input_value)
        result = tf.concat(input_value, axis=op.options["axis"])
        subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
            subgraph.append(result)
        # print("Concatenation output:", result.shape)

    elif op.name == "Split":
        num_or_size_splits_tensor = tf.constant_initializer(int(input_value.shape[op.options["axis"]]), dtype=tf.int32)
        result = tf.split(input_value, int(input_value.shape[op.options["axis"]]), axis=op.options["axis"])
        subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
            subgraph.append(result)
        print("Split output(s):")
        for tensor in result:
            print(tensor.shape)

    elif op.name == "Split2":
        num_or_size_splits_tensor = tf.constant_initializer(int(op.options["split_size"]), dtype=tf.int32)
        result = tf.split(input_value, int(op.options["split_size"]), axis=op.options["axis"])
        print(result)
        subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
            subgraph.append(result)
        print("Split output(s):")
        for tensor in result:
            print(tensor.shape)

    elif op.name == "Add":
        if isinstance(input_value, list):
            result = tf.math.add(input_value[0], input_value[1])
            subgraph.append(result)
        else:
            result = tf.math.add(input_value, op.inputs[1].data)
            subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
            subgraph.append(result)

    elif op.name == "Mul":
        result = tf.multiply(input_value, op.inputs[1].data)
        subgraph.append(result)

    elif op.name == "FullyConnected":
        if use_layers_fully_connected:
            weight_as_tensor = tf.constant_initializer(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
            bias_as_tensor = tf.constant_initializer(op.inputs[2].data, dtype=type_name_to_tf(op.inputs[2].type_name))
            result = tf.layers.dense(input_value,
                    op.inputs[1].shape[0],
                    activation=activation_function_to_tf(op.options["fused_activation_function"]),
                    kernel_initializer=weight_as_tensor,
                    use_bias=True,
                    bias_initializer=bias_as_tensor,
                   )
            subgraph.append(result)
        else:
            activation_function = activation_function_to_tf(op.options["fused_activation_function"])
            result = tf.squeeze(input_value, axis=[0, 1])
            subgraph.append(result)
            result = tf.matmul(result, op.inputs[1].data.transpose())
            subgraph.append(result)
            if op.inputs[2].data is not None:
                result = result + op.inputs[2].data
                subgraph.append(result)
            if activation_function is not None:
                result = activation_function(result)
                subgraph.append(result)

    elif op.name == "Pad" or op.name == "MirrorPad":
        mode = "ERROR"
        if op.name == "Pad":
            mode = "CONSTANT"
        paddings = tf.constant(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
        result = tf.pad(input_value, paddings, mode)
        subgraph.append(result)

    elif op.name == "Mean":
        axis = tf.constant(op.inputs[1].data, dtype=type_name_to_tf(op.inputs[1].type_name))
        keep_dims = bool(op.options["keep_dims"])
        result = tf.reduce_mean(input_value, axis, keepdims=keep_dims)
        subgraph.append(result)

    elif op.name == "Dragunov":
        (stride_h, stride_w, padding, fused_activation_function) = (op.options["stride_h"], op.options["stride_w"], op.options["padding"], activation_function_to_int(op.options["fused_activation_function"]))
        result = tf.user_ops.dragunov(input_value, op.inputs[1].data, op.inputs[2].data, op.inputs[3].data, op.inputs[4].data, op.inputs[5].data, op.inputs[6].data, stride_h=stride_h, stride_w=stride_w, padding=padding, fused_activation_function=fused_activation_function)
        subgraph.append(result)
        print("Dragunov output:", result.shape)

    else:
        print("Unsupported operation: " + op.name)
        exit(1)

    if result == None:
        print("Error: result unassigned. Op name: " + op.name)
        exit(1)
    return subgraph


def accuracy_approximation(ops, tensors, op_name, index, strategy="bisubspace_svd"):
    # Count how many tensors indexes are in use
    tensor_indexes = [tensor.index for tensor in tensors]

    # Find layer to apply approximation to
    conv = None
    count = 0
    for i in range(len(ops)):
        op = ops[i]
        if op.name == op_name:
            if count == index:
                break
            count += 1
    else:
        print("Error: you requested op '" + op_name + "' of index " + str(index) + ", but there are only " + str(count) + " operations of this type.")
        exit()

    # pickle.dump(op, open("layer.pkl", "wb"))
    # exit()
    ### Approximation starts here ###
    # [Wapprox, Wmono, colors, perm, num_weights] = approximate(op, strategy="bisubspace_svd")
    Wapprox = approximate(op, strategy=strategy, num_colors=8 ,even=False)[0]
    ### Wapprox = np.random.uniform(size=op.inputs[1].shape, low=np.min(op.inputs[1].data), high=np.max(op.inputs[1].data))
    op.inputs[1].data = Wapprox
    # num_colors = colors.shape[1]
    # num_expansions = op.outputs[0].shape[3] // num_colors

    ops[i] = op

    return (ops, tensors)

def computation_approximation(ops, tensors, op_name, index, strategy="bisubspace_svd", offset=0):
    # Count how many tensors indexes are in use
    tensor_indexes = [tensor.index for tensor in tensors]

    new_offset = offset

    if op_name == "Conv2D":
        eqv_op_name = ["Conv2D", "Dragunov"]
    else:
        eqv_op_name = [op_name]

    # Find layer to apply approximation to
    target_op_i = None
    count = 0
    for i in range(len(ops)):
        op = ops[i]
        if op.name in eqv_op_name:
            if count == index + offset:
                target_op_i = i
                break
            count += 1
    if strategy == "monochromatic":
        [Wapprox, Wmono, colors, perm, num_weights] = approximate(op, strategy=strategy)
        new_ops = []
        num_colors = colors.shape[1]
        # tensor that holds the weights to calculate the monochrome tensor
        transformation_weights_index = [len(tensor_indexes) + i for i in range(1)]
        tensor_indexes += transformation_weights_index
        # tensor that holds the monochrome tensors, before they are split
        monotensor_index = [len(tensor_indexes) + i for i in range(1)]
        tensor_indexes += monotensor_index

        new_conv = Op(name="Conv2D")
        new_conv.options = fix_dictionary_enum({"padding":"SAME", "stride_h":1, "stride_w":1, "fused_activation_function":"NONE"})
        new_conv.inputs.append(op.inputs[0])
        new_weights_tensor = Tensor(index=transformation_weights_index[0], type_name=op.inputs[0].type_name)
        new_weights_tensor.data = colors.transpose().reshape(num_colors, 1, 1, colors.shape[0])
        new_weights_tensor.shape = new_weights_tensor.data.shape
        new_conv.inputs.append(new_weights_tensor)
        new_mono_tensor = Tensor(index=monotensor_index[0], shape=[1, op.inputs[0].shape[1], op.inputs[0].shape[2], num_colors], type_name=op.inputs[0].type_name)
        new_conv.outputs = [new_mono_tensor]
        new_ops.append(new_conv)

        expand_weights_tensor = [len(tensor_indexes) + i for i in range(1)]
        tensor_indexes += expand_weights_tensor
        expand_biases_tensor = [len(tensor_indexes) + i for i in range(1)]
        tensor_indexes += expand_biases_tensor

        new_conv = Op(name="DepthwiseConv2D")
        depth_multiplier = op.outputs[0].shape[3] // num_colors
        new_conv.options = fix_dictionary_enum({"padding":"SAME", "depth_multiplier":depth_multiplier, "stride_h":2, "stride_w":2, "fused_activation_function":"RELU6"})
        new_conv.inputs.append(new_mono_tensor)
        # weights
        new_weights_tensor = Tensor(index=expand_weights_tensor[0], type_name=op.inputs[0].type_name)
        new_weights_tensor.data = np.random.randn(depth_multiplier, op.inputs[1].shape[1], op.inputs[1].shape[2], num_colors).astype("float32")
        new_weights_tensor.shape = new_weights_tensor.data.shape
        new_conv.inputs.append(new_weights_tensor)
        # biases
        new_biases_tensor = Tensor(index=expand_biases_tensor[0], type_name=op.inputs[0].type_name)
        new_biases_tensor.data = np.random.randn(num_colors * depth_multiplier).astype("float32")
        new_biases_tensor.shape = new_biases_tensor.data.shape
        new_conv.inputs.append(new_biases_tensor)
        # output
        new_conv.outputs.append(op.outputs[0])
        new_ops.append(new_conv)
        ops = ops[0 : target_op_i] + new_ops + ops[target_op_i + 1 : len(ops)]

        tensors = [item for sublist in [op.inputs + op.outputs for op in ops] for item in sublist]
        tensors.sort(key=lambda item: item.index)
        tensors = remove_successive_duplicates(tensors)
    elif strategy == "bisubspace_svd":
        print("Original op options:" + str(op.options))
        [Wapprox, C, Z, F, idx_input, idx_output] = approximate(op, strategy=strategy)
        new_ops = []
        if use_custom_implementation:
            # TODO: once the Dragunov standalone op is working, delete the old method of adding new ops
            out_size = op.outputs[0].shape[1]
            new_dragunov = Op(name="Dragunov")
            padding = padding_type_to_int(op.options["padding"])
            stride_h = op.options["stride_h"]
            stride_w = op.options["stride_w"]
            fused_activation_function = op.options["fused_activation_function"]
            new_dragunov.options = fix_dictionary_enum({"stride_h":stride_h, "stride_w":stride_w, "padding":padding, "fused_activation_function":fused_activation_function})
            new_dragunov.outputs.append(op.outputs[0])
            new_dragunov.inputs.append(op.inputs[0])
            (C_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
            (Z_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
            (F_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
            C_tensor = Tensor(index=C_index[0], shape=C.shape, type_name=op.inputs[0].type_name)
            Z_tensor = Tensor(index=Z_index[0], shape=Z.shape, type_name=op.inputs[0].type_name)
            F_tensor = Tensor(index=F_index[0], shape=F.shape, type_name=op.inputs[0].type_name)
            C_tensor.data = C
            Z_tensor.data = Z
            F_tensor.data = F
            new_dragunov.inputs.append(C_tensor)
            new_dragunov.inputs.append(Z_tensor)
            new_dragunov.inputs.append(F_tensor)

            iidx = np.asarray(cluster_mapping_to_index_mapping(idx_input))
            oidx = np.asarray(cluster_mapping_to_index_mapping(idx_output))
            (iidx_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
            (oidx_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
            iidx_tensor = Tensor(index=iidx_index[0], shape=iidx.shape, type_name="INT32")
            oidx_tensor = Tensor(index=oidx_index[0], shape=oidx.shape, type_name="INT32")
            iidx_tensor.data = iidx
            oidx_tensor.data = oidx
            new_dragunov.inputs.append(iidx_tensor)
            new_dragunov.inputs.append(oidx_tensor)

            new_dragunov.inputs.append(op.inputs[2]) # the bias tensor

            new_ops.append(new_dragunov)
            # tf.user_ops.dragunov(input, filter_c, filter_z, filter_f, iidx, oidx, ...)
            ops = ops[0 : target_op_i] + new_ops + ops[target_op_i + 1 : len(ops)]
            new_tensors = [item for sublist in [op.inputs + op.outputs for op in new_ops] for item in sublist]
            tensors = [item for sublist in [op.inputs + op.outputs for op in ops] for item in sublist]
            tensors.sort(key=lambda item: item.index)
            tensors = remove_successive_duplicates(tensors)
            new_offset += 0
            return (ops, tensors, new_offset)
        # end of TODO

        iidx = cluster_mapping_to_index_mapping(idx_input)
        oidx = cluster_mapping_to_index_mapping(idx_output)
        ic_count = max(idx_input) + 1
        oc_count = max(idx_output) + 1

        # alternative: fill weight tensor with zeros

        new_C_weights = np.zeros([C.shape[2] * C.shape[3] * C.shape[1], 1, 1, Wapprox.shape[3]])
        for o in range(oc_count):
            for i in range(ic_count):
                C_ = C[:, :, i, o]
                filter_range_beginning = o * C.shape[2] * C.shape[1] + i * C.shape[1]
                f_range = range(filter_range_beginning, filter_range_beginning + C.shape[1])
                ch_range = iidx[i]
                for f in range(C_.shape[1]):
                    for ch in range(C_.shape[0]):
                        new_C_weights[f_range[f] , 0, 0, ch_range[ch]] = C_[ch, f]

        C_conv = Op(name="Conv2D")
        C_conv.options = fix_dictionary_enum({"padding":"SAME", "stride_h":1, "stride_w":1, "fused_activation_function":"NONE"})
        C_conv.inputs.append(op.inputs[0])
        (C_weights_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
        C_weights_tensor = Tensor(index=C_weights_index[0], shape=new_C_weights.shape, type_name=op.inputs[0].type_name)
        C_weights_tensor.data = new_C_weights
        C_conv.inputs.append(C_weights_tensor)

        (C_presplit_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
        C_presplit_tensor = Tensor(index=C_presplit_index[0], shape=[1, op.inputs[0].shape[1], op.inputs[0].shape[2], new_C_weights.shape[0]], type_name=op.inputs[0].type_name)
        C_conv.outputs = [C_presplit_tensor]
        new_ops.append(C_conv)

        # split
        (C_split_indexes, tensor_indexes) = new_tensor_indexes(ic_count * oc_count, tensor_indexes)
        C_split = Op(name="Split2")
        C_split.options["axis"] = 3
        C_split.options["split_size"] = len(C_split_indexes)
        C_split.options["fused_activation_function"] = "NONE"
        C_split.inputs = [C_presplit_tensor]
        C_split_tensors = []
        C_split_shape = [C_presplit_tensor.shape[0], C_presplit_tensor.shape[1], C_presplit_tensor.shape[2], C_split.options["split_size"]]
        for c in C_split_indexes:
            C_split_tensor = Tensor(index=c, shape=C_split_shape)
            C_split_tensors.append(C_split_tensor)
            C_split.outputs.append(C_split_tensors[-1])
        new_ops.append(C_split)

        # 3x3 conv
        (Z_weights_indexes, tensor_indexes) = new_tensor_indexes(len(C_split_tensors), tensor_indexes)
        (Z_outputs_indexes, tensor_indexes) = new_tensor_indexes(len(C_split_tensors), tensor_indexes)
        Z_output_tensors = [None] * len(C_split_tensors)

        for o in range(oc_count):
            for i in range(ic_count):
                split_tensor_relative_index = o * ic_count + i
                split_tensor_index = C_split_indexes[split_tensor_relative_index]
                new_Z_weights = Z[:, :, :, :, i, o]
                Z_conv = Op(name="Conv2D")
                Z_conv.options = fix_dictionary_enum({"padding":op.options["padding"], "stride_h":op.options["stride_h"], "stride_w":op.options["stride_w"], "fused_activation_function":"NONE"})
                Z_conv.inputs.append(C_split_tensors[split_tensor_relative_index])

                Z_weights_tensor = Tensor(index=Z_weights_indexes[split_tensor_relative_index], shape=new_Z_weights.shape, type_name=op.inputs[0].type_name)
                Z_weights_tensor.data = new_Z_weights
                Z_conv.inputs.append(Z_weights_tensor)


                Z_output_tensor = Tensor(index=Z_outputs_indexes[split_tensor_relative_index], shape=[1, op.outputs[0].shape[1], op.outputs[0].shape[2], Z_weights_tensor.shape[0]], type_name=op.inputs[0].type_name)
                Z_output_tensors[split_tensor_relative_index] = Z_output_tensor
                print(Z_output_tensor.shape)
                Z_conv.outputs = [Z_output_tensor]

                new_ops.append(Z_conv)

        # concat
        F_concat = Op("Concatenation")
        F_concat.options["fused_activation_function"] = "NONE"
        F_concat.options["axis"] = 3
        (F_concat_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
        F_concat_tensor = Tensor(index=F_concat_index[0], shape=[Z_output_tensor.shape[0], Z_output_tensor.shape[1], Z_output_tensor.shape[2], Z_output_tensor.shape[3] * ic_count * oc_count], type_name=op.inputs[0].type_name)
        F_concat.inputs = Z_output_tensors
        F_concat.outputs = [F_concat_tensor]
        new_ops.append(F_concat)

        # 1x1 conv
        new_F_weights = np.zeros([op.outputs[0].shape[3], 1, 1, F_concat_tensor.shape[3]])
        print(new_F_weights.shape)

        for o in range(oc_count):
            for i in range(ic_count):
                # truth is, for Inceptionv4, we only need a few of the output channels, the rest can be zero
                F_ = F[:, :, i, o]
                print(F_.shape)
                channel_range_beginning = o * F.shape[2] * F.shape[1] + i * F.shape[1]
                ch_range = range(channel_range_beginning, channel_range_beginning + F.shape[1])
                f_range = oidx[o]
                print(f_range)
                print(ch_range)
                for f in range(F_.shape[0]):
                    for ch in range(F_.shape[1]):
                        new_F_weights[f_range[f], 0, 0, ch_range[ch]] = F_[f, ch]

        F_conv = Op(name="Conv2D")
        F_conv.options = fix_dictionary_enum({"padding":"SAME", "stride_h":1, "stride_w":1, "fused_activation_function":op.options["fused_activation_function"]})
        F_conv.inputs.append(F_concat_tensor)
        (F_weights_index, tensor_indexes) = new_tensor_indexes(1, tensor_indexes)
        F_weights_tensor = Tensor(index=F_weights_index[0], shape=new_F_weights.shape, type_name=op.inputs[0].type_name)
        F_weights_tensor.data = new_F_weights
        F_conv.inputs.append(F_weights_tensor)
        F_conv.inputs.append(op.inputs[2])

        F_conv.outputs = op.outputs

        new_ops.append(F_conv)

        # exit()

        # split 64ch into 2 32ch
        # perform 4 1x1 conv from 32ch to 12ch
        # perform 4 3x3 conv from 12ch to 19ch
        # unclear part starts here: experimentation (or careful study) is required
        # perform 4 1x1 conv from 19ch to 48ch
        # accumulate (sum) 2 of the previous 48ch outputs to produce a single 48ch for each pair
        # splice (combine) the 2 48ch outputs into a single 96ch output

        ops = ops[0 : target_op_i] + new_ops + ops[target_op_i + 1 : len(ops)]

        # TODO: code from here until the end of this block is just a copy of the previous case

        new_tensors = [item for sublist in [op.inputs + op.outputs for op in new_ops] for item in sublist]

        tensors = [item for sublist in [op.inputs + op.outputs for op in ops] for item in sublist]
        tensors.sort(key=lambda item: item.index)
        tensors = remove_successive_duplicates(tensors)

        new_offset += oc_count * ic_count + 1
    else:
        print("Error: invalid strategy '%s'." % strategy)
        exit(1)

    return (ops, tensors, new_offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild a CNN (Convolutional Neural Network) given a .pkl generated by dump_parser.py")

    # parser.add_argument("input_path", help="path to the output file of the input file, which must be a pickle dump generated by dump_parser.py")
    parser.add_argument("--image", "-i", help="path to the input image")
    parser.add_argument("--model", "-m", help="path to the output file of the input file, which must be a pickle dump generated by dump_parser.py")
    parser.add_argument("--xorapu", "-x", action="store_true", help="if set, a xorapu benchmark will be executed to estimate model accuracy")
    parser.add_argument("--computational", "-c", action="store_true", help="if set, the reconstructed model will serve for estimating computational cost rather than accuracy")
    parser.add_argument("--disable_approximation", "-d", action="store_true", help="if set, no approximation will be applied to the model")
    parser.add_argument("--use_custom_implementation", "-u", action="store_true", help="if set, custom implementations for approximate ops will be used")
    parser.add_argument("--layers", "-l", help="the list of conv layers, comma-separated, to apply the approximation to")

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    run_xorapu = args.xorapu
    layers = args.layers
    approximate_accuracy = not args.computational
    enable_approximation = not args.disable_approximation
    use_custom_implementation = args.use_custom_implementation
    if use_custom_implementation:
        approximate_accuracy = False

    if layers is not None:
        layers = [int(conv_index.strip()) for conv_index in layers.split(",")]

    input_mode = (image_path != None)

    mimetype = magic.Magic().from_file(model_path)

    # if mimetype == "text/plain":
    #     model_path = dump_parser.text_to_pickle(model_path, filename)
    #     mimetype = magic.Magic().from_file(model_path)

    basename = os.path.basename(model_path)
    (filename, file_extension) = os.path.splitext(basename)

    ops = []

    if mimetype == "application/octet-stream" and (file_extension in [".pkl", ".pickle"]):
        ops = pickle.load(open(model_path, "rb"))
    else:
        print("Model file: " + model_path + " is of unsupported type " + mimetype)
        exit(1)

    for i in range(len(ops)):
        ops[i].options = fix_dictionary_enum(ops[i].options)

    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    tensors = [item for sublist in [op.inputs + op.outputs for op in ops] for item in sublist]
    tensors.sort(key=lambda item: item.index)
    tensors = remove_successive_duplicates(tensors)

    candidate_convs = []
    conv_count = 0
    for i in range(0, len(ops)):
        if ops[i].name == "Conv2D":
            if ops[i].inputs[1].shape[1] > 1 and ops[i].inputs[1].shape[2] > 1:
                candidate_convs.append(conv_count)
            conv_count += 1
    print(", ".join([str(i) for i in candidate_convs]))
    print("%d out of %d Conv2D ops (%.2f%%)" % (len(candidate_convs), conv_count, 100.0 * len(candidate_convs) / conv_count))

    if enable_approximation:
        target_op_indexes = []
        target_op_indexes = [0] # monochromatic
        # target_op_indexes = [3, 6] # squeezenet
        # target_op_indexes = [28, 56, 70, 84] # inception_v2_resnet
        # target_op_indexes = [16, 29, 71, 75] # inception_v3
        # target_op_indexes = [16, 22, 34, 37] # inception_v4
        # target_op_indexes = [1, 2, 4, 7, 9, 10, 14, 16, 17, 21, 23, 24, 26, 28, 29, 71, 75, 81, 90] # inception_v3 all
        # target_op_indexes = [1, 2, 3, 5, 9, 10, 13, 15, 16, 20, 22, 23, 27, 29, 30, 34, 36, 37, 39, 41, 42, 114, 118] # inception_v4 all
        # target_op_indexes = [3, 6, 9, 13, 16, 19, 22, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 96, 99, 102] # resnet_v2 all
        # target_op_indexes = [1, 2, 4, 7, 9, 10, 14, 16, 17, 21, 23, 24, 28, 30, 31, 35, 37, 38, 42, 44, 45, 49, 51, 52, 56, 58, 59, 63, 65, 66, 70, 72, 73, 77, 79, 80, 82, 84, 85, 187] # inception_resnet_v2 all
        # target_op_indexes = [3, 6, 9, 12, 15, 18, 21, 24] # squeezenet all
        if layers is not None:
            target_op_indexes = layers
        print("Target conv layers : [%s]" % (", ". join([str(conv_index) for conv_index in target_op_indexes])))

        if approximate_accuracy:
            for i in target_op_indexes:
                (ops, tensors) = accuracy_approximation(ops, tensors, "Conv2D", i, strategy="monochromatic")
        else:
            new_offset = 0
            for i in target_op_indexes:
                (ops, tensors, new_offset) = computation_approximation(ops, tensors, "Conv2D", i, strategy="bisubspace_svd", offset=new_offset)

    # Determine from input tensors which one is the network's input
    empty_indexes_set = set([tensor.index for tensor in tensors if tensor.data is None])
    output_tensors = set([tensor.index for sublist in [op.outputs for op in ops] for tensor in sublist])
    input_tensors = empty_indexes_set - output_tensors
    if len(input_tensors) != 1:
        print("Error: Set of input tensors has", len(input_tensors), "elements.")
        exit(1)
    network_input_index = list(input_tensors)[0]
    network_input_tensor = None

    # Determine which ops depend directly on the input tensor (i.e. they are on the first layer)
    input_ops = []
    for op in ops:
        for tensor in op.inputs:
            if tensor.index == network_input_index:
                input_ops.append(op)
                if network_input_tensor is None:
                    network_input_tensor = tensor

    if len(input_ops) < 0:
        print("Error: input op not found.")
        exit(1)

    # Allocate placeholder for the network's input
    if network_input_tensor.type_name == "FLOAT32":
        data_type = tf.float32
    elif network_input_tensor.type_name == "INT32":
        data_type = tf.int32
    elif network_input_tensor.type_name == "UINT8":
        data_type = tf.uint8
    else:
        print("Unsupported data type " + network_input_tensor.type_name)
        exit(1)
    input_placeholder = tf.placeholder(dtype=data_type, shape=network_input_tensor.shape)


    # Map tensor index to where its value is/will be held
    index_to_tensor = {network_input_index : input_placeholder}
    layer_list = []

    for op in ops:
        input_indexes = [tensor.index for tensor in op.inputs if tensor.data is None]
        if len(input_indexes) > 1: # More than one input tensor: input values will be passed as a list
            op_input = []
            for input_index in input_indexes:
                op_input.append(index_to_tensor[input_index])
        else:
            # Single input tensor: input value will be passed as tensor
            op_input = index_to_tensor[input_indexes[0]]
        # Convert layer to chain of ops
        subgraph = op_to_tf(op, op_input)
        # Remember last op in the chain
        last_node = subgraph[-1]
        output_indexes = [tensor.index for tensor in op.outputs]
        # Update map
        if type(last_node) is not list:
            index_to_tensor[output_indexes[0]] = last_node
        else:
            for i in range(len(op.outputs)):
                index_to_tensor[output_indexes[i]] = last_node[i]
        layer_list.append(subgraph)

    if not run_xorapu:
        for layer in layer_list:
            for node in layer:
                print(node)
            print("----------------")

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        # tf.tables_initializer().run(session=sess)

        # Save flatbuffer
        # tensorflow 1.11:
        # converter = tf.contrib.lite.TocoConverter
        # tensorflow 1.12:
        converter = tf.contrib.lite.TFLiteConverter
        converter = converter.from_session(sess, [input_placeholder], [last_node])
        converter.allow_custom_ops = True

        quantize = False
        if quantize:
            converter.post_training_quantize = True
            input_name = "Placeholder"
            quantized_input_stats = {input_name : (127.5, 127.5)}
            converter.quantized_input_stats = quantized_input_stats
            converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8

        tflite_model = converter.convert()
        reconstructed_model = open("reconstructed_" + filename + ".tflite", "wb")
        reconstructed_model.write(tflite_model)

        if input_mode:
            input_image = read_tensor_from_image_file(image_path)
            image = input_image.reshape([1, 224, 224, 3])
            out_tensor = sess.run(last_node, {input_placeholder : image})
            sorted_out_tensor = np.flipud(np.sort(out_tensor[0]))
            indexes = np.argsort(-out_tensor[0])
            print("Top 5:")
            for i in range(5):
                print("%03d : %05.2f%% (%s)" % (indexes[i], sorted_out_tensor[i] * 100, labels[indexes[i]]))
            # print(sess.run(evaluated_tensors[2], {input_placeholder : image}))
            # print("----------------")
            # for tensor in graph:
            #     print(tensor)
            #     out_tensor = sess.run(tensor, {input_placeholder : image})
            #     print(out_tensor.flatten().tolist()[0])
        if run_xorapu:
            (top1_accuracy, top5_accuracy) = xorapu.test_model(reconstructed_model.name, None, None, classes_to_test=400, images_per_class=10)
            print("Top 1 accuracy: %.02f%%" % (top1_accuracy))
            print("Top 5 accuracy: %.02f%%" % (top5_accuracy))
            if top1_accuracy > 92.00 and False:
                random_int = np.random.randint(100000000000)
                random_model_file = open("good%.02f_%u.tflite" % (top1_accuracy, random_int), "wb")
                random_model_file.write(tflite_model)
                random_model_file.close()
