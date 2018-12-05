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
        result = tf.concat(input_value, axis=op.options["axis"])
        subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
            subgraph.append(result)
        # print("Concatenation output:", result.shape)

    elif op.name == "Add":
        result = tf.math.add(input_value[0], input_value[1])
        subgraph.append(result)
        activation_function = activation_function_to_tf(op.options["fused_activation_function"])
        if activation_function is not None:
            result = activation_function(result)
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
            result = tf.squeeze(input_value, axis=[0, 1, 2])
            subgraph.append(result)
            result = result * op.inputs[1].data
            subgraph.append(result)
            if op.inputs[2].data is not None:
                result = result + op.inputs[2].data
                subgraph.append(result)
            if activation_function is not None:
                result = activation_function(result)
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

    # parser.add_argument("input_path", help="path to the output file of the input file, which must be a pickle dump generated by dump_parser.py")
    parser.add_argument("--image", "-i", help="path to the input image")
    parser.add_argument("--model", "-m", help="path to the output file of the input file, which must be a pickle dump generated by dump_parser.py")
    parser.add_argument("--xorapu", "-x", action="store_true", help="if set, a xorapu benchmark will be executed to estimate model accuracy")


    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    run_xorapu = args.xorapu

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

    # Count how many tensors indexes are in use
    tensor_indexes = [tensor.index for tensor in tensors]

    # Find layer to apply approximation to
    conv = None
    count = 0
    for op in ops:
        if op.name == "Conv2D":
            count += 1
            if count == 1:
                conv = op
                break
    # pickle.dump(conv, open("layer.pkl", "wb"))
    [Wapprox, Wmono, colors, perm, num_weights] = approximate(op)
    monochrome_tensors = [len(tensor_indexes) + i for i in range(colors.shape[1])]
    tensor_indexes += monochrome_tensors
    rgb_to_mono_tensors = [len(tensor_indexes) + i for i in range(colors.shape[1])]
    tensor_indexes += rgb_to_mono_tensors
    approx_weights = []
    for i in range(len(perm)):
        # TODO: group Wmonos that use the same grayscale as input
        # attribution = perm[i]
        mono_weights = Wmono[i]
        mono_weights = mono_weights.reshape([1, mono_weights.shape[0], mono_weights.shape[1] , 1])
        approx_weights.append(mono_weights)
    new_ops = []
    # perform the conv that outputs the monochrome tensors
    for i in range(colors.shape[1]):
        new_conv = Op(name="Conv2D")
        new_conv.options = fix_dictionary_enum({"padding":"SAME", "stride_h":1, "stride_w":1, "fused_activation_function":"NONE"})
        new_conv.inputs.append(op.inputs[0])
        new_weights_tensor = Tensor(index=rgb_to_mono_tensors[i], type_name=op.inputs[0].type_name)
        new_weights_tensor.data = colors[:,i].reshape(1, 1, 1, colors.shape[0])
        new_weights_tensor.shape = new_weights_tensor.data.shape
        new_conv.inputs.append(new_weights_tensor)
        new_mono_tensor = Tensor(index=monochrome_tensors[i], shape=[1, op.inputs[0].shape[1], op.inputs[0].shape[2], 1], type_name=op.inputs[0].type_name)
        new_conv.outputs = [new_mono_tensor]
        new_ops.append(new_conv)
    mono_weights_tensors = [len(tensor_indexes) + i for i in range(len(perm))]
    tensor_indexes += mono_weights_tensors
    monoconv_indexes = [len(tensor_indexes) + i for i in range(len(perm))]
    monoconv_tensors = []
    tensor_indexes += monoconv_indexes
    mono_bias_indexes = [len(tensor_indexes) + i for i in range(len(perm))]
    tensor_indexes += mono_bias_indexes
    # use the monochrome tensors from the previous step to approximate the output channels of this layer
    for i in range(len(perm)):
        new_conv = Op(name="DepthwiseConv2D")
        new_conv.options = fix_dictionary_enum({"padding":"SAME", "depth_multiplier":1, "stride_h":2, "stride_w":2, "fused_activation_function":"RELU6"})
        new_conv.inputs.append(new_ops[perm[i]].outputs[0]) # the input is the output of the corresponding depthwise convolution
        new_weights_tensor = Tensor(index=mono_weights_tensors[i], shape=mono_weights.shape, type_name=op.inputs[0].type_name)
        new_weights_tensor.data = approx_weights[i]
        new_conv.inputs.append(new_weights_tensor)
        new_bias_tensor = Tensor(index=mono_bias_indexes[i], shape=[1], type_name=op.inputs[0].type_name)
        new_bias_tensor.data = np.asarray([op.inputs[2].data[i]])
        new_conv.inputs.append(new_bias_tensor)
        new_output_tensor = Tensor(index=monoconv_indexes[i], shape=[1, op.outputs[0].shape[1], op.outputs[0].shape[2], 1], type_name=op.inputs[0].type_name)
        new_conv.outputs = [new_output_tensor]
        monoconv_tensors.append(new_output_tensor)
        new_ops.append(new_conv)
    # concatenate the output channels of this layer
    new_concat = Op(name="Concatenation")
    new_concat.options =  fix_dictionary_enum({"axis":3, "fused_activation_function":"NONE"})
    for i in range(len(monoconv_tensors)):
        new_concat.inputs.append(monoconv_tensors[i])
    new_concat.outputs = [op.outputs[0]]
    new_ops.append(new_concat)

    # op.inputs[1].data = Wapprox
    # ops = [ops[0]] + new_ops + ops[1:]
    ops = new_ops + ops[1:]


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
        index_to_tensor[output_indexes[0]] = last_node
        layer_list.append(subgraph)

    if not run_xorapu:
        for layer in layer_list:
            for node in layer:
                print(node)
            print("----------------")

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        # tf.tables_initializer().run(session=sess)

        # tensorflow 1.11
        # save flatbuffer
        converter = tf.contrib.lite.TocoConverter.from_session(sess, [input_placeholder], [last_node])
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
            (top1_accuracy, top5_accuracy) = xorapu.test_model(reconstructed_model.name, None, None, classes_to_test=20 ,images_per_class=10)
            print("Top 1 accuracy: %.02f%%" % (top1_accuracy))
            print("Top 5 accuracy: %.02f%%" % (top5_accuracy))
