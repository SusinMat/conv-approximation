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

from tf_op import *

class State(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name
    START = enum.auto()
    OP = enum.auto()
    INPUT_HEADER = enum.auto()
    INPUT_INFO = enum.auto()
    TENSOR_DATA = enum.auto()
    OUTPUT_HEADER = enum.auto()
    OUTPUT_INFO = enum.auto()

def text_to_pickle(input_path, filename):
    f = open(input_path, 'r')
    dump = [line for line in [line.strip() for line in f.readlines()] if line != ""]
    f.close()

    # match portion of the 'tensor line' that is common to input and output tensors
    common_info_pattern = r"\* (?P<index>\d+):s=(?P<shape>\[\d+(, \d+){0,3}\]),t=(?P<data_type>\w+)(,q=(?P<quantization_parameters>\[[\w\s\-\.,]+\]))?"

    # match the first line
    start_pattern = re.compile(r"Operators:")
    # match the 'op line'
    op_pattern = re.compile(r"builtin_op: (?P<op_name>\w+)\((?P<index>\d+)\), options: (?P<options>{ [\w:\"\-\.\(\)\[\],\s]*}), inputs:(?P<inputs>( \d+)+| ), outputs:(?P<outputs>( \d+)+| )$")
    # match the line that introduces the list of input tensors
    input_header_pattern = re.compile(r"\+ input tensors:")
    # match input tensors, minus the data (weights), which end with 'd=', introducing the data (weights)
    input_info_pattern = re.compile(common_info_pattern + r",d=$")
    # match any non-empty line, for use with the data (weight) line
    tensor_data_pattern = re.compile(r"(?P<data>.+)$")
    # match the line that introduces the list of output tensors
    output_header_pattern = re.compile(r"\+ output tensors:")
    # match input tensors, which don't feature data (weights)
    output_info_pattern = re.compile(common_info_pattern + r"$")

    state = State.START
    i = 0
    ops = []
    shape = None
    current_op = None
    current_input_tensor = None
    current_output_tensor = None
    max_error_line_length = 160
    while(i < len(dump)):
        line = dump[i]
        redo = False # if True, loop restarts without incrementing i, emulating Perl's 'redo' statement

        if state == State.START:
            match = start_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = state.OP

        elif state == State.OP:
            match = op_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.INPUT_HEADER
            index = match.group("index")
            index = int(index)
            op_name = match.group("op_name")
            if op_name.endswith("Options"):
                op_name = re.sub(r"Options$", "", op_name)
            else:
                print("Operator name " + op_name + "does not end with 'Options'")
                exit(1)
            options = match.group("options").strip()
            # special case: values that such as NONE(0) and SAME(1) that come from enums must be treated as strings
            options = re.sub(r":(?P<value>(\w+\(\d+\)))", r":'\g<value>'", options)
            # keys are strings
            options = re.sub(r"(?P<key>\w+):", r"'\g<key>':", options)
            options = ast.literal_eval(options)
            inputs = match.group("inputs").strip().split(" ")
            outputs = match.group("outputs").strip().split(" ")
            if current_op != None:
                ops.append(current_op)
            current_op = Op()
            current_op.name = op_name
            current_op.index = index
            current_op.options = options

        elif state == State.INPUT_HEADER:
            match = input_header_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.INPUT_INFO

        elif state == State.INPUT_INFO:
            match = input_info_pattern.match(line)
            if match != None: # matched input info
                state = State.TENSOR_DATA
                index = match.group("index")
                index = int(index)
                shape = match.group("shape")
                shape = ast.literal_eval(shape)
                data_type = match.group("data_type")
                quantization_parameters = match.group("quantization_parameters")
                if quantization_parameters is not None:
                    quantization_parameters = quantization_parameters_list_from_str(quantization_parameters)
                    quantization_parameters = Quantization(min=quantization_parameters[0], max=quantization_parameters[1], scale=quantization_parameters[2], zero_point=quantization_parameters[3])
                else:
                    quantization_parameters = None
                if current_input_tensor is not None:
                    current_op.inputs.append(current_input_tensor)
                current_input_tensor = Tensor()
                current_input_tensor.index = index
                current_input_tensor.shape = shape
                current_input_tensor.type_name = data_type
                current_input_tensor.quantization = quantization_parameters
            else:
                state = State.OUTPUT_HEADER
                if current_input_tensor != None:
                    current_op.inputs.append(current_input_tensor)
                current_input_tensor = None
                match = output_header_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                redo = True

        elif state == State.TENSOR_DATA:
            match = tensor_data_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            data = match.group("data")
            if data.startswith("Empty"):
                data = None
            else:
                data = ast.literal_eval(data)
                data = np.asarray(data, dtype=getattr(np, current_input_tensor.type_name.lower()))
            if current_input_tensor.quantization is not None:
                if data is not None:
                    data = dequantize(data, current_input_tensor.quantization.scale, current_input_tensor.quantization.zero_point)
                current_input_tensor.type_name = "FLOAT32"
            current_input_tensor.data = data
            state = State.INPUT_INFO

        elif state == State.OUTPUT_HEADER:
            match = output_header_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.OUTPUT_INFO

        elif state == State.OUTPUT_INFO:
            match = output_info_pattern.match(line)
            if match != None: # matched output info
                state = State.OUTPUT_INFO
                index = match.group("index")
                index = int(index)
                shape = match.group("shape")
                shape = ast.literal_eval(shape)
                data_type = match.group("data_type")
                quantization_parameters = match.group("quantization_parameters")
                if quantization_parameters != None:
                    quantization_parameters = quantization_parameters_list_from_str(quantization_parameters)
                    quantization_parameters = Quantization(min=quantization_parameters[0], max=quantization_parameters[1], scale=quantization_parameters[2], zero_point=quantization_parameters[3])
                else:
                    quantization_parameters = None
                current_output_tensor = Tensor()
                current_output_tensor.index = index
                current_output_tensor.shape = shape
                current_output_tensor.type_name = data_type
                current_output_tensor.type_name = "FLOAT32"
                current_output_tensor.quantization = quantization_parameters
                current_op.outputs.append(current_output_tensor)
            else:
                state = State.OP
                match = op_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), max_error_line_length)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                redo = True

        if not redo:
            i += 1
    if current_op != None:
        ops.append(current_op)
        current_op = None
        current_input_tensor = None
        current_output_tensor = None

    model_filename = filename + ".pkl"
    pickle.dump(ops, open(model_filename, "wb"))
    return model_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the output of the Neural Network Transpiler (NNT) weight dump function.")

    parser.add_argument("input_path", help="path to the output file of the input file, which can be either: * an NNT weight dump function in text form, or * a pickle dump of a previous execution of this program")

    args = parser.parse_args()

    input_path = args.input_path

    mimetype = magic.Magic().from_file(input_path)

    basename = os.path.basename(input_path)
    (filename, file_extension) = os.path.splitext(basename)

    ops = []

    if mimetype == "text/plain":
        model_filename = text_to_pickle(input_path, filename)
        ops = pickle.load(open(model_filename, "rb"))
    elif mimetype == "application/octet-stream" and (file_extension in [".pkl", ".pickle"]):
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
