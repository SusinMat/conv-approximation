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

class Tensor:
    index = -1
    shape = None
    type_name = ""
    data = None
    def __init__(self):
        self.index = -1
        self.shape = None
        self.type_name = ""
        self.data = None

class Op:
    name = ""
    index = -1
    options = {}
    inputs = []
    outputs = []
    def __init__(self):
        self.name = ""
        self.index = -1
        self.options = {}
        self.inputs = []
        self.outputs = []

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
        f = open(input_path, 'r')
        dump = [line for line in [line.strip() for line in f.readlines()] if line != "" ]
        f.close()

        common_info_pattern = r"\* (?P<index>\d+):s=(?P<shape>\[\d+(, \d+){0,3}\]),t=(?P<data_type>\w+)"

        start_pattern = re.compile(r"Operators:")
        op_pattern = re.compile(r"index: (?P<index>\d+), builtin_op: (?P<op_name>\w+), options: (?P<options>{ [\w=\-\.\(\)\[\],\s]*}), inputs:(?P<inputs>( \d+)+| ), outputs:(?P<outputs>( \d+)+| )$")
        input_header_pattern = re.compile(r"\+ input tensors:")
        input_info_pattern = re.compile(common_info_pattern + r",d=$")
        tensor_data_pattern = re.compile(r"(?P<data>.+)$")
        output_header_pattern = re.compile(r"\+ output tensors:")
        output_info_pattern = re.compile(common_info_pattern + r"$")


        state = State.START
        i = 0
        shape = None
        current_op = None
        current_input_tensor = None
        current_output_tensor = None
        while(i < len(dump)):
            line = dump[i]
            redo = False

            if state == State.START:
                match = start_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                state = state.OP

            elif state == State.OP:
                match = op_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
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
                    print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
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
                    if current_input_tensor != None:
                        current_op.inputs.append(current_input_tensor)
                    current_input_tensor = Tensor()
                    current_input_tensor.index = index
                    current_input_tensor.shape = shape
                    current_input_tensor.type_name = data_type
                else:
                    state = State.OUTPUT_HEADER
                    if current_input_tensor != None:
                        current_op.inputs.append(current_input_tensor)
                    current_input_tensor = None
                    match = output_header_pattern.match(line)
                    if match == None:
                        print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
                        exit(1)
                    redo = True

            elif state == State.TENSOR_DATA:
                match = tensor_data_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                data = match.group("data")
                if data.startswith("Empty"):
                    data = None
                else:
                    data = ast.literal_eval(data)
                    data = np.asarray(data)
                current_input_tensor.data = data
                state = State.INPUT_INFO

            elif state == State.OUTPUT_HEADER:
                match = output_header_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
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
                    current_output_tensor = Tensor()
                    current_output_tensor.index = index
                    current_output_tensor.shape = shape
                    current_output_tensor.type_name = data_type
                    current_op.outputs.append(current_output_tensor)
                else:
                    state = State.OP
                    match = op_pattern.match(line)
                    if match == None:
                        print("Line: '\n" + line[:min(len(line), 160)] + "\n' did not match pattern for state " + str(state))
                        exit(1)
                    redo = True

            if not redo:
                i += 1
        if current_op != None:
            ops.append(current_op)
            current_op = None
            current_input_tensor = None
            current_output_tensor = None
        pickle.dump(ops, open(filename + ".pkl", "wb"))
    elif mimetype == "application/octet-stream" and (file_extension in [".pkl", ".pickle"]):
        ops = pickle.load(open(input_path, "rb"))
    else:
        print("Input file: " + input_path + " is of unsupported type " + mimetype)
        exit(0)

    op_names = [op.name for op in ops]
    for i in range(len(ops)):
        op = ops[i]
        op_name = op.name
        op_inputs = [tensor for tensor in op.inputs]
        op_outputs = [tensor for tensor in op.outputs]
        print("Name: %s, inputs: %s, outputs: %s, options: %s" % (op_name, str([tensor.index for tensor in op_inputs]), str([tensor.index for tensor in op_outputs]), op.options))
        for tensor in op_inputs:
            print(" * --> input " + str(tensor.index) + " : s=" + str(tensor.shape) + " <--")
        for tensor in op_outputs:
            print(" * <-- output " + str(tensor.index) + " : s=" + str(tensor.shape) + " -->")
