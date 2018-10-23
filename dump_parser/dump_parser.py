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
import enum
import os
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the output of the Neural Network Transpiler (NNT) weight dump function.")

    parser.add_argument("input_path", help="path to the output file of the NNT weight dump function")

    args = parser.parse_args()

    input_path = args.input_path

    f = open(input_path, 'r')
    dump = [line for line in [line.strip() for line in f.readlines()] if line != "" ]
    f.close()

    start_pattern = re.compile(r"Operators:")
    op_pattern = re.compile(r"index: \d+, builtin_op: \w+, inputs: |( \d+)*, outputs: |( \d+)*")
    input_header_pattern = re.compile(r"\+ input tensors:")
    input_info_pattern = re.compile(r"\* \d+:s=\[\d+(, \d+){0,3}\],t=\w+,d=")
    tensor_data_pattern = re.compile(r".+")
    output_header_pattern = re.compile(r"\+ output tensors:")
    output_info_pattern = re.compile(r"\* \d+:s=\[\d+(, \d+){0,3}\],t=\w+")

    state = State.START
    i = 0
    while(i < len(dump)):
        line = dump[i]
        redo = False

        if state == State.START:
            match = start_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = state.OP

        elif state == State.OP:
            match = op_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.INPUT_HEADER

        elif state == State.INPUT_HEADER:
            match = input_header_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.INPUT_INFO

        elif state == State.INPUT_INFO:
            match = input_info_pattern.match(line)
            if match != None: # matched input info
                state = State.TENSOR_DATA
            else:
                state = State.OUTPUT_HEADER
                match = output_header_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                redo = True

        elif state == State.TENSOR_DATA:
            match = tensor_data_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.INPUT_INFO

        elif state == State.OUTPUT_HEADER:
            match = output_header_pattern.match(line)
            if match == None:
                print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                exit(1)
            state = State.OUTPUT_INFO

        elif state == State.OUTPUT_INFO:
            match = output_info_pattern.match(line)
            if match != None: # matched output info
                state = State.OUTPUT_INFO
            else:
                state = State.OP
                match = op_pattern.match(line)
                if match == None:
                    print("Line: '\n" + line[:min(len(line), 80)] + "\n' did not match pattern for state " + str(state))
                    exit(1)
                redo = True

        if not redo:
            i += 1
    
#     pattern = re.compile(r"(?P<name>[^\s]+)\s+" +
#             r"\((?P<model>[^\s]+)\)\s+" +
#             r"is available\.?")
#     for line in check_output:
#         match = pattern.match(line)
#         if match != None:
#             if model == match.group("model"):
#                 name = match.group("name")
#                 if name in exclude_list:
#                     continue
#                 elif len(force_list) > 0 and name not in force_list:
#                     continue
#                 else:
#                     available = name
#                 break
