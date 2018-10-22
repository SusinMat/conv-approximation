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
    INPUT_TENSOR_INFO = enum.auto()
    TENSOR_DATA = enum.auto()
    OUTPUT_HEADER = enum.auto()
    OUTPUT_TENSOR_INFO = enum.auto()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the output of the Neural Network Transpiler (NNT) weight dump function.")

    parser.add_argument("input_path", help="path to the output file of the NNT weight dump function")

    args = parser.parse_args()

    input_path = args.input_path

    f = open(input_path, 'r')
    dump = [line for line in [line.strip() for line in f.readlines()] if line != "" ]
    f.close()

    s = State.START
    while(True):
        None

    
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

