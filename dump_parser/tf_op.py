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

import ast
import numpy as np
import re

class Quantization:
    min = float("nan")
    max = float("nan")
    scale = float("nan")
    zero_point = None
    def __init__(self, min=float("nan"), max=float("nan"), scale=float("nan"), zero_point=None):
        self.min = min
        self.max = max
        self.scale = scale
        self.zero_point = zero_point

class Tensor:
    index = -1
    shape = None
    type_name = ""
    quantization = None
    data = None
    def __init__(self, index=-1, shape=None, type_name="", quantization=None, data=None):
        self.index =index
        self.shape = shape
        self.type_name = type_name
        self.quantization = quantization
        self.data = data

class Op:
    name = ""
    index = -1
    options = {}
    inputs = []
    outputs = []
    def __init__(self, name=""):
        self.name = name
        self.index = -1
        self.options = {}
        self.inputs = []
        self.outputs = []

def remove_successive_duplicates(old_list):
    new_list = []
    last_index = None

    for item in old_list:
        if item.index != last_index:
            last_index = item.index
            new_list.append(item)

    return new_list

def quantization_parameters_list_from_str(str_list):
    # new_list = re.sub(r"([A-Za-z]+)", r"'\1'", str_list)
    new_list = re.sub(r"((\+|\-)?[A-Za-z]+)", r"'\1'", str_list)
    new_list = ast.literal_eval(new_list)
    new_list = [float(item) for item in new_list]
    return new_list

def dequantize(data, scale, zero_point):
    data_array = data.astype(dtype=np.float32)
    data_array = scale * (data_array - zero_point)
    return data_array
