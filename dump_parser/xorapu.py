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
import magic # package is called 'python-libmagic' in pip, and it also requires libmagic-dev to be installed in the system
import numpy as np
import os
import pickle
import re
import sys
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the beeswax benchmark and report accuracy.")

    parser.add_argument("--model", "-m", help="Path to the model that will be tested. Always required.")
    parser.add_argument("--images", "-i", help="Path to the directory of the input images. Required only once.")
    parser.add_argument("--beeswax", "-b", help="Path to the directory that contains the beeswax executable. Required only once.")

    args = parser.parse_args()

    model_path = None
    image_directory = None
    beeswax_directory = None

    paths_filename = "xorapu_paths.pkl"

    if os.path.isfile(paths_filename):
        paths = pickle.load(open(paths_filename, "rb"))
        if "image_directory" in paths.keys():
            image_directory = paths["image_directory"]
        if "beeswax_directory" in paths.keys():
            beeswax_directory = paths["beeswax_directory"]
    else:
        pass

    model_path = args.model
    if image_directory == None and args.images is not None:
        image_directory = args.images
    if beeswax_directory == None and args.beeswax is not None:
        beeswax_directory = args.beeswax

    if model_path is None:
        print("Error: path to the model is always required")
        exit(1)

    paths = {"image_directory":image_directory, "beeswax_directory":beeswax_directory}
    with open(paths_filename, "wb") as f:
        pickle.dump(paths, f)


    if None in [image_directory, beeswax_directory]:
        print("Error: One or more paths were not set previously.")
        exit(1)

    mimetype = magic.Magic().from_file(model_path)

    basename = os.path.basename(model_path)
    (filename, file_extension) = os.path.splitext(basename)

    if (mimetype == "application/octet-stream") and (file_extension == ".tflite"):
        pass
    else:
        print("Model file: " + model_path + " is of unsupported type " + mimetype)
        exit(1)
