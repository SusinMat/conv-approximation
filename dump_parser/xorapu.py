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
import glob
import magic # package is called 'python-libmagic' in pip, and it also requires libmagic-dev to be installed in the system
import numpy as np
import os
import pickle
import re
import subprocess
import sys
import tempfile


def get_output(command):
    return subprocess.run(command, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).stdout.decode("utf-8")

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
        print("Error: One or more paths were not set previously")
        exit(1)

    mimetype = magic.Magic().from_file(model_path)

    basename = os.path.basename(model_path)
    (filename, file_extension) = os.path.splitext(basename)

    if not (mimetype == "application/octet-stream" and file_extension == ".tflite"):
        print("Model file: " + model_path + " is of unsupported type " + mimetype)
        exit(1)

    if not os.path.isfile(beeswax_directory + "/beeswax"):
        print("An application named 'beeswax' was not found in " + beeswax_directory)
        exit(1)

    image_directory = os.path.abspath(image_directory)
    classes = [os.path.basename(d[0]) for d in os.walk(image_directory)]
    if len(classes) <= 1:
        print(image_directory + " is empty")
        exit(1)
    classes = classes[1:]

    beeswax_path = beeswax_directory + "/beeswax"

    # labels = [l.strip() for l in open("labels.txt", "r").readlines()]

    images = []
    classes = classes[0:101]
    for c in classes:
        class_dir = image_directory + "/" + c + "/"
        class_images = glob.glob(class_dir + "*.bmp")[0:2]
        images += class_images

    temp_list = tempfile.NamedTemporaryFile(mode="w+", prefix="tmp_", suffix=".txt", delete=True)
    temp_list.write("\n".join(images) + "\n")
    temp_list.seek(0)

    beeswax_output = [line[7:] for line in get_output(beeswax_path
            + " -m " + model_path
            + " -l " + "labels.txt"
            + " -f " + temp_list.name).split('\n') if line.startswith("top-5")]

    for line in beeswax_output:
        print(line)
    print(len(beeswax_output))
