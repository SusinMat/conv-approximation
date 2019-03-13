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

def check_model(model_path):
    mimetype = magic.Magic().from_file(model_path)

    basename = os.path.basename(model_path)
    (filename, file_extension) = os.path.splitext(basename)

    if not (mimetype == "application/octet-stream" and file_extension == ".tflite"):
        print("Model file: " + model_path + " is of unsupported type " + mimetype)
        exit(1)

def check_beeswax(beeswax_path):
    if not os.path.isfile(beeswax_path):
        print("An application named 'beeswax' was not found in " + beeswax_directory)
        exit(1)

def read_classes(image_directory, classes_to_test):
    classes = [os.path.basename(d[0]) for d in os.walk(image_directory)]
    if len(classes) <= 1:
        print(image_directory + " is empty")
        exit(1)
    classes = classes[1:]
    classes = classes[0:min(len(classes), classes_to_test)]
    return classes

def read_images(classes, image_directory, images_per_class):
    images = []
    for c in classes:
        class_dir = image_directory + "/" + c + "/"
        class_images = glob.glob(class_dir + "*.bmp")
        class_images = class_images[0:min(len(class_images), images_per_class)]
        images += class_images
    return images

def get_output(command):
    return subprocess.run(command, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).stdout.decode("utf-8")

def create_temp_list(images):
    temp_list = tempfile.NamedTemporaryFile(mode="w+", prefix="tmp_", suffix=".txt", delete=True)
    # temp_list = open("tmp_foo.txt", "w+")
    temp_list.write("\n".join(images) + "\n")
    temp_list.seek(0)
    return temp_list

def run_beeswax(beeswax_path, model_path, images):
    temp_list = create_temp_list(images)
    split_output = get_output(beeswax_path
        + " -m " + model_path
        + " -l " + "labels.txt"
        + " -f " + temp_list.name).split("\n")

    temp_list.close()

    return split_output

def filter_output(output):
    return [line for line in output if line.startswith("top-5:") or line.startswith("image-path:")]

def count_hits(output):
    top1_hits = 0
    top5_hits = 0

    image_path_pattern = re.compile(r"image-path: (?P<image_path>/.*/(?P<class_name>[\w'\-]+)/\w+\.bmp)$")
    # top_pattern = re.compile(r"top-5: (?P<foo>( \| )?(?P<class_name>[\w']+) \(\d+\.\d+%\))*")
    top_pattern = re.compile(r"top-5: ?(( \| )?[\w'\-]+ \(\d+\.\d+%\))*")
    top_classes_pattern = re.compile(r"([\w'\-]+ \(\d+\.\d+%\))")

    next_class = None
    image_path = None

    for line in output:
        if next_class is None:
            match = image_path_pattern.match(line)
            if match is not None:
                next_class = match["class_name"]
                image_path = match["image_path"]
            else:
                print("Class not found in line: " + line)
                exit(1)
        elif next_class is not None:
            match = top_pattern.match(line)
            if match is not None:
                top_classes = [match.split(" ")[0] for match in top_classes_pattern.findall(line)]
                # print("Guessed: " + str(top_classes) + " for class " + next_class.upper())
                if len(top_classes) > 0:
                    if next_class == top_classes[0]:
                        top1_hits += 1
                        top5_hits += 1
                        # print(image_path)
                    elif next_class in top_classes:
                        top5_hits += 1
                else:
                    print("The following line does not contain a top list:\n" + line)
                next_class = None
            else:
                print("Top predictions not found in line: " + line)
                # exit(1)

    return [top1_hits, top5_hits]


def batchify(original_list, batch_size):
    for i in range(0, (len(original_list) // batch_size) + (len(original_list) % batch_size > 0)):
        yield original_list[i * batch_size : min((i + 1) * batch_size, len(original_list))]
    return

def test_model(model_path, image_directory, beeswax_directory, classes_to_test=100, images_per_class=2, batch_size=800, threads=1, seed=None):
    if None in [image_directory, beeswax_directory]:
        paths_filename = "xorapu_paths.pkl"

        if os.path.isfile(paths_filename):
            paths = pickle.load(open(paths_filename, "rb"))
            image_directory = paths["image_directory"]
            beeswax_directory = paths["beeswax_directory"]
        else:
            print("Error: One or more paths were not set previously, %s was not found." % (paths_filename))
            print("Run ./xorapu.py -h for info on how to set them.")
            exit(1)

    check_model(model_path)

    beeswax_path = beeswax_directory + "/beeswax"
    check_beeswax(beeswax_path)

    image_directory = os.path.abspath(image_directory)
    classes = read_classes(image_directory, classes_to_test)

    images = read_images(classes, image_directory, images_per_class)

    # print(len(images))

    split_output = []
    for subset in batchify(images, batch_size):
        split_output += run_beeswax(beeswax_path, model_path, subset)

    # [print(line) for line in split_output]

    filtered_output = filter_output(split_output)

    (top1_hits, top5_hits) = count_hits(filtered_output)

    top1_accuracy = float(top1_hits) / len(images) * 100.0
    top5_accuracy = float(top5_hits) / len(images) * 100.0

    return [top1_accuracy, top5_accuracy]

if __name__ == "__main__":

    # Preparing environment

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

    model_path = args.model
    if args.images is not None:
        image_directory = args.images
    if args.beeswax is not None:
        beeswax_directory = args.beeswax


    paths = {"image_directory":image_directory, "beeswax_directory":beeswax_directory}
    with open(paths_filename, "wb") as f:
        pickle.dump(paths, f)

    if model_path is None:
        print("Path to model not set. Saved dataset and beeswax directories. Now exiting.")
        exit(0)

    # Done preparing environment

    (top1_accuracy, top5_accuracy) = test_model(model_path, image_directory, beeswax_directory)

    # Printing result

    print("Top 1 accuracy: %.02f%%" % (top1_accuracy))
    print("Top 5 accuracy: %.02f%%" % (top5_accuracy))

    # Done printing result
