#!/usr/bin/env python3

import subprocess
import re

if __name__ == "__main__":
    top_file = open("top1.txt", "r")
    file_list = [line.strip() for line in top_file.readlines() if line != ""]
    image_path_pattern = re.compile(r"(?P<image_path>/.*/(?P<class_name>[\w'\-]+)/(?P<image_name>\w+\.bmp))$")
    class_count = {}
    for line in file_list:
        match = image_path_pattern.match(line)
        if match is None:
            continue
        image_path = match["image_path"]
        class_name = match["class_name"]
        image_name = match["image_name"]
        if class_name not in class_count.keys():
            class_count[class_name] = 0
        class_count[class_name] += 1
        if class_count[class_name] > 5:
            continue
        new_path = re.sub("/subset_\w+/", "/subset_top1/", image_path)
        new_dir = re.sub("/\w+\.bmp", "/", new_path)
        print("mkdir -p " + new_dir + " ; cp " + image_path + " " + new_path)
