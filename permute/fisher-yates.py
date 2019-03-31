#!/usr/bin/env python3

import numpy as np

def get_rand_int(n):
    seed = 0
    m = 2147483648
    a = 1103515245
    c = 12345
    seed = (a * seed + c) % m
    r = seed % n + 1
    return r

def random_permutation(n):
    array = np.arange(n, dtype=np.int)
    if n == 1:
        return array
    for i in range(n - 1, 0, -1):
        # j = np.random.randint(0, i)
        j = get_rand_int(i + 1) - 1
        print(j)
        aux = array[i]
        array[i] = array[j]
        array[j] = aux
    return array

np.random.seed(0)
print(random_permutation(64))
