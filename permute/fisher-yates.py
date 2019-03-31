#!/usr/bin/env python3

import numpy as np

def random_permutation(n):
    array = np.arange(n, dtype=np.int)
    if n == 1:
        return array
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i)
        aux = array[i]
        array[i] = array[j]
        array[j] = aux
    return array

np.random.seed(0)
print(random_permutation(64))
