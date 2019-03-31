#!/usr/bin/env python3

import numpy as np

seed = 2500

def get_rand_int(n):
    global seed
    m = 100003
    a = 1103515245
    c = 12345
    seed = (a * seed + c) % m
    print("seed after == " + str(seed))
    r = seed % n
    return r

def random_permutation(n):
    array = np.arange(n, dtype=np.int)
    if n == 1:
        return array
    for i in range(n - 1, 0, -1):
        # j = np.random.randint(0, i)
        j = get_rand_int(i + 1)
        print("j == " + str(j))
        aux = array[i]
        array[i] = array[j]
        array[j] = aux
    return array

# np.random.seed(0)
print(random_permutation(640) + 1)
exit()

for i in range(500):
    get_rand_int(2)
