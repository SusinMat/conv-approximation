This folder contains matlab code to compute the decompositions described in the paper Exploiting Linear Structure Within Convolutional
Networks for Efficient Evaluation (http://arxiv.org/pdf/1404.0736.pdf).

Decomposition functions:
monochromatic_approx.m
bisubspace_lowrank_approx.m
bisubspace_svd_example.lua

Examples of use:
monochromatic_example.m
bisubspace_lowrank_example.m
bisubspace_svd_example.lua

Note that the approximation in the example scripts will be terrible since the weights are initialized to be random (out method relies on the structure inherent in trained convolutional network weights).
