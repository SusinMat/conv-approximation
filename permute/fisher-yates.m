#!/usr/bin/env octave

function [r] = get_rand_int(n)
	seed = 0;
	m = 2147483648;
	a = 1103515245;
	c = 12345;
	seed = mod(a * seed + c, m);
	r = mod(seed, n) + 1;
end

function [array] = random_permutation(n)
	array = [1 : n];
	if n == 1
		return
	end
	for i = n : -1 : 2
		% j = randi([1, i], 1);
		j = get_rand_int(i);
		printf('%d\n', j);
		aux = array(i);
		array(i) = array(j);
		array(j) = aux;
	end
end

rand('seed', 0);
printf('%s\n', mat2str(random_permutation(64)));
