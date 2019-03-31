#!/usr/bin/env octave

global seed;

function [r] = get_rand_int(n)
	global seed;
	m = 100003;
	a = 1103515245;
	c = 12345;
	seed = mod(a * seed + c, m);
	printf('seed after == %d\n', seed);
	r = mod(seed, n);
end

function [array] = random_permutation(n)
	array = [1 : n];
	if n == 1
		return
	end
	for i = n : -1 : 2
		% j = randi([1, i], 1);
		j = get_rand_int(i) + 1;
		printf('j == %d\n', j);
		aux = array(i);
		array(i) = array(j);
		array(j) = aux;
	end
end

% rand('seed', 0);
seed = 2500;
printf('%s\n', mat2str(random_permutation(640)));
exit()
for i = 1 : 500
	get_rand_int(2);
end
