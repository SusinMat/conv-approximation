#!/usr/bin/env octave

function [array] = random_permutation(n)
	array = [1 : n];
	if n == 1
		return
	end
	for i = n : -1 : 2
		j = randi([1, i], 1);
		aux = array(i);
		array(i) = array(j);
		array(j) = aux;
	end
end

rand('seed', 87958177);
printf('%s\n', mat2str(random_permutation(64)));
