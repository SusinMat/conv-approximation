W = randn(256, 5, 5, 96); --create random weights with 96 input channels, 256 output channels and 5x5 filters

iclust = 2; % number of input clusters
oclust = 2; % number of output clusters
oratio = 0.4; % (0.6 --> 76), (0.5 --> 64)
iratio = 0.4;  % (0.6 --> 78), (0.5 --> 24), (0.4 --> 19)
odegree = floor(size(W, 1) * oratio / oclust);
idegree = floor(size(W, 4) * iratio / iclust);
code = sprintf('in%d_out%d', idegree, odegree);

in_s = 55;
out_s = 51;
fprintf('||W|| = %f \n', norm(W(:)));
[Wapprox, C, Z, F, idx_input, idx_output] = bispace_svd(W, iclust, iratio, oclust, oratio, 0, in_s, out_s);
% 
