W = randn(256, 5, 5, 96); % create random weights with 96 input channels, 256 output channels and 5x5 filters

% Replace first convolutional layer weights with approximated weights
args.iclust = 48; % number of input clusters
args.oclust = 2; % number of output clusters
args.k = 12;
args.in_s = 55; % don't worry about this, just to print out stats (input spatial extent)
args.out_s = 51; % don't worry about this, just to print out stats (output spatial extent)

args.cluster_type = 'kmeans';

fprintf('||W|| = %f \n', norm(W(:)));

[Wapprox, F, C, XY, perm_in, perm_out, num_weights] = bisubspace_lowrank_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('||W - Wapprox|| / ||W|| = %f (*note* error should be high since weights are random and thus have no structure...\n', L2_err);

