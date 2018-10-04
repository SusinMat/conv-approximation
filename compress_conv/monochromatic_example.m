W = abs(randn(96, 7, 7, 3));  % make random weights with 3 input color channels, 96 output channelsand 7x7 filters

% Compute approximation
fprintf('||W|| = %f \n', norm(W(:)));
args.num_colors = 6;
args.even = false; % default: true
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), args);
L2_err = norm(W(:) - Wapprox(:)) / norm(W(:));
fprintf('||W - Wapprox|| / ||W|| = %f (*note* error should be high since weights are random and thus have no structure...\n', L2_err);

