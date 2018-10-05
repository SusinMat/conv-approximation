function [Wapprox, Wmono, colors, perm, num_weights] = monochromatic_approx(W, args)
% This approximation clusters the first left singular vectors of each of
% the convolution kernels associated with each output feature. Filters in 
% the same cluster share the same inner color component. The reconstructed 
% weight matrix, Wapprox, is returned along with the the color
% transformation matrix, the monochromatic weights and the permutation of
% the weights. These matrices can be used to more efficiently compute the 
% output of the convolution.
%
% args.even : 1 if clusters should be constrained to be equal sizes, 0 
%             otherwise
% args.num_colors : number of clusters (or "colors") to use


    printf('W before permutation: %s\n', mat2str(size(W)));
    W = permute(W, [1, 4, 2, 3]);
    printf('W after permutation: %s\n', mat2str(size(W)));
    % Simple re-parametrization of first layer with monochromatic filters
    for f = 1 : size(W, 1)
	% printf('\n---- Iteration f == %d\n', f);
	if f == 1
	    printf('squeeze(W(f,:,:))--%s\n', mat2str(size(squeeze(W(f,:,:)))));
	end
        [u,s,v] = svd(squeeze(W(f,:,:)),0);
	if f == 1
	    printf('u--%s s--%s v--%s\n', mat2str(size(u)), mat2str(size(s)), mat2str(size(v)));
	end
        C(f, :) = u(:, 1);
	if f == 1
	    printf('u(:, 1)--%s\n', mat2str(size(u(:, 1))));
	end
        S(f, :) = s(1, 1) * v(:, 1);
	if f == 1
	    printf('s(1, 1)--%s * v(:, 1)--%s == %s\n', mat2str(size(s(1, 1))), mat2str(size(v(:, 1))), mat2str(size(s(1, 1) * v(:, 1))));
	end
        chunk = u(:, 1) * s(1, 1) * v(:, 1)'; % ????
	if f == 1
	    printf('chunk == u(:, 1)--%s * s(1, 1)--%s * v(:, 1)t--%s == %s\n', mat2str(size(u(:, 1))), mat2str(size(s(1, 1))), mat2str(size(v(:, 1)')), mat2str(size(u(:, 1) * s(1, 1) * v(:, 1)')));
	end
        approx0(f, :, :, :) = reshape(chunk, 1, size(W, 2), size(W, 3), size(W, 4));
	if f == 1
            printf('approx0(f)--%s\n', mat2str(size(approx0)));
	end
    end

    printf('C--%s\n', mat2str(size(C)));
    printf('S--%s\n', mat2str(size(S)));

    if args.even
        [assignment,colors] = litekmeans(C', args.num_colors);
        colors = colors';
    else
        pkg('load', 'statistics')
        MAXiter = 1000; % Maximum iteration for KMeans Algorithm
        % REPlic = Val(args, 'rep', 100); % Replication for KMeans Algorithm
        % [assignment,colors] = kmeans(C, args.num_colors, 'start', 'sample', 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
        [assignment,colors] = kmeans(C, args.num_colors, 'start', 'sample', 'maxiter', MAXiter, 'EmptyAction', 'singleton');
    end

    printf('assignment--%s colors--%s\n', mat2str(size(assignment)), mat2str(size(colors)));

    Wapprox = zeros(size(W));
    for f=1:size(W,1)
        chunk = (colors(assignment(f),:)') * (S(f,:));
        if f == 1
            printf('colors(assignment(f),:)t--%s * S(f,:)--%s == %s\n', mat2str(size(colors(assignment(f),:)')), mat2str(size(S(f,:))), mat2str(size(chunk)));
	end
        Wapprox(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end 
    
    Wmono = reshape(S,size(W,1),size(W,3),size(W,4));
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
    
    assert(norm(squeeze(Wmono(1, :, :)) * colors(assignment(1), 1) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 2) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 3) - ...
                        (squeeze(Wapprox(1, :, :, 1)) + squeeze(Wapprox(1, :, :, 2)) + squeeze(Wapprox(1, :, :, 3)))) < 1e-7);
                    
    
    [~, perm] = sort(assignment);
    colors = colors';
    
    num_weights = prod(size(colors)) + prod(size(Wmono));
    printf('num_weights == %d\n', num_weights);
end

