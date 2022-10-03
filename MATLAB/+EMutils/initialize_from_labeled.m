function [mu, sg, w_labeled] = initialize_from_labeled(...
    X_labeled_pos, X_labeled_neg, dim, K, ...
    k_mu, initialization)

% Initialize parameters from labeled data X_labeled_pos, X_labeled_neg. 
% Data matrices are Nxdim where N is the number of points
% One of the data matrices can be empty, but not both
%
% Input: data matrices X_labeled_pos, X_labeled_neg [Nxdim] or empty
%        dim: (int) dimension of data
%        K: (int) number of components in data
%        k_mu: (vector [(2*K)xdim]) means from kmeans fit of unlabeled data
%        initialization: (str) either 'labeled_means' (default) or 'order_match'
%
% If initialization is 'labeled_means', EM will be used to estimate means
% independently on labeled (positive and/or negative) components and these
% means will be returned at the initializations for mu. 
% If using 'order_match', the EM means will be used to match to the closest 
% centers from kmeans provided by the k_mu vectors and those will be used 
% for assigning initial values of mu for positive and negative components.
% That is, the K centers closest to the K means from EM on X_labeled_pos
% will be assigned to the positive class and the remaining to the negative
% class

if nargin < 6
    initialization = 'order_match';
end

[mu, sg, w_labeled] = init_empty_parms(dim, K);                             

N_labeled_pos = size(X_labeled_pos, 1); % num positive labeled samples
N_labeled_neg = size(X_labeled_neg, 1); % num negative labeled samples

if N_labeled_pos == 0 && N_labeled_neg == 0
    % return parameters with no labeled information
    return 
end

if N_labeled_pos > 0
    [w_labeled1_pos, mu_labeled_pos, sg_labeled_pos] = EMutils.em(X_labeled_pos, K);
    w_labeled(1,:) = w_labeled1_pos;      
end

if N_labeled_neg > 0
    [w_labeled1_neg, mu_labeled_neg, sg_labeled_neg] = EMutils.em(X_labeled_neg, K);
    w_labeled(2,:) = w_labeled1_neg;
end

switch initialization
    case 'labeled_means' % use labeled means as unlabeled means
        if N_labeled_pos > 0
            mu(1,:,:) = mu_labeled_pos;
            sg(1,:,:,:) = sg_labeled_pos;
            if N_labeled_neg == 0
               % use kmeans centers furthest from labeled positive means
               mu_order = match_vectors(k_mu, mu_labeled_pos'); 
               mu(2,:,:) = k_mu(setdiff(1:2*K,mu_order),:)';
            end
        end
        if N_labeled_neg > 0
            mu(2,:,:) = mu_labeled_neg;
            sg(2,:,:,:) = sg_labeled_neg;
            if N_labeled_neg == 0
                % use kmeans centers furthest from labeled negative means
               mu_order = match_vectors(k_mu, mu_labeled_neg'); 
               mu(1,:,:) = k_mu(setdiff(1:2*K,mu_order),:)';
            end
        end
    case 'order_match'
        if N_labeled_pos > 0 && N_labeled_neg > 0
           mu_order = match_vectors(k_mu, [mu_labeled_pos';mu_labeled_neg']) ;
           mu(1,:,:) = k_mu(mu_order(1:K),:)'; 
           mu(2,:,:) = k_mu(mu_order(K+1:end),:)';
        elseif N_labeled_pos > 0 
            % match kmeans centers to closest labeled means
            mu_order = match_vectors(k_mu, mu_labeled_pos');  % find  kmeans centers closet to labled positive mus
            %closest
            mu(1,:,:) = k_mu(mu_order,:)';  % assign initial positive mus to the closest labeled positive centers          
            mu(2,:,:) = k_mu(setdiff(1:2*K,mu_order),:)';  % assign initial negative mus to the remaining centers          
        elseif N_labeled_neg >0
            % match kmeans centers to closest labeled means
            mu_order = match_vectors(k_mu, mu_labeled_neg');  % find  kmeans centers closet to labled negative mus
            %closest
            mu(2,:,:) = k_mu(mu_order,:)'; 
            mu(1,:,:) = k_mu(setdiff(1:2*K,mu_order),:)'; 
        end  
end
    
end

function [mu, sg, w_labeled] = init_empty_parms(dim, K)

    sg = zeros(2, dim, dim, K);
    for c=1:2
        for k = 1 : K
            sg(c, :, :, k) = eye(dim);
        end
    end
    
    mu = zeros(2, dim, K);    
    w_labeled = zeros(2, K);
end

function assignment = match_vectors(X, Y)
% Find the minimum distance assignment of vectors from Y to vectors of X
% X is a mxd matrix, Y is a nxd matrix
% 
% returns a vector of indices indicating which n rows of X are closest to
% the n rows of Y. The indices are not repeated. i.e. the sum of the
% distances between row Y(i,:) and row X(ASSIGNMENT(i)) for all i is
% minimized for all possible assignments

num_components = size(Y,1);% the number of matches that we need to find
total_comps = size(X,1);
% Pairwise Eucl. dist between all of vectors of X w/ all vectors of Y
dists = pdist2(X, Y);        
choose = nchoosek(1:total_comps,num_components);

min_d = inf;
for c = 1:size(choose, 1)
    permutations = perms(choose(c,:));
    dist_permutations = zeros(size(permutations,1),1);
    for p = 1:size(permutations,1) 
        dist_permutations(p) = sum(diag(dists(permutations(p,:),:)));
        if dist_permutations(p) < min_d
            min_d = dist_permutations(p);
            assignment = permutations(p,:);
        end
    end
end

end
