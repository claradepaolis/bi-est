function logl  = pnu_loglikelihood(X_unlabeled, X_labeled_pos, X_labeled_neg, ...
                                   mu, sg, alphas, w, w_labeled, eps)
                               
% Compute loglikelihood for all data comprised of unlabeled, 
% labeled positives, and labeled negatives.
%
% mu, sg(sigma), alphas, w, and w_labeled are the mixture parameters
% eps: epsilon value for correcting any log(0) calculations

N_unlabeled = size(X_unlabeled, 1); % num unlabeled samples
N_labeled_pos = size(X_labeled_pos, 1); % num positive labeled samples
N_labeled_neg = size(X_labeled_neg, 1); % num negative labeled samples

ll = compute_ll(X_unlabeled, mu, sg, [alphas(1)*w(1,:);alphas(2)*w(2,:)], eps);
if N_labeled_pos > 0
    ll_labeled_pos = compute_ll(X_labeled_pos, ...
        mu(1,:), sg(1,:,:), w_labeled(1,:), eps);
end
if N_labeled_neg > 0 
    ll_labeled_neg = compute_ll(X_labeled_neg, ...
        mu(2,:), sg(2,:,:), w_labeled(2,:), eps);
end
if N_labeled_pos > 0
    ll = ll + ll_labeled_pos;
end
if N_labeled_neg > 0
    ll = ll + ll_labeled_neg;
end
logl = ll/(N_unlabeled + N_labeled_pos + N_labeled_neg);
end


function ll = compute_ll(X, mu, sigma, w, eps)
% mu should be dim*number of components
% sigma should be dim*dim*number of components
    [N,dim] = size(X);
    K = length(w);
    if length(size(mu))== 3 
        C = size(mu, 1);
    else
        C = 1;
    end
    
    ll = zeros(N,C);
    twopidim = (2*pi) ^ dim;
    for c = 1:C
        m = reshape(mu(c, :, :), [dim, K]);
        sg = reshape(sigma(c, :, :, :), [dim,dim,K]);
        l = zeros(N, K);
        for k = 1:K
            sig_ij = sg(:, :, k); %select sigma for component c
            detsig = sqrt(twopidim * det(sig_ij));
            xdiff = (X - m(:, k)');
            squareterm = sum((xdiff / sig_ij) .* xdiff,2);% equivalend to (xdiff * inv(sig_ij)) .* xdiff;
            N_ck = w(c,k) * exp(-0.5 * squareterm)/detsig;
            N_ck(N_ck==0) = eps;
            l(:,k) = N_ck;
        end
        ll(:,c) = sum(l,2);
    end
    ll = sum(log(sum(ll,2)));
end