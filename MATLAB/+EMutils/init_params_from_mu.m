function [alphas, ws, sg] = init_params_from_mu(X, mu)
% Calculate alphas, ws, and sigma for dataset with points X and means mu
% Points will be assigned to closest mean mu(i,:) and used to compute
% parameters for each component.

[N_unlabeled, dim] = size(X);
K = size(mu,3);

% distance from points to positive component means
dist_to_pos_comps = pdist2(X, reshape(mu(1,:,:), [dim, K])');
% distance from points to negative component means
dist_to_neg_comps = pdist2(X, reshape(mu(2,:,:), [dim, K])');
[dist_to_pos, label_poscomp] = min(dist_to_pos_comps,[],2);
[dist_to_neg, label_negcomp] = min(dist_to_neg_comps,[],2);
% assign to either positive or negative
[~, label_posneg] = min([dist_to_pos, dist_to_neg], [], 2);
X_pos = X(label_posneg==1);
X_neg = X(label_posneg==2);
num_pos=size(X_pos,1);
num_neg=size(X_neg,1);
alpha = num_pos/(N_unlabeled);
alphas = [alpha, 1- alpha];

ws = (1/K)*ones(2,K);
sg = zeros(2, dim, dim, K);
for c=1:2
    for k = 1 : K
        sg(c, :, :, k) = eye(dim);
    end
end

for k = 1 : K
    pos_component = label_poscomp(label_posneg==1);
    if sum(pos_component==k) > 0
        t = X_pos(pos_component==k, :) - mu(1,:,k);
        sg(1, :, :, k) = t' * t / sum(pos_component==k);
        ws(1, k) = sum(pos_component==k) / num_pos;
    end
    neg_component = label_negcomp(label_posneg==2);
    if sum(neg_component==k) > 0
        t = X_neg(neg_component==k, :) - mu(2,:,k);
        sg(2, :, :, k) = t' * t / sum(neg_component==k);
        ws(2, k) = sum(neg_component==k) / num_neg;
    end
end
ws(1,:) = ws(1,:)/sum(ws(1,:));
ws(2,:) = ws(2,:)/sum(ws(2,:));
end