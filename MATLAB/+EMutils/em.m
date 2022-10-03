function [w, mu, sg] = em(X, K, max_step)

% function [w, mu, sg] = em (X, M)
%
% Inputs:  X = n-by-d matrix of observations
%          K = the number of clusters
%
% Outputs: w = M-by-1 vector, fractions of data points in each cluster
%          mu = d-by-M matrix of means for each cluster
%          sg = covariance matrices for each cluster

N=size(X, 1);
eps = 0.000001; % tolerance level
if nargin < 3
    max_step = 250; % maximum number of iterations
end

% seed the random number generator so that the experiment is repeatable
%randn('state', 12345)

dim = size(X, 2); % dimensionality of the sample

% initialize w, mu, and sg
w = ones(1, K) / K;

d = 0.01 * mean(sum((X - repmat(mean(X, 1), size(X, 1), 1)) .^ 2, 2));

[~, mu] = kmeans(X, K); mu = mu'; % use K-means to find initial clusters

for k = 1 : K
    sg(:, :, k) = diag(ones(dim, 1));
end

step = 0;
diff = 2 * eps; % just to ensure it is greater than eps


while diff > eps && step < max_step
    
    w_old = w;
    mu_old = mu;
    sg_old = sg;
    
    %% E step
    posteriors = zeros(N, K); % row 1 for positive, row 2 for negative, dim 2 for component
    numerator_N = zeros(N, K);
    mult=zeros(1,K);
    inv_sig=zeros(K,dim,dim);
    det_sig=zeros(1,K);
    
    % terms from unlabeled points
    for k = 1 : K % components within pos or neg
        t_N = X - mu(:, k)';  % N x dim
        sig = reshape(sg(:, :, k),[dim,dim]); %select sigma for component k 
        inv_sig(k,:,:) = inv(sig);
        det_sig(k) = det(sig);
        mult(k)=w(k) / sqrt((2*pi) ^ dim * det_sig(k));
        numerator_N(:,k) = mult(k) * exp(-0.5* sum((t_N * sqrtm(reshape(inv_sig(k,:,:),[dim,dim]))).^2, 2));  %
    end
    denom = sum(numerator_N, [2,3]);
    denom(denom==0) = eps;
    for k = 1 : K
        posteriors(:,k) = numerator_N(:,k)./denom;
    end  
    
%% M step
     % given label and component posteriors, update parameters: alphas, w, mu, and sg
     
     %% Update parameters alpha and w 
     % get sorted sums for more accurate results
     p = zeros(1,K);
    for k = 1 : K
        p(k) = sum(sort(posteriors(:,k),'ascend')); % sum over instances
    end
    comp_posterior_sum = sum(p); % sum over subcomponents
    for k = 1 : K
        w(k) = sum(p(k))/comp_posterior_sum;
    end
    
    % Correct mixing proportions
    % prevent later taking log(w_i) if w_i==0
    if sum(w==0)>0
        w(w==0) = eps;
        w = w/sum(w);
    end    
    
    %% Update parameters mu & sigma
    denom = reshape(sum(posteriors, 1), 1, K);
    mu = zeros(dim, K);
    sg = zeros(dim, dim, K);
    
    for k = 1 : K
        pX = posteriors(:, k) .* X;
        mu(:, k) = sum(sort(pX));
        xmu = X - mu_old(:, k)';
        pxmu = sqrt(posteriors(:, k)).*xmu;
        sg(:, :, k) = reshape(pxmu' * pxmu, dim, dim);              
    end

    denom(denom==0) = eps;
    for k = 1:K
        mu(:,k) = mu(:,k) ./ denom(k);
        sg(:,:,k) = sg(:,:,k) ./ denom(k);
    end
    
    % recondition covariance matrix if necessary
    sg = EMutils.check_conditioning(sg);

    %% Termination conditions
%     diff = sum(abs(mu - mu_old),'all')./abs(sum(mu,'all'));
    diff = sum(abs(w - w_old)) + sum(sum(abs(mu - mu_old))) + sum(sum(sum(abs(sg - sg_old))));
    step = step + 1;
end

end
