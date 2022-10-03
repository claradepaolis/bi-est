function [alpha, negative_params, positive_params, w_labeled] = ...
    PNU_nested_em(X_unlabeled, X_labeled_pos, X_labeled_neg, K, ...
    varargin)

% nested EM for mixture of mixture for (positive-negative-unlabeled) data 
%   X_ul, X_l_pos, X_l_neg: unlabeled, labeled pos, labeled neg data points
%                           size is N*D for N points in D dimensions.
%                           Either labeled set can be empty 
%   K: number of components in each of the mixtures
%   optional argument to display a progress bar

%% Optional progress bar
nVarargs = length(varargin);
if nVarargs > 0
    prog_bar = varargin{1};
else
    prog_bar = false;
end
if prog_bar
    fig = uifigure;
    fig.Position = [500 500 500 150];
    prog = uiprogressdlg(fig,'Title','Progress',...
                             'Message','Optimization');
    drawnow
end
%% setup
% optimization options
max_steps = 2000;
tol = 1e-8; % tolerance level
eps = 1e-300; % small number to replace 0 for log calculations


% data settings
dim = size(X_unlabeled, 2); % dimension  
N_unlabeled = size(X_unlabeled, 1); % num unlabeled samples
N_labeled_pos = size(X_labeled_pos, 1); % num positive labeled samples
N_labeled_neg = size(X_labeled_neg, 1); % num negative labeled samples
N = N_unlabeled + N_labeled_neg + N_labeled_pos;

%% Initialize parameters

% initialize unlabeled means with kmeans
[~, k_mu] = kmeans(X_unlabeled, 2*K, 'Start', 'plus'); % kmeans++ centers

% initialize class assignments from labeled
[mu, ~, w_labeled] = EMutils.initialize_from_labeled(...
    X_labeled_pos, X_labeled_neg, dim, K, k_mu);

[alphas, w, sg] = EMutils.init_params_from_mu(X_unlabeled, mu);

% recondition covariance matrix if necessary
sg = EMutils.check_conditioning(sg);

ll = EMutils.pnu_loglikelihood(X_unlabeled, X_labeled_pos, X_labeled_neg, ...
                               mu, sg, alphas, w, w_labeled, eps);
lls = [ll];

%% Optimization
step = 0;
diff = 2 * tol;

while (step < max_steps) && (diff > tol)
    mu_old =  mu;
    ll_old = ll;
    
    %% E step
    posteriors = zeros(N_unlabeled, 2, K); % row 1 for positive, row 2 for negative, dim 2 for component
    numerator_N = zeros(N_unlabeled, 2,K);
    mult=zeros(2,K);
    inv_sig=zeros(2,K,dim,dim);
    det_sig=zeros(2,K);
    
    % terms from unlabeled points
    for c = 1 : 2 % pos c=1 and neg c=2 components
        for k = 1 : K % components within pos or neg
            t_N = (X_unlabeled - mu(c, :, k));  % N x dim
            sig = reshape(sg(c, :, :, k),[dim,dim]); %select sigma for component c, subcomponent k 
            inv_sig(c,k,:,:) = inv(sig);
            det_sig(c,k) = det(sig);
            mult(c,k)=alphas(c)*w(c,k) / sqrt((2*pi) ^ dim * det_sig(c,k));
            numerator_N(:,c,k) = mult(c,k) * exp(-0.5* sum((t_N * sqrtm(reshape(inv_sig(c,k,:,:),[dim,dim]))).^2, 2));  %
        end
    end
    denom = sum(numerator_N, [2,3]);
    denom(denom==0) = eps;
    for c = 1 : 2
        for k = 1 : K
            posteriors(:,c,k) = numerator_N(:,c,k)./denom;
        end
    end    
    
    %% terms from labeled positives points
    if N_labeled_pos > 0
        posteriors_labeled = zeros(N_labeled_pos, K); 
        numerator_N =zeros(N_labeled_pos,K);
        mult=zeros(1,K);
        inv_sig=zeros(K,dim,dim);
        det_sig=zeros(K,1);
        c = 1; % select positive param index
        for k = 1 : K % subcomponents within pos 
            t_N = (X_labeled_pos - mu(c, :, k));  % N x dim
            sig = reshape(sg(c, :, :, k), [dim,dim]); %select sigma for component c, subcomponent k
            inv_sig(k,:,:) = inv(sig);
            det_sig(k) = det(sig);
            mult(k)= w_labeled(c,k) / sqrt((2*pi) ^ dim * det_sig(k));
            numerator_N(:,k) = mult(k) * exp(-0.5* sum((t_N * sqrtm(reshape(inv_sig(k,:,:),[dim,dim]))).^2, 2));  % sh
        end

        denom = sum(numerator_N, 2);
        denom(denom==0) = eps;
        for k = 1 : K
            posteriors_labeled(:,k) = numerator_N(:,k)./denom;
        end
    end
    
    %% terms from labeled negative points
    if N_labeled_neg > 0
        posteriors_labeled_neg = zeros(N_labeled_neg, K); 
        numerator_N = zeros(N_labeled_neg,K);
        mult = zeros(1,K);
        inv_sig = zeros(K,dim,dim);
        det_sig = zeros(1,K);
        c = 2; % select negative param index
        for k = 1 : K % components within neg 
            t_N = (X_labeled_neg - mu(c, :, k));  % N x dim
            sig = reshape(sg(c, :, :, k), [dim,dim]); %select sigma for component c
            inv_sig(k,:,:) = inv(sig);
            det_sig(k) = det(sig);
            mult(k)= w_labeled(c,k) / sqrt((2*pi) ^ dim * det_sig(k));
            numerator_N(:,k) = mult(k) * exp(-0.5* sum((t_N * sqrtm(reshape(inv_sig(k,:,:),[dim,dim]))).^2, 2));  % sh
        end

        denom = sum(numerator_N, 2);
        denom(denom==0) = eps;
        for k = 1 : K
            posteriors_labeled_neg(:,k) = numerator_N(:,k)./denom;
        end
    end
  
    if N_labeled_pos > 0
        full_posteriors_labeled=zeros(N_labeled_pos,2,K);
        full_posteriors_labeled(:,1,:) = reshape(posteriors_labeled,N_labeled_pos,1,K);
        posteriors_labeled = full_posteriors_labeled;
    end
    if N_labeled_neg > 0
        full_posteriors_labeled_neg=zeros(N_labeled_neg,2,K);
        full_posteriors_labeled_neg(:,2,:) = reshape(posteriors_labeled_neg,N_labeled_neg,1,K);
        posteriors_labeled_neg = full_posteriors_labeled_neg;
    end 
%% M step
     % given label and component posteriors, update parameters: alphas, w, mu, and sg
     % alphas and ws are estimated using only unlabeled data
     %% Update parameters alpha and w 
     % get sorted sums for more accurate results
     p = zeros(2,K);
     p_labeled = zeros(2,K);
     for c = 1 : 2
        for k = 1 : K
            p(c,k) = sum(sort(posteriors(:,c,k),'ascend')); % sum over instances
            if N_labeled_pos > 0 && c==1
                p_labeled(c,k) = sum(sort(posteriors_labeled(:,c,k),'ascend')); 
            end
            if N_labeled_neg > 0 && c==2
                p_labeled(c,k) = sum(sort(posteriors_labeled_neg(:,c,k),'ascend')); 
            end
        end
     end
     for c = 1 : 2
        comp_posterior_sum = sum(p(c,:)); % sum over subcomponents
        alphas(c) = comp_posterior_sum/N_unlabeled;
        for k = 1 : K
            w(c,k) = sum(p(c,k))/comp_posterior_sum;
            if N_labeled_pos > 0 && c==1
                lcomp_posterior_sum = sum(p_labeled(c,:));
                w_labeled(c, k) = p_labeled(c,k)/lcomp_posterior_sum;
            end
            if N_labeled_neg > 0 && c==2
                lcomp_posterior_sum = sum(p_labeled(c,:));
                w_labeled(c, k) = p_labeled(c,k)/lcomp_posterior_sum;
            end
        end
     end
    
    % Correct mixing proportions
    % prevent later taking log(w_i) if w_i==0
    if sum(w==0)>0
        w(w==0) = eps;
        w(1,:) = w(1,:)/sum(w(1,:));
        w(2,:) = w(2,:)/sum(w(2,:));
    end    
    if N_labeled_pos > 0
        w_labeled(1, w_labeled(1, :)==0)=eps;
        w_labeled(1,:) = w_labeled(1,:)/sum(w_labeled(1,:));
    end
    if N_labeled_neg > 0
        w_labeled(2, w_labeled(2, :)==0)=eps;
        w_labeled(2,:) = w_labeled(2,:)/sum(w_labeled(2,:));
    end
    
    %% Update parameters mu & sigma
    denom = reshape(sum(posteriors, 1), 2, K);
    if N_labeled_pos > 0 || N_labeled_neg > 0 
        denom = denom + p_labeled;
    end  
    mu = zeros(2, dim, K);
    sg = zeros(2, dim, dim, K);
    
    for k = 1 : K
        for c = 1:2
            pX = posteriors(:, c, k) .* X_unlabeled;
            mu(c, :, k) = sum(sort(pX));
            xmu_unlabeled = X_unlabeled - mu_old(c, :, k);
            pxmu = sqrt(posteriors(:, c, k)).*xmu_unlabeled;
            sg(c, :, :, k) = reshape(pxmu' * pxmu, 1, dim, dim);
        end    
        if N_labeled_pos > 0 
            c=1;  % select positive param index
            xmu_labeled = (X_labeled_pos - mu_old(c, :, k));
            pX = posteriors_labeled(:, c, k) .* X_labeled_pos;
            mu(c, :, k) = mu(c, :, k) + (sum(sort(pX)));
            pxmu = sqrt(posteriors_labeled(:, c, k)).*xmu_labeled;
            sg(c, :, :, k) = sg(c, :, :, k) + (reshape(pxmu' * pxmu, 1, dim, dim));
        end            
        if N_labeled_neg > 0 
            c=2; % select negative param index
            xmu_labeled_neg = (X_labeled_neg - mu_old(c, :, k));
            pX = posteriors_labeled_neg(:, c, k) .* X_labeled_neg;
            mu(c, :, k) = mu(c, :, k) + (sum(sort(pX)));
            pxmu = sqrt(posteriors_labeled_neg(:, c, k)).*xmu_labeled_neg;
            sg(c, :, :, k) = sg(c, :, :, k) + (reshape(pxmu' * pxmu, 1, dim, dim));
        end            
    end

    denom(denom==0) = eps;
    for c = 1:2
        for k = 1:K
            mu(c,:,k) = mu(c,:,k) ./ denom(c, k);
            sg(c,:,:,k) = sg(c,:,:,k) ./ denom(c, k);
        end
    end
    
    % recondition covariance matrix if necessary
    sg = EMutils.check_conditioning(sg);

    %% Compute loglikelihood
    ll = EMutils.pnu_loglikelihood(X_unlabeled, X_labeled_pos, X_labeled_neg, ...
                                   mu, sg, alphas, w, w_labeled, eps);

    lls = [lls,ll];
    
    %% Termination conditions
    diff = abs(ll-ll_old)/abs(ll_old);
    step = step + 1;
    if prog_bar
        prog.Value = step/max_steps;
    end
end

if prog_bar
    prog.Value = 1;
    if step < max_steps
        prog.Message = 'Converged';
    else
        prog.Message = 'Max steps reached';
    end
    pause(0.2);
    close(prog);
    close(fig);
end

alpha = alphas(1);
positive_params = struct('mu', squeeze(mu(1,:,:)), ...
    'sig', squeeze(sg(1,:,:, :)), 'w',w(1,:));
negative_params = struct('mu',squeeze(mu(2,:,:)), ...
    'sig',squeeze(sg(2,:,:, :)), 'w',w(2,:));
end