

%% 

% load data
datafile = 'dataset.mat';
load(datafile);
unlabeled = data.unlabeled;
labeled_pos = data.labeled_pos;
labeled_neg = data.labeled_neg;

% set optimization options
num_components = 4;
progress_bar = true;

% run EM optimization
[alpha, negative_params, positive_params, w_labeled] = ...
    PNU_nested_em(unlabeled, labeled_pos, labeled_neg, num_components, ...
    progress_bar);
