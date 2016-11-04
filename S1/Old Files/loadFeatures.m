function [data] = loadFeatures()

data = [];

%load features from features directory, currently supporting only .mat
%files
samples = dir('Features\*.mat');

for sample = samples'
    % Load feature file and append to data
    d = load(['Features\',sample.name]);
    data = [data, d.Feats];     % rows are features, columns are frames
end
data = data'; % rows are samples (frames), columns are features (variables)