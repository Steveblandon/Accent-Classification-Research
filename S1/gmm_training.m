data = [];
k = 3;              % number of mixture components
maxIter = 1000;     % maximum number of iterations to allow for EM


%load features from features directory, currently supporting only .mat
%files
data = loadFeatures();

%initialize GMM using K-means
gmIni = kmeans(data, k);    %vector of cluster assignments for each sample

%fit GMM
gmm = fitgmdist(data, k, 'Options', statset('MaxIter',1000),...
    'CovarianceType','diagonal', 'Start', gmIni);

%clear features directory
rmdir('Models','s');
mkdir('Models');

%save model
modelDir = dir('Models\*.mat');
modelCount = size(modelDir,1);
save(['Models\gmm_model_',num2str(modelCount+1),'.mat'],'gmm');