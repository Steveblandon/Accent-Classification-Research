%system controls
partition = 0;          %1=on, 0=off
extractFeatures = 1;    %1=on, 0=off
trainGMMs = 1;          %1=on, 0=off
testGMMs = 1;           %1=on, 0=off
showProgress = 'final'; %'final' = partial | 'iter' = on | 'off'
testRatio = 0.25;       %ratio of data assigned to testset
k = 1;              % number of mixture components if training GMMs
maxIter = 1000;     % maximum number of iterations to allow for EM


%dependencies
path(path,'C:\Users\steve\Workshop\Accent Classification Research\S1\Libraries\rastamat');

%data specification
if exist('audio','var') == 0
    audio = {};
    audio{1}.class = 'italian';
    audio{1}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_IT_RU_200\IT';
    audio{2}.class = 'russian';
    audio{2}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_IT_RU_200\RU';
    class_count = length(audio); 
end


%partition data into training set and test set
if partition == 1
    for i=1:class_count
        audio{i} = partitionData(audio{i},testRatio);
    end
end


if trainGMMs == 1
    for i=1:class_count
        if partition == 1 || isfield(audio{i},'trainset') == 0
            %extract PLP features
            disp(['initializing feature extraction for ', audio{i}.class]);
            audio{i}.trainset = PLPFeatureExtraction(audio{i}.trainset_raw,audio{i}.path_train);
        end
        %train GMMs
        disp(['commencing k-means cluster initialization [',audio{i}.class,']...']);
        gmIni = kmeans(audio{i}.trainset, k, 'Display',showProgress, 'maxIter', maxIter);    %initialize GMM using K-means
        disp(['commencing EM optimization [',audio{i}.class,']...']);
        audio{i}.gmmfit = fitgmdist(audio{i}.trainset, k, 'Options',... 
        statset('Display',showProgress,'maxIter',maxIter),...
        'CovarianceType','diagonal',...
        'Start', gmIni); 
        disp(['converged?',num2str(audio{i}.gmmfit.Converged),', | AIC:',num2str(audio{i}.gmmfit.AIC)]);
        disp(['GMM for ', audio{i}.class, ' complete.']);
        disp('...');
    end
    
end


%test GMM
if testGMMs == 1
    
end