%control
extractFeatures = 0;    %1=on, 0=off
trainGMMs = 1;          %1=on, 0=off
testGMMs = 1;           %1=on, 0=off
testRatio = 0.25;       %ratio of data assigned to testset
k = 8;              % number of mixture components if training GMMs
maxIter = 1000;     % maximum number of iterations to allow for EM

if extractFeatures == 1
    %load needed libraries
    path(path,'C:\Users\steve\Google Drive\Academic\Research\Steve Research\Systems\S1\Libraries\rastamat');

    %specify audio to extract features from
    audio = {};
    audio{1}.class = 'italian';
    audio{1}.path = 'C:\Users\steve\Downloads\FAE corpora\subset_IT_RU_200\IT';
    audio{2}.class = 'russian';
    audio{2}.path = 'C:\Users\steve\Downloads\FAE corpora\subset_IT_RU_200\RU';

    %extract PLP features
    class_count = length(audio);
    for i=1:class_count
        audio{i}.features = PLPFeatureExtraction(audio{i});        
    end
end


%train GMMs
if trainGMMs == 1
    class_count = length(audio);
    for i=1:class_count
        trainset = audio{i}.features;
        sampleCount = length(trainset);
        ind = 1:sampleCount;
        if testGMMs == 1
            [trainset, ind] = datasample(trainset,...
                round((1-testRatio)*sampleCount), 'replace', false);
            audio{i}.testset = audio{i}.features(setdiff(1:sampleCount,ind),:);
        end
        disp(['commencing k-means cluster initialization [',audio{i}.class,']...']);
        gmIni = kmeans(trainset, k, 'Display','iter', 'maxIter', maxIter);    %initialize GMM using K-means
        disp(['commencing EM optimization [',audio{i}.class,']...']);
        audio{i}.gmmfit = fitgmdist(trainset, k, 'Options',... 
        statset('Display','iter','maxIter',maxIter),...
        'CovarianceType','diagonal',...
        'Start', gmIni); 
        disp(['converged?',num2str(audio{i}.gmmfit.Converged),',...',num2str(audio{i}.gmmfit.AIC)]);
        disp(['GMM for ', audio{i}.class, ' complete.']);
    end
    
end


%test GMM
if testGMMs == 1
    
end