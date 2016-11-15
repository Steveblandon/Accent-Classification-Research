%system controls
partition = 0;          %1=on, 0=off
trainGMMs = 1;          %1=on, 0=off
testGMMs = 1;           %1=on, 0=off
k = 2;                  %number of mixture components if training GMMs
epochs = 1;             %number of times to run test
findBestK = 0;          %1=on, 0=off, if active on each epoch k increases by a factor of 2
logData = 1;            %1=on, 0=off
testRatio = 0.25;       %ratio of data assigned to testset, must repartition data
GMMmaxIter = 1000;      %maximum number of iterations to allow for EM optimization
datapath = 'C:\Users\steve\Workshop\FAE corpora\fullset';
logpath = '.\Logs';

% for standalone app, specify settings
disp('Initializing ACR_System 1');
disp('-------------------------');
disp('default settings:');
disp(['partition: ', num2str(partition), ' (1=on, 0=off)']);
disp(['trainGMMs: ', num2str(trainGMMs), ' (1=on, 0=off)']);
disp(['testGMMs: ', num2str(testGMMs), ' (1=on, 0=off)']);
disp(['k: ', num2str(k),' (number of mixture components if training GMMs)']);
disp(['epochs: ', num2str(epochs), ' (number of times to run test)']);
disp(['findBestK: ', num2str(findBestK), ' (1=on, 0=off, if active on each epoch k increases by a factor of 2)']);
disp(['logData: ', num2str(logData), ' (1=on, 0=off)']);
disp(['testRatio: ', num2str(testRatio), ' (between 0 and 1)']);
disp(['GMMmaxIter: ', num2str(GMMmaxIter), ' (maximum number of iterations to allow for EM optimization)']);
disp(['datapath: ', datapath]);
disp(['logpath: ', logpath]);
prompt = 'would you like to change settings? Y/N [N]: ';
str = input(prompt,'s');
if strcmp(str,'Y') || strcmp(str,'y')
    disp('you can change only what you need to change, any empty inputs will use default values...');
    partition = input('partition: ');
    if isempty(partition)
        partition = 0;
    end
    trainGMMs = input('trainGMMs: ');
    if isempty(trainGMMs)
        trainGMMs = 1;
    end
    testGMMs = input('testGMMs: ');
    if isempty(testGMMs)
        testGMMs = 1;
    end
    k = input('k: ');
    if isempty(k)
        k = 2;
    end
    epochs = input('epochs: ');        
    if isempty(epochs)
        epochs = 1;
    end
    findBestK = input('findBestK: ');
    if isempty(findBestK)
        findBestK = 0;
    end
    logData = input('logData: ');
    if isempty(logData)
        logData = 1;
    end
    testRatio = input('testRatio: ');
    if isempty(testRatio)
        testRatio = 0.25;
    end
    GMMmaxIter = input('GMMmaxIter: ');
    if isempty(GMMmaxIter)
        GMMmaxIter = 1000;
    end
    datapath = input('datapath: ','s');
    if isempty(datapath)
        datapath = 'C:\Users\steve\Workshop\FAE corpora\fullset';
    end
    logpath = input('logpath: ','s');
    if isempty(logpath)
        logpath = '.\Logs';
    end
end
disp('-------------------------');


%dependencies for development testing
%addpath('C:\Users\steve\Workshop\Accent Classification Research\S1\Libraries\rastamat');


%data specification
if exist('model','var') == 0
    model = {};
%     model{1}.class = 'brazilian';
%     model{1}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\BP';
%     model{2}.class = 'mandarin';
%     model{2}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\MA';
%     model{3}.class = 'russian';
%     model{3}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\RU';
%     model{4}.class = 'italian';
%     model{4}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\IT';
    classes = dir(datapath);
    class_count = length(classes);
    parfor i=3:class_count
        model{i-2}.class =  classes(i).name;
        model{i-2}.path = [datapath, '\', classes(i).name];
    end
    class_count = length(model);
end


%get metadata on partitions (trainset/testset)
dataSize = 0;
if partition == 1
    disp('partitioning data...');
end
parfor i=1:class_count
    [model{i}, sampleCount] = partitionData(model{i},testRatio, partition);
    dataSize = dataSize + sampleCount;
end
partition = 0;


%determine number of tests to carry out
if findBestK == 0
   epochs = 1;
end
detectionRate = zeros(epochs,2);

if logData == 1
    if exist(logpath,'dir') == 0
            mkdir(logpath);
    end
    diary([logpath,'\log__',datestr(datetime('now'),'mm-dd-yyyy_HHMMSS'),'.txt']); 
end

% if a single model doesn't have extracted features, activate feature
% extraction control
extractFeats = 0;
for i=1:class_count
    if isfield(model{i},'trainset') == 0
        extractFeats = 1;
        break;
    end 
end

featsExtracted = 0;
for epoch=1:epochs
    disp('--------------------------------------------------');
    disp(['epoch:',num2str(epoch),' | dataset_size:',num2str(dataSize),...
        ' | classes=',num2str(class_count),' | testRatio=',num2str(testRatio),...
        ' | k=',num2str(k)]);
    disp('--------------------------------------------------');
    tic;    %start timer
    if trainGMMs == 1
        %extract features first if not present
        if extractFeats == 1
           %extract PLP features
           disp('commencing PLP feature extraction...');
            parfor i=1:class_count
                disp(['extracting PLP features [', model{i}.class,']...']);
                model{i}.trainset = featExtractPLP(model{i}.trainset_raw,...
                    model{i}.path_train);  
            end
            extractFeats = 0;
        end 
        parfor i=1:class_count
            %train GMMs
            disp(['commencing EM optimization [',model{i}.class,']...']);    
            model{i}.gmmfit = fitgmdist(model{i}.trainset, k, 'Options',... 
            statset('maxIter',GMMmaxIter),...
            'CovarianceType','diagonal','RegularizationValue',1e-12); 
            conv = 'FALSE';
            if model{i}.gmmfit.Converged == 1
                conv = 'TRUE';
            end
            disp(['EM optimization for GMM complete [', model{i}.class,...
                ']>> convergence: ',conv,...
                '  |  log-nlogl: -',num2str(log(model{i}.gmmfit.NegativeLogLikelihood)),...
                '  |  AIC: ',num2str(model{i}.gmmfit.AIC),...
                '  |  iterations: ',num2str(model{i}.gmmfit.NumIterations)]);
        end

    end

    %test GMMs
    if testGMMs == 1
        disp('setting up confusion matrix for testing...');
        labels = cell(class_count,1);
        totalSampleCount = 0;
        for i=1:class_count
           labels{i} =  model{i}.class;
           totalSampleCount = totalSampleCount + length(model{i}.testset_raw);
        end
        confMatrix = confMat_construct(labels);
        likelihood = zeros(class_count,1);
        disp('commencing testing phase...');
        for i=1:class_count
            set = model{i}.testset_raw;
            sampleCount = length(set);
            sample = cell(sampleCount,1);
            path = model{i}.path_test;
            parfor s=1:sampleCount
               sample{s} = featExtractPLP(set(s), path);
            end
            for s=1:sampleCount
                for c=1:class_count
                    likelihood(c) = log(sum(pdf(model{c}.gmmfit, sample{s})));
                end
                [maxVal, indx] = max(likelihood);
                confMatrix{i+1,indx+1} = confMatrix{i+1,indx+1} + 1;
            end
        end
        %analyze results
        disp('testing phase complete.');
        disp(confMatrix);
        matValues = reshape([confMatrix{2:class_count+1,2:class_count+1}],[class_count,class_count]);
        acc = sum(diag(matValues))/sum(sum(matValues));
        disp(['sample count: ', num2str(totalSampleCount), ' | accuracy: ', num2str(acc*100),'%']);
    end
    toc;    %check timer
    if findBestK == 1
        detectionRate(epoch,:) = [k , acc];
        k = k * 2;
    end
end

if findBestK == 1
   [bestAcc, index] = max(detectionRate(:,2));
   bestK = detectionRate(index,1);
   disp(['highest detection rate of ',num2str(bestAcc),' achieved with  k=',num2str(bestK)]);
end
diary off;