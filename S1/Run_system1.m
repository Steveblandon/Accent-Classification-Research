%system controls
partition = 0;          %1=on, 0=off
trainGMMs = 1;          %1=on, 0=off
testGMMs = 0;           %1=on, 0=off
logData = 1;            %1=on, 0=off
findBestK = 0;          %1=on, 0=off
epochs = 1;             %number of trials to find best K, max k = k^epochs if k=2
progressFactor = 0;     %intervals at which progress is shown as a percentage (for feature extraction and testing)
testRatio = 0.25;       %ratio of data assigned to testset, must repartition data
k = 2;              % number of mixture components if training GMMs
GMMmaxIter = 100000;     % maximum number of iterations to allow for GMM training

%dependencies
addpath('C:\Users\steve\Workshop\Accent Classification Research\S1\Libraries\rastamat');

%data specification
if exist('audio','var') == 0
    audio = {};
%     audio{1}.class = 'brazilian';
%     audio{1}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\BP';
%     audio{2}.class = 'mandarin';
%     audio{2}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\MA';
%     audio{3}.class = 'russian';
%     audio{3}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\RU';
%     audio{4}.class = 'italian';
%     audio{4}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\IT';
    path = 'C:\Users\steve\Workshop\FAE corpora\fullset';
    classes = dir(path);
    class_count = length(classes);
    parfor i=3:class_count
        audio{i-2}.class =  classes(i).name;
        audio{i-2}.path = [path, '\', classes(i).name];
    end
    class_count = length(audio);
end


%get metadata on partitions (trainset/testset)
dataSize = 0;
parfor i=1:class_count
    [audio{i}, sampleCount] = partitionData(audio{i},testRatio, partition);
    dataSize = dataSize + sampleCount;
end
partition = 0;


%determine number of tests to carry out
if findBestK == 0
   epochs = 1;
end
detectionRate = zeros(epochs,2);

if logData == 1
   diary(['.\Logs\log__',datestr(datetime('now'),'mm-dd-yyyy_HHMMSS'),'.txt']); 
end


for epoch=1:epochs
    disp('--------------------------------------------------');
    disp(['epoch:',num2str(epoch),' | dataset_size:',num2str(dataSize),...
        ' | classes=',num2str(class_count),' | testRatio=',num2str(testRatio),...
        ' | k=',num2str(k)]);
    disp('--------------------------------------------------');
    tic;    %start timer
    if trainGMMs == 1
        parfor i=1:class_count
            if partition == 1 || isfield(audio{i},'trainset') == 0
                %extract PLP features
                disp(['extracting PLP features [', audio{i}.class,']...']);
                audio{i}.trainset = featExtractPLP(audio{i}.trainset_raw,...
                    audio{i}.path_train, progressFactor);
            end
            %train GMMs
            disp(['commencing EM optimization [',audio{i}.class,']...']);    
            audio{i}.gmmfit = fitgmdist(audio{i}.trainset, k, 'Options',... 
            statset('maxIter',GMMmaxIter),...
            'CovarianceType','diagonal','RegularizationValue',1e-12); 
            conv = 'FALSE';
            if audio{i}.gmmfit.Converged == 1
                conv = 'TRUE';
            end
            disp(['EM optimization for GMM complete [', audio{i}.class,...
                ']>> convergence: ',conv,...
                '  |  log-nlogl: -',num2str(log(audio{i}.gmmfit.NegativeLogLikelihood)),...
                '  |  AIC: ',num2str(audio{i}.gmmfit.AIC),...
                '  |  iterations: ',num2str(audio{i}.gmmfit.NumIterations)]);
        end

    end

    %test GMMs
    if testGMMs == 1
        disp('creating confusion matrix for testing phase...');
        labels = cell(class_count,1);
        totalSampleCount = 0;
        counter = 1;
        for i=1:class_count
           labels{i} =  audio{i}.class;
           totalSampleCount = totalSampleCount + length(audio{i}.testset_raw);
        end
        notifIntervals = round(totalSampleCount/progressFactor);
        confMatrix = confMat_construct(labels);
        likelihood = zeros(class_count,1);
        disp('commencing testing phase...');
        for i=1:class_count
            sampleCount = length(audio{i}.testset_raw);
            for s=1:sampleCount
                sample = featExtractPLP(audio{i}.testset_raw(s),audio{i}.path_test,progressFactor);
                for c=1:class_count
                    likelihood(c) = log(sum(pdf(audio{c}.gmmfit, sample)));     %use pdf 
%                     [p, likelihood(c)] = posterior(audio{c}.gmmfit, sample);    %use nlogl from posterior
%                     p = posterior(audio{c}.gmmfit, sample);                     %use log posterior
%                     likelihood(c) = sum(sum(p));
                end
                [m, indx] = max(likelihood);
                confMatrix{i+1,indx+1} = confMatrix{i+1,indx+1} + 1;
                if mod(counter, notifIntervals) == 0
                    disp([num2str(counter/totalSampleCount*100),'% complete...']);
                end
                counter = counter + 1;
            end
        end
        %analyze results
        disp('testing phase complete');
        disp(confMatrix);
        matValues = reshape([confMatrix{2:class_count+1,2:class_count+1}],[class_count,class_count]);
        acc = sum(diag(matValues))/sum(sum(matValues));
        disp(['sample count: ', num2str(totalSampleCount), ' | accuracy: ', num2str(acc*100),'%']);
    end
    toc;    %check timer
    if findBestK == 1
       detectionRate(epoch,:) = [k , acc];
    end
    if k == 1
        k = 2;
    else
        k = k * 2;
    end
end

if findBestK == 1
   [bestAcc, index] = max(detectionRate(:,2));
   bestK = detectionRate(index,1);
   disp(['highest detection rate of ',num2str(bestAcc),' achieved with  k=',num2str(bestK)]);
   plot(detectionRate(:,1),detectionRate(:,2));
end
diary off;