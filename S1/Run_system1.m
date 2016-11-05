%system controls
partition = 1;          %1=on, 0=off
extractFeatures = 1;    %1=on, 0=off
trainGMMs = 1;          %1=on, 0=off
testGMMs = 1;           %1=on, 0=off
logData = 1;            %1=on, 0=off
findBestK = 1;          %1=on, 0=off
epochs = 10;             %number of trials to find best K
showProgress = 'final'; %'final' = partial | 'iter' = on | 'off'
progressFactor = 0;     %intervals at which progress is shown as a percentage
testRatio = 0.25;       %ratio of data assigned to testset
k = 2;              % number of mixture components if training GMMs
GMMmaxIter = 100000;     % maximum number of iterations to allow for GMM training


%dependencies
path(path,'C:\Users\steve\Workshop\Accent Classification Research\S1\Libraries\rastamat');

%data specification
if exist('audio','var') == 0
    audio = {};
    audio{1}.class = 'brazilian';
    audio{1}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_IT_RU_200\BP';
    audio{2}.class = 'mandarin';
    audio{2}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_IT_RU_200\MA';
    %audio{3}.class = 'russian';
    %audio{3}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_IT_RU_200\RU';
    class_count = length(audio); 
end


%partition data into training set and test set
if partition == 1
    dataSize = 0;
    for i=1:class_count
        [audio{i}, sampleCount] = partitionData(audio{i},testRatio);
        dataSize = dataSize + sampleCount;
    end
end


%determine number of tests to carry out
if findBestK == 0
   epochs = 1;
end
errorRate = zeros(epochs,2);

if logData == 1
   diary(['.\Logs\log__',datestr(datetime('now'),'mm-dd-yyyy_HHMMSS')]); 
end


for epoch=1:epochs
    disp('--------------------------------------------------');
    disp(['epoch:',num2str(epoch),' | dataset_size:',num2str(dataSize),...
        ' | classes=',num2str(class_count),' | testRatio=',num2str(testRatio),...
        ' | k=',num2str(k)]);
    disp('--------------------------------------------------');
    tic;    %start timer
    if trainGMMs == 1
        for i=1:class_count
            if partition == 1 || isfield(audio{i},'trainset') == 0
                %extract PLP features
                disp(['extracting PLP features [', audio{i}.class,']...']);
                audio{i}.trainset = featExtractPLP(audio{i}.trainset_raw,...
                    audio{i}.path_train, progressFactor);
            end
            %train GMMs
            disp(['initializing GMM with k-means clustering [',audio{i}.class,']...']);
            gmIni = kmeans(audio{i}.trainset, k, 'Display',showProgress,...
                'maxIter', GMMmaxIter);
            disp(['commencing EM optimization [',audio{i}.class,']...']);
            audio{i}.gmmfit = fitgmdist(audio{i}.trainset, k, 'Options',... 
            statset('Display',showProgress,'maxIter',GMMmaxIter),...
            'CovarianceType','diagonal',...
            'Start', gmIni); 
            disp(['EM optimization for GMM complete [', audio{i}.class, '].']);
            conv = 'FALSE';
            if audio{i}.gmmfit.Converged == 1
                conv = 'TRUE';
            end
            disp(['converged:',conv,', | AIC:',num2str(audio{i}.gmmfit.AIC)]);
            disp('...');
        end

    end

    %test GMMs
    if testGMMs == 1
        disp('creating confusion matrix for testing phase...');
        labels = cell(class_count,1);
        totalSampleCount = 0;
        counter = 0;
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
                counter = counter + 1;
                sample = featExtractPLP(audio{i}.testset_raw(s),audio{i}.path_test,progressFactor);
                for c=1:class_count
                    likelihood(c) = log(sum(pdf(audio{c}.gmmfit, sample)));
                end
                [m, indx] = max(likelihood);
                confMatrix{i+1,indx+1} = confMatrix{i+1,indx+1} + 1;
                if mod(counter, notifIntervals) == 0
                    disp([num2str(counter/totalSampleCount*100),'% complete...']);
                end
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
       errorRate(epoch,:) = [k , 1-acc];
    end
    k = k * 2;
end

if findBestK == 1
   [bestErr, index] = min(errorRate(:,2));
   bestK = errorRate(index,1);
   disp(['lowest error rate of ',num2str(bestErr),' achieved with  k=',num2str(bestK)]);
end
diary off;