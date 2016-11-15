function [updatedModel,sampleCount] = partitionData(model, testRatio, repartition)

updatedModel = model;
dataset = dir([model.path,'\*.wav']);
sampleCount = length(dataset);

if repartition == 1
    %set up data for repartition if already partitioned
    if sampleCount == 0 && exist([model.path,'\trainset'],'dir') ~= 0 && exist([model.path,'\testset'],'dir') ~= 0
        trainset =  dir([model.path,'\trainset\*.wav']);
        testset =  dir([model.path,'\testset\*.wav']);
        for sample = trainset'
            movefile([model.path,'\trainset\',sample.name],[model.path,'\',sample.name]);
        end
        for sample = testset'
            movefile([model.path,'\testset\',sample.name],[model.path,'\',sample.name]);
        end
        dataset = dir([model.path,'\*.wav']);
        sampleCount = length(dataset);
    end


    %partition data by random sampling
    [trainset, ind] = datasample(dataset,...
                    round((1-testRatio)*sampleCount), 'replace', false);
    testset = dataset(setdiff(1:sampleCount,ind),:);


    %update folders
    for sample = trainset'
        if exist([model.path,'\trainset'],'dir') == 0
            mkdir([model.path,'\trainset']);
        end
        movefile([model.path,'\',sample.name],[model.path,'\trainset\',sample.name]);
    end

    for sample = testset'
        if exist([model.path,'\testset'],'dir') == 0
            mkdir([model.path,'\testset']);
        end
        movefile([model.path,'\',sample.name],[model.path,'\testset\',sample.name]);
    end
else
    trainset =  dir([model.path,'\trainset\*.wav']);
    testset =  dir([model.path,'\testset\*.wav']);
    sampleCount = length(trainset) + length(testset);
end


%update structure
updatedModel.path_train = [model.path,'\trainset'];
updatedModel.path_test = [model.path,'\testset'];
updatedModel.trainset_raw = trainset;
updatedModel.testset_raw = testset;