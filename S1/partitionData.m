function [updatedAudio,sampleCount] = partitionData(audio, testRatio)

updatedAudio = audio;
dataset = dir([audio.path,'\*.wav']);
sampleCount = length(dataset);


%set up data for repartition if already partitioned
if sampleCount == 0 && exist([audio.path,'\trainset'],'dir') ~= 0 && exist([audio.path,'\testset'],'dir') ~= 0
    trainset =  dir([audio.path,'\trainset\*.wav']);
    testset =  dir([audio.path,'\testset\*.wav']);
    for sample = trainset'
        movefile([audio.path,'\trainset\',sample.name],[audio.path,'\',sample.name]);
    end
    for sample = testset'
        movefile([audio.path,'\testset\',sample.name],[audio.path,'\',sample.name]);
    end
    dataset = dir([audio.path,'\*.wav']);
    sampleCount = length(dataset);
end


%partition data by random sampling
[trainset, ind] = datasample(dataset,...
                round((1-testRatio)*sampleCount), 'replace', false);
testset = dataset(setdiff(1:sampleCount,ind),:);


%update folders
for sample = trainset'
    if exist([audio.path,'\trainset'],'dir') == 0
        mkdir([audio.path,'\trainset']);
    end
    movefile([audio.path,'\',sample.name],[audio.path,'\trainset\',sample.name]);
end

for sample = testset'
    if exist([audio.path,'\testset'],'dir') == 0
        mkdir([audio.path,'\testset']);
    end
    movefile([audio.path,'\',sample.name],[audio.path,'\testset\',sample.name]);
end


%update structure
updatedAudio.path_train = [audio.path,'\trainset'];
updatedAudio.path_test = [audio.path,'\testset'];
updatedAudio.trainset_raw = trainset;
updatedAudio.testset_raw = testset;