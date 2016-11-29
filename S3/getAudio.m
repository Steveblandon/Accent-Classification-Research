function [dataset] = getAudio(path)

%this only loads the audio file names, not the actual files

dataset = dir([path,'\*.wav']);
sampleCount = length(dataset);

% the trainset/testset folders are for those other systems that might
% partition the audio files
if sampleCount == 0 && exist([path,'\trainset'],'dir') ~= 0 && exist([path,'\testset'],'dir') ~= 0
    trainset =  dir([path,'\trainset\*.wav']);
    testset =  dir([path,'\testset\*.wav']);
    for sample = trainset'
        movefile([path,'\trainset\',sample.name],[path,'\',sample.name]);
    end
    for sample = testset'
        movefile([path,'\testset\',sample.name],[path,'\',sample.name]);
    end
    dataset = dir([path,'\*.wav']);
end

