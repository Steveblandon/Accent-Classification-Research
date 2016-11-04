function [plpSet] = PLPFeatureExtraction(set, path)

% NOTE: the following extracts 12th order PLP-based 13 dimensional feature 
% vectors. 
% the first, second, and third derivatives of the matrix are concatenated
% onto the matrix as rows for a complete 52 dimensional feature matrix. 

plpSet = [];
sampleCount = length(set);
counter = 0;
notificationIntervals = 25;

for sample = set'
    counter = counter + 1;
    % Load speech waveform
    [d,sr] = audioread([path,'\',sample.name]);

    % Calculate 12th order PLP features without RASTA
    cep = rastaplp(d, sr, 0, 12);

    % calculate deltas to obtain 1st, 2nd and 3rd derivatives
    d1 = deltas(cep);
    d2 = deltas(deltas(cep,5),5);
    d3 = deltas(deltas(deltas(cep,3),3),3);

    % concatenate original features with derivatives
    features = [cep;d1;d2;d3]; 

    % add features into sequence
    plpSet = [plpSet,features];
    if mod(counter, notificationIntervals) == 0
        disp([num2str(counter/sampleCount*100),'% complete...']);
    end
end

plpSet = plpSet';     %invert to have n by p matrix where n is frames (observations/samples)
                %and p is parameters/features