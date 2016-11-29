function [features] = featExtractPLP(path, sample)

% NOTE: the following extracts 12th order PLP-based 13 dimensional feature 
% vectors. 
% the first, second, and third derivatives of the matrix are concatenated
% onto the matrix as rows for a complete 52 dimensional feature matrix. 


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



