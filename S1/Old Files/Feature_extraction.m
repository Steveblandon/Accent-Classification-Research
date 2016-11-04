% NOTE: the following extracts 12th order PLP-based 13 dimensional feature 
% vectors. These feature vectors, each corresponding to a single frame, are 
% represented as columns in a matrix. 
% the first, second, and third derivatives of the matrix are concatenated
% onto the matrix as rows for a complete 52 dimensional feature matrix. 


%load samples from audio directory
samples = dir('Audio\*.wav');

%clear features directory
rmdir('Features','s');
mkdir('Features');

for sample = samples'
    % Load speech waveform
    [d,sr] = audioread(['Audio\',sample.name]);

    % Calculate 12th order PLP features without RASTA
    cep = rastaplp(d, sr, 0, 12);

    % calculate deltas to obtain 1st, 2nd and 3rd derivatives
    d1 = deltas(cep);
    d2 = deltas(deltas(cep,5),5);
    d3 = deltas(deltas(deltas(cep,3),3),3);

    % concatenate original features with derivatives
    Feats = [cep;d1;d2;d3];

    % save features either in matlab format or ascii
    save(['Features\',sample.name,'_fts','.mat'],'Feats');
    %save(['Features\',sample.name,'_fts','.txt'],'Feats','-ascii');
end