hiddenLayerSize = floor((52+2)/2);
fExtract = 0;       %force feature extraction

%dependencies for development testing
addpath('C:\Users\steve\Workshop\Accent Classification Research\S1\Libraries\rastamat');


%data specification
if exist('model','var') == 0
    model = {};
    model{1}.class = 'brazilian';
    model{1}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\BP';
    model{2}.class = 'mandarin';
    model{2}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\MA';
%     model{3}.class = 'russian';
%     model{3}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\RU';
%     model{4}.class = 'italian';
%     model{4}.path = 'C:\Users\steve\Workshop\FAE corpora\subset_200\IT';
%     classes = dir(datapath);
%     class_count = length(classes);
%     parfor i=3:class_count
%         model{i-2}.class =  classes(i).name;
%         model{i-2}.path = [datapath, '\', classes(i).name];
%     end
    class_count = length(model);
end

%retrieve audio meta data
for i=1:length(model)
   model{i}.audio = getAudio(model{i}.path);
end

tic;
% if a single model doesn't have data, proceed to loading/extracting it
getFeats = 0;
for i=1:class_count
    if isfield(model{i},'data') == 0
        getFeats = 1;
        break;
    end 
end
if getFeats == 1
    if fExtract == 1 && exist('.\Features','dir') ~= 0
        rmdir('.\Features','s');
    end
    if exist('.\Features','dir') == 0
       mkdir('.\Features');
    end
    feats = dir('.\Features');
    if length(feats)-2 == class_count
        %load features from folders corresponding to each class
        for c=1:class_count
            m = model{c};
            feat_path = ['.\Features\',m.class];
            sC = length(m.audio);     %sample count
            m.fC = 0;                   %frame count
            m.data = cell(sC,1);
            for i=1:sC
                name = m.audio(i).name(1:strfind(m.audio(i).name, '.wav')-1);
                load([feat_path,'\',name]);
                m.data{i} = sample;
                m.fC = m.fC + length(sample);
            end
            model{c} = m;
       end
    else
       %extract features and save them in folders corresponding to each class
       for c=1:class_count
           m = model{c};
           feat_path = ['.\Features\',m.class];
           if exist(feat_path,'dir') ~= 0
               rmdir(feat_path,'s');
           end
           mkdir(feat_path);
           sC = length(m.audio);     %sample count
           m.fC = 0;                   %frame count
           m.data = cell(sC,1);
           for i=1:sC
               sample = featExtractPLP(m.path,m.audio(i));
               name = m.audio(i).name(1:strfind(m.audio(i).name, '.wav')-1);
               save([feat_path,'\',name],'sample');
               m.data{i} = sample;
               m.fC = m.fC + length(sample);
           end
           model{c} = m;
       end
    end
end


tfC = 0;         %total frame count
for c=1:class_count
    tfC = tfC + model{c}.fC;
end
inputs = zeros(size(m.data{1},1),tfC);
targets = zeros(1,tfC);
indS = 1;
indE = length(m.data{1});
for c=1:class_count
    m = model{c};
    sC = length(m.data);   %sample count
    for i=1:sC
        inputs(:,indS:indE) = m.data{i};
        targets(:,indS:indE) = c;
        indS = indE + 1;
        if i == sC
            indE = indE + length(m.data{i});
        else
            indE = indE + length(m.data{i+1});
        end
    end
end

scount = 60000;
inputs = [inputs(:,1:scount),inputs(:,length(inputs)-scount:length(inputs))];
targets = ind2vec(targets);
targets = [targets(:,1:scount),targets(:,length(targets)-scount:length(targets))];
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
net = patternnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net, tr] = train(net, inputs, targets);

% Test the Network
predictions = net(inputs);
error = gsubtract(targets,predictions);
performance = perform(net,targets,predictions);
tind = vec2ind(targets);
pind = vec2ind(predictions);
percentErrors = sum(tind ~= pind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(error)
% figure, plotconfusion(t,predictions)
% figure, plotroc(t,predictions)


toc