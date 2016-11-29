function [confMatrix] = confMat_construct(labels)

% construct a confusion matrix given a set of labels

classCount = length(labels);
confMatrix = repmat({0},classCount+1,classCount+1);
for i=1:classCount
   confMatrix{1,i+1} = labels{i};
   confMatrix{i+1,1} = labels{i};
end
confMatrix{1,1} = '';