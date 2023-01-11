function [model] = ovrtrain(y, x)

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    %models{i} = fitcsvm(x, double(y == labelSet(i)),'Standardize',true, 'KernelFunction','RBF');
    models{i} = fitcsvm(x, double(y == labelSet(i)),'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', ...
    'expected-improvement-plus'))
end

model = struct('models', {models}, 'labelSet', labelSet);
