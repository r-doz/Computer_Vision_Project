function [pred, ac, decv] = ovrpredict(y, x, model)

labelSet = model.labelSet;
labelSetSize = length(labelSet);
models = model.models;
decv= zeros(size(y, 1), labelSetSize);
Scores = zeros(size(y, 1),labelSetSize);

for i=1:labelSetSize
  [~,score]= predict(models{i}, x);
  Scores(:,i) = score(:,2);
  %decv(:, i) = d * (2 * models{i}.Label(1) - 1);
end
%[~,pred] = max(decv, [], 2);
[~,pred] = max(Scores,[],2);
pred = labelSet(pred);
ac = sum(y==pred) / size(x, 1);
