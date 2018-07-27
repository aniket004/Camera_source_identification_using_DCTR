
clear all;
clc;
data = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Dresden_DCTR_1_5021.csv');
feature = data(:,1:end-1);
label = data(:,end);

% [eigenvectors, projected_data, eigenvalues] = princomp(feature);
% [foo, feature_idx] = sort(eigenvalues, 'descend');
% selected_projected_data = projected_data(:, feature_idx(1:1000));

%data = selected_projected_data;
%label = 

fold = 5;


indices = crossvalind('Kfold',length(data(:,1)),fold);
cp = classperf(label);
for i = 1:fold
    test = (indices == i); 
    train = ~test;
    class = TreeBagger(100,feature(train,:),label(train,:),'Method','Classification');
    [Y_tb, classifScore] = class.predict(feature(test,:));
    classperf(cp,double(nominal(Y_tb))-1,double(label(test,:))-1);
    size(Y_tb)
    size(test)
end
cp.ErrorRate




% % Train the classifier
% tb = TreeBagger(100,Xtrain,Ytrain,'Method','Classification');
% 
% % Make a prediction for the test set
% [Y_tb, classifScore] = tb.predict(Xtest);
% Y_tb = nominal(Y_tb);