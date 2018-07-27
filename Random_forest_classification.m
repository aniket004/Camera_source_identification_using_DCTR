clear all;
clc;
feature = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Dresden_DCTR_1_5021.csv');
no_of_sample = length(feature(:,1));
cv_feature = feature(:,1:end-1);

label = feature(:,end);
k = 10;
X = cv_feature;
Y = label;

%cv = cvpartition(length(cv_feature),'KFold',k);
cv = cvpartition(no_of_sample,'holdout',0.4);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

% disp('Training Set')
% tabulate(Ytrain)
% disp('Test Set')
% tabulate(Ytest)

% Cost of misclassification
% cost = [0 1
%         5 0];
%opts = statset('UseParallel',false);
% Train the classifier
tb = TreeBagger(100,Xtrain,Ytrain,'Method','Classification');

% Make a prediction for the test set
[Y_tb, classifScore] = tb.predict(Xtest);
Y_tb = nominal(Y_tb);

% Compute the confusion matrix
C_tb = confusionmat(double(Ytest),double(Y_tb));
% Examine the confusion matrix for each class as a percentage of the true class
C_tb = bsxfun(@rdivide,C_tb,sum(C_tb,2)) * 100