

%% Reading data
data = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Dresden_DCTR_1_10507.csv');
feature = data(:,1:end-1);
label = data(:,end);

%% Dimensionality reduction by PCA
[eigenvectors, projected_data, eigenvalues] = princomp(feature);
[foo, feature_idx] = sort(eigenvalues, 'descend');
selected_projected_data = projected_data(:, feature_idx(1:160));

feature = selected_projected_data;
%%

    X = feature;
    y = label;
    

 % Return the confusion matrix using stratified 10-fold
       % cross-validation.
       % Each of the ten confusion matrices needs to sort the group labels
       % according to the same order
       yorder = unique(y);   
       % A stratified partition is preferred to evaluate classification
       % algorithms.   
       cp = cvpartition(y,'k',10);
       f = @(xtr,ytr,xte,yte) confusionmat(yte,str2double(predict(TreeBagger(500,xtr,ytr,'Method','classification'),xte)),'order',yorder);
       cfMat = crossval(f,feature,label,'partition',cp);
       % cfMat is the summation of 10 confusion matrices from 10 test sets.
       cfMat1 = reshape(sum(cfMat),10,10);
       
       [row col] = size(cfMat1);
       
       acc = 0;
       for i = 1: row
           cf(i,:) = 100*cfMat1(i,:)/sum(cfMat1(i,:));
           acc = acc + cf(i,i);
       end
       
       cf
       
       Accuracy = acc/10