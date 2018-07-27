

%% Reading data
data = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Dresden_DCTR_1_10507.csv');
feature_data = data(:,1:end-1);
label = data(:,end);

%% Dimensionality reduction by PCA
[eigenvectors, projected_data, eigenvalues] = princomp(feature_data);
[foo, feature_idx] = sort(eigenvalues, 'descend');
selected_projected_data = projected_data(:, feature_idx(1:165));

feature = selected_projected_data;
%%

    X = feature;
    y = label;
    %data partition
    cp = cvpartition(y,'k',10); %10-folds
    %prediction function
    classF = @(XTRAIN,ytrain,XTEST)(predict(TreeBagger(100,XTRAIN,ytrain),XTEST));
    %missclassification error 
    missclasfError = crossval('mcr',X,y,'predfun',classF,'partition',cp);
    
%%

Acc = (1-missclasfError)*100 