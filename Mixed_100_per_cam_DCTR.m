
%%%%% Reading 100 images from each class
%% Reading data
data = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Dresden_DCTR_1_10507.csv');

class_0 = data(1:100,:);
class_1 = data(980:1079,:);
class_2 = data(2020:2119,:);
class_3 = data(3351:3450,:);
class_4 = data(5022:5121,:);
class_5 = data(6022:6121,:);
class_6 = data(6947:7046,:);
class_7 = data(7577:7676,:);
class_8 = data(8577:8676,:);
class_9 = data(9508:9607,:);


Mixed_100_per_cam_DCTR = vertcat(class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9);

filename = 'Mixed_100_per_cam_DCTR.csv';
csvwrite(filename,Mixed_100_per_cam_DCTR);

%%% RF classification
data = Mixed_100_per_cam_DCTR;

feature = data(:,1:end-1);
label = data(:,end);

%% Dimensionality reduction by PCA
[eigenvectors, projected_data, eigenvalues] = princomp(feature);
[foo, feature_idx] = sort(eigenvalues, 'descend');
selected_projected_data = projected_data(:, feature_idx(1:100));

feature = selected_projected_data;

filename = 'PCA_100_reduced_Mixed_100_cam_DCTR.csv';
csvwrite(filename,horzcat(feature,label));
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