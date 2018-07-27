%% Reading data
data = csvread('E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1\Mixed_100_per_cam_DCTR.csv');
feature = data(:,1:end-1);
label = data(:,end);

%% Dimensionality reduction by PCA

% Number of iteration selection by 95% of variance retained
[row col] = size(feature);
%sigma = (feature'*feature);
sigma = cov(feature);
[u s v ] = svd(sigma);
[m,m] = size(s);

itr = 1;
s_add = s(1,1);
while( s_add/sum(sum(s)) <= 0.95  )
    itr = itr+1;
    s_add = s_add + s(itr,itr);
end
    

itr = min(itr,m);
% PCA analysis using optimum value i    

[eigenvectors, projected_data, eigenvalues] = princomp(feature);
[foo, feature_idx] = sort(eigenvalues, 'descend');
selected_projected_data = projected_data(:, feature_idx(1:itr));

feature = selected_projected_data;
%%

    X = feature;
    y = label;
    %data partition
    cp = cvpartition(y,'k',10); %10-folds
    %prediction function
    classF = @(XTRAIN,ytrain,XTEST)(predict(TreeBagger(500,XTRAIN,ytrain),XTEST));
    %missclassification error 
    missclasfError = crossval('mcr',X,y,'predfun',classF,'partition',cp);
    
%%

Acc = (1-missclasfError)*100 