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
    %data partition
    
    labeled_feature = horzcat(feature,label);
    
filename = 'PCA_reduced_dimension';
xlswrite(filename,labeled_feature);
