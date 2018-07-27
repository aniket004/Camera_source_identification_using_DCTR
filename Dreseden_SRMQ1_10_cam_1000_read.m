
%% Reading 100 images per cam (10 camera) and their SRMQ1 feature vector
data = csvread('Dresden_SRMQ1_start_stop.csv');

data_1_1000 = data(1:1000,:);

filename = 'Dresden_SRMQ1_10_cam_1000.csv';
csvwrite(filename,data_1_1000);