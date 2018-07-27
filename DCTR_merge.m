
path = 'E:\research_MS_code\DCTR_feature\DCTR_matlab_v1.0\DCTR_matlab_v1.1';

t1 = csvread([path,'\','Dresden_DCTR.csv']);
t2 = csvread([path,'\','Dresden_DCTR_1413_2500.csv']);
t3 = csvread([path,'\','Dresden_DCTR_2500_3500.csv']);
t4 = csvread([path,'\','Dresden_DCTR_3500_4072.csv']);
t5 = csvread([path,'\','Dresden_DCTR_4073_4300.csv']);
t6 = csvread([path,'\','Dresden_DCTR_4301_4500.csv']);
t7 = csvread([path,'\','Dresden_DCTR_4500_5021.csv']);

t8 = t1+t2+t3+t4+t5+t6+t7;

filename = 'Dresden_DCTR_1_5021.csv';
xlswrite(filename,t8);