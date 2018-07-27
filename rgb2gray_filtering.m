function[] = rgb2gray_filtering()
% function for gaussian filtering 

fprintf('Provide location of source images\n');

dr_l=uigetdir();

list_left = dir([dr_l, '/*.JPG']);        %here jpg is file extension of the images

fprintf('Provide location of filtered images\n');
 
dr_r=uigetdir();


%list_left contains original images

for x=1:length(list_left)
    
    x
    img_left{x}=imread([dr_l, '/', list_left(x).name]);
    img_right{x} = rgb2gray(img_left{x});
    
    list_right = [dr_r, '/', list_left(x).name ];
    imwrite(img_right{x},list_right);
    clear img_left{x} img_right{x};
end
end