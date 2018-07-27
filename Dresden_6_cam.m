%%% Feature extraction with DCTR feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;

dr=uigetdir();

list=dir([dr, '/*.JPG']);        %here jpg is file extension of the images

for x=1:length(list)

    % classify labels of image according to available camera model
    
        if( strmatch('Agfa',list(x).name) == 1 )
        small_label(x) = 4; 
    elseif( strmatch('Casio',list(x).name) == 1 )
        small_label(x) = 5;
    elseif( strmatch('FujiFilm',list(x).name) == 1 )
        small_label(x) = 6;
        elseif( strmatch('Kodak',list(x).name) == 1 )
        small_label(x) = 7;
    elseif( strmatch('Nikon',list(x).name) == 1 )
        small_label(x) = 8;       
    elseif( strmatch('Panasonic',list(x).name) == 1 )
        small_label(x) = 9;        
        end   

end


% Specify JPEG image for feature extraction and its quality factor
quality_factor = 75;

    F_small = zeros(length(list),8001);

for x=1:length(list)
    
    x
    
    imagePath = [dr, '/', list(x).name];
    
    I_STRUCT = jpeg_read(imagePath);
    F_small(x,1:8000) = DCTR(I_STRUCT,quality_factor);  
    F_small(x,8001) = small_label(x);
    
end


filename = 'Dresden_small_DCTR.xls';
xlswrite(filename,F_small);
