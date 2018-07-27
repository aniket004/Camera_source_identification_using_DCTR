function arffFileName=csv2arff(csvFileName,separator)
% Converts the provided CSV file to ARFF file and returns the ARFF
% filename
% 
% csv2arff(csvFileName)
%
% csvFileName     = Name of the CSV file to convert. [ i.e. 'text.csv' ]
% separator    = sign separating the values (default = ',')
%         by Eftim Zdravevski, 2014

if ~exist('separator', 'var') || ~ischar(separator)
    separator = ',';
end

    arffFileName=strrep(csvFileName, '.csv', '.arff');
    data=csvread(csvFileName);
    col=size(data,2);
    fil=fopen(arffFileName, 'wt');
 
    fprintf (fil, '%s', '@relation ');
    fprintf (fil, '%s', '''');
    fprintf (fil, '%s', csvFileName);
    fprintf (fil, '%s', '''');
    fprintf (fil, '\n');
    fprintf (fil, '\n');
   
    classes=unique(data(:,size(data,2)));
    num_classes=length(classes);
    for r=1:col
        if r<col
            fprintf(fil, '%s', ['@attribute k',int2str(r), ' numeric']);
            fprintf(fil, '\n');
        else
            classes_string='';
            for cl=1:num_classes-1
                    classes_string=[classes_string num2str(classes(cl)) separator];
            end
            classes_string=[classes_string num2str(classes(num_classes))];
            fprintf(fil, ['@attribute class {' classes_string '}']);
            fprintf(fil, '\n');
        end
    end
    fprintf(fil, '\n');
    fprintf (fil, '%s', '@data');
    fprintf(fil, '\n');
    for i=1:size(data,1)
        for j=1:col
            if j==col
                fprintf(fil, '%d', data(i,j));
                fprintf(fil, '\n');
            else
                fprintf(fil, '%f ', data(i,j));     
            end
        end
    end
    fclose(fil);
end
        