function generate_filters()

prompt = 'Folder: ';
fold = input(prompt,'s');
folder = strcat('../',fold)

%file=['./texturefilters/ICAtextureFilters_11x11_8bit'];
%load(file, 'ICAtextureFilters');

cd(folder);
myFiles = dir('S*');

mkdir ../results3/;

column = importdata('../results2/gender_column.dat',',',0);

cd ../../bsif/texturefilters/;
filters = dir('*.mat');

cd ../;

for i = 1:length(filters)
    
    filter_name = filters(i).name;
    fullFilter = fullfile('./texturefilters/', filter_name);
    fprintf(1, 'Filter %s\n', fullFilter);
    
    [filepath,name,ext] = fileparts(filter_name);
    
    part = strsplit(fold,'/');
    path = char(strcat('../',part(1:1),'/results3/',name,'.csv'));
    
    load(fullFilter, 'ICAtextureFilters');
    
    matrix = [];

    for k = 1:length(myFiles)
        FileName = myFiles(k).name;
        fullFileName = fullfile(folder, FileName);
        %fprintf(1, 'Now reading %s\n', fullFileName);
  
        img=double(rgb2gray(imread(fullFileName)));
        %img = imresize(img, [240 320]);
        img = imresize(img, [180 240]);
        %imshow(img)
        %k = waitforbuttonpress
        bsifhistnorm=bsif(img, ICAtextureFilters,'nh');
        %figure('Name',FileName);
        %bar(bsifhistnorm);
        %k = waitforbuttonpress
        %close
        
        matrix = vertcat(matrix, bsifhistnorm);
    end

    matrix = [matrix column];
    size(matrix)
    dlmwrite(path,matrix)
end