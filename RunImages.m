% script which runs through all images and extracts faces and labels
folders = dir('Images/Individual*');
for i =1: size(folders)
    f = folders(i);
    path = sprintf('Images/%s/IMG_*.JPG', f.name);
    files = dir(path);
    for j=1 : size(files)
        file = files(j);
        I = imread(sprintf('Images/%s/%s', f.name, file.name));
        SortImages(I, file.name);

    end
end
