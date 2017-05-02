function [] = RunVideoLabel(filename)
% Iterates through the frames of each video and extracts face and label
vidReadObj = VideoReader(strcat('Finals/Movies/IndividualMovies6/', filename, '.mov'));

counter = 0;
while hasFrame(vidReadObj)
    I = readFrame(vidReadObj);    
    counter = counter + 1;
    file_name = sprintf('%s_%d.JPG', filename, counter);
    try
        % unlike images, video wasn't rotated
        I = imrotate(I, 90);
        SortImages(I, file_name);
    catch
        continue
    end
    
end