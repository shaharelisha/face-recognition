faceDatabase = imageSet('Faces','recursive');

image_sets = size(faceDatabase,2);
for i=1:image_sets
    for j = 1:faceDatabase(i).Count
        A = read(faceDatabase(i),j);
        
        % if the size of the image isn't already 100X100, resize to 100X100
        % this assumes that all photos are square.
        if (size(A, 1) ~= 100)
            location = faceDatabase(i).ImageLocation(j);
            A = imresize(A, [100 100]);
            imwrite(A, location{1,1});
        end
    end
end