function [new_image] = ColorFaces(I)
    % given an image I, with the combination of skin segmentation and
    % k-means clustering, a new image is returned where the skin in the
    % image is transformed into a Warhol-inspired painting.
    
    % Al-Tairi et al. - Code from lecture
    YUV = rgb2ycbcr(I);
    U = YUV(:, :, 2); V = YUV(:, :, 3);
    R = I(:, :, 1); G = I(:, :, 2); B = I(:, :, 3);
    [rows, cols, planes] = size(I);

    skin = zeros(rows, cols);
    ind = find(80 < U & U < 130 & 136 < V & ...
    V <= 200 & V > U & R > 80 & G > 30 & ...
    B > 15 & abs(R-G) > 15);
    skin(ind) = 1;

    % Run k means on the RGB data, k = 10
    R = double(reshape(I, rows*cols, 3));
    [clusterID, clusterCentre] = kmeans(R, 10);
    % Reshape back to an image
    clusterID = reshape(clusterID, rows, cols);

    % replace clusters with new random colors
    three = zeros(rows, cols, 3);

    for i=1:10
        [r,c] = find(clusterID==i);
        colors = [rand rand rand];
        for j=1:numel(r)
            three(r(j),c(j),:) = colors;
        end
    end

    % only take the colorful section of the image in the indices where
    % there is skin-color, otherwise the image pixels remain as the
    % original
    new_image = I;
    [r,c] = find(skin==1);
    for j=1:numel(r)
        new_image(r(j),c(j),:) = uint8(three(r(j),c(j),:) .* 255);
    end
