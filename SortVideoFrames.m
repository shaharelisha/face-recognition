function [  ] = SortVideoFrames(I , image_name)
% Detects and extracts faces from frame I. If a label is detected
% then the extracted faces will be saved into a folder of that label,
% while keeping its original file name. 
    FaceDetector = vision.CascadeObjectDetector('MergeThreshold', 8);
    
    % Detect and locate faces in image I
    BBOX = step(FaceDetector,I);
    [r, c] = size(BBOX);
    
    % if at least one face is detected
    if r > 0
        % resize image I and change to grayscale
        A = imresize(I,	0.33);
        A = rgb2gray(A);

        % binarize image A and get the negative
        BW = imbinarize(A, 'adaptive', 'Sensitivity', 0.7);
        WB = imcomplement(BW);

        stats = regionprops(WB, 'Area', 'Solidity');
        CC = bwconncomp(WB);
        L = labelmatrix(CC);
        % Limit the white regions to ones which have an area between 20 and 
        % 2000, and a solidity of under 1
        BW2 = ismember(L, find([stats.Area] <= 2000 & [stats.Area] > 20 & [stats.Solidity] < 1));

        % Additional narrowing down using blob analysis
        blobAnalyzer = vision.BlobAnalysis('MaximumCount', 10);
        
        % locate any remaining blobs in image BW2
        [area, centroids, roi] = step(blobAnalyzer, BW2);
        
        % parameters taken from regions of interest (roi) - aka each blob
        % detected
        width  = roi(:,3);
        height = roi(:,4);
        x_coor = roi(:,1);
        y_coor = roi(:,2);
        img_size = size(BW2);
        y_total = img_size(1);
        x_total = img_size(2);
        aspectRatio = width ./ height;
        
        % additional constraints mean that the blobs detected must have a width
        % to height ratio between 0.35 and 1.5, a width between 5 and 40, a
        % height between 10 and 50, and a location within the middle 70% of the
        % x-axis.
        ratioConstraint = aspectRatio < 1.5 & aspectRatio > 0.35;
        shapeConstraint = width < 40 & width > 5 & height < 50 & height > 10;
        positionConstraint = x_coor < (x_total - (x_total * 0.15)) & x_coor > (x_total * 0.15);
        totalConstraint = ratioConstraint & shapeConstraint & positionConstraint;
        % roi2 represents all the remaining blobs
        roi2 = double(roi(totalConstraint, :));
        
        % number of blobs found
        blobs_found = size(roi2);
        blobs_found = blobs_found(1);
        if blobs_found >= 2
            if blobs_found == 2
                % if exactly 2 blobs were found, try running FindLabel, and
                % hopefully it detects the text
                [label_found, label] = FindLabel(1,2, WB, roi2)
            else
                % if there are more than 2 blobs detects, the distance between
                % two blobs should be under 70 in order to qualify. Loop 
                % through all unique pairs of blobs, and if an adjacent pair is
                % found, run FindLabel and hopefully it detects the text.
                rows = size(roi2, 1);
                combos = nchoosek(1:rows, 2);
                for i=1:size(combos,1)
                    j = combos(i,1);
                    k = combos(i,2);
                    distance = abs(roi2(j, 1) - roi2(k,1));
                    if distance < 70
                        [label_found, label] = FindLabel(j,k, WB, roi2)
                        if label_found
                            % if a label is found before all pairs are checked,
                            % break the loops - there's no point in checking
                            % the others
                            break
                        end
                    end
                end
            end
            if label_found
                % if a label is found, extract the face(s) detected from the
                % original image and save it within folder
                % 'Faces/[label-name]/[original-image-name]'
                for x = 1:r
                    FF = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
                    FF = imresize(FF, [100 100]);
                    imwrite(FF, sprintf('SortedVideos/%s/%s', label, image_name));
                    clear FF;
                end
            else
                % no label found
                for x = 1:r
                    FF = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
                    imwrite(FF, sprintf('SortedVideos/Not Found/%s', image_name));
                    clear FF;
                end
            end
        else
        % If less than 2 blobs were found, extract the face(s) detected from 
        % the original image, and then place them within the
        % 'Not Found' folder within 'Faces'.
            for x = 1:r
                FF = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
                imwrite(FF, sprintf('SortedVideos/Not Found/%s', image_name));
                clear FF;
            end
        end
    end
end

