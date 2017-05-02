function [ P ] = RecogniseFace(I, featureType, classifierName, creativeMode)
% Given image I, a featureType, a classifierName, and the option to
% activate the creative mode, the function will output a matrix where each
% row represents the ID, x location (center), y location (center), and 
% emotion of the person identified in an image. The featureType and
% classifierName will determine how to predict the identity of the face,
% and if creativeMode is selected, an image will show. IMAGE MIGHT NEED TO
% BE ROTATED IN ADVANCE.

FaceDetector = vision.CascadeObjectDetector('MergeThreshold', 8);

% steps through image, detecting faces
BBOX = step(FaceDetector,I);
[r, c] = size(BBOX);
P = zeros(r, 4);

% load emotionClassifier
load('EmotionClassifier.mat');
% If HOG + SVM
if strcmpi(featureType, 'HOG') && strcmpi(classifierName, 'SVM')
    % load compacted classifier
    load('CompactHogSvmClassifier.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [100 100]);
        queryFeatures = extractHOGFeatures(face);
        label = predict(compactFaceClassifier, queryFeatures);
        label = str2num(label{1});
        
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P(x, :) = [label mid_x mid_y emotion];
    end
    
% if HOG + MLP
elseif strcmpi(featureType, 'HOG') && strcmpi(classifierName, 'MLP')
    % load classifier
    load('HogFnnClassifier.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [100 100]);
        queryFeatures = extractHOGFeatures(face);
        queryFeaturesT = queryFeatures';
        outPuts = net(queryFeaturesT);
        [value label] = max(outPuts(:,1));
        
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P(x, :) = [label, mid_x, mid_y, emotion]; %center point, %center point]
    end
% if Bag + SVM
elseif strcmpi(featureType, 'Bag') && strcmpi(classifierName, 'SVM')
    % load classifier
    load('BagSvmClassifier.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        
        face = imresize(face, [100 100]);
        label = predict(categoryClassifier, face);
        
        queryFeatures = extractHOGFeatures(face);
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P(x, :) = [label, mid_x, mid_y, emotion]; %center point, %center point]
    end
% if Bag + MLP
elseif strcmpi(featureType, 'Bag') && strcmpi(classifierName, 'MLP')
    % load classifier
    load('BagFnnClassifier.mat');
    % step through each face, and populate matrix P
    for x = 1:r
        face = I(BBOX(x,2):BBOX(x,2)+BBOX(x,4), BBOX(x,1):BBOX(x,1)+BBOX(x,3), :);
        face = imresize(face, [100 100]);
        queryFeatures = encode(bag,face).';
        outPuts = net(queryFeatures);
        [value label] = max(outPuts(:,1));
        
        queryFeatures = extractHOGFeatures(face);
        emotion = predict(emotionClassifier, queryFeatures);
        emotion = str2num(emotion{1});
        
        mid_point = BBOX(x,4)/2;
        mid_x = BBOX(x,1) + mid_point;
        mid_y = BBOX(x,2) + mid_point;
        P(x, :) = [label, mid_x, mid_y, emotion]; %center point, %center point]
    end
else
    % if the function is called without matching feature/classifier names,
    % the following error will show
    error('Please select from feature options ["HOG", "Bag"] and classifier options ["SVM", "MLP"])')
end

if creativeMode == 1
    for x = 1:r
        % expanded box by 10 pixels in each direction, because it usually
        % cuts the chin and forehead, and sometimes the sides slightly
        face = I(BBOX(x,2)-10:BBOX(x,2)+BBOX(x,4)+10, BBOX(x,1)-10:BBOX(x,1)+BBOX(x,3)+10, :);
        colorfulFace = ColorFaces(face);
        I(BBOX(x,2)-10:BBOX(x,2)+BBOX(x,4)+10, BBOX(x,1)-10:BBOX(x,1)+BBOX(x,3)+10, :) = colorfulFace;
        clear face colorfulFace;
    end
    imshow(I);
end

