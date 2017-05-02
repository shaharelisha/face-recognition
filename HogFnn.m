%% Load all images in the Faces folder
faceDatabase = imageSet('Faces','recursive');

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2], 'random');

%% Extract HOG Features for Training Set
trainingSets = numel(training);        % number of categories (37)
trainingSetSize = sum([training.Count]);      % total number of training images

trainingLabels = zeros(trainingSets, trainingSetSize);      % zeros matrix for labels
featureCount = 1;

% loop through each image, extract and add HOG Features to trainingFeatures
% matrix, and populate labels matrix by setting a 1 at the index of the
% label number
for i=1:trainingSets
    for j = 1:training(i).Count
        trainingFeatures(:, featureCount) = extractHOGFeatures(read(training(i),j));
        trainingLabels(i, featureCount) = 1;
        featureCount = featureCount + 1;
    end
end

%% Set up and train feedforward neural network

net = feedforwardnet(15, 'trainscg');
net = configure(net,trainingFeatures,trainingLabels);
net = train(net,trainingFeatures, trainingLabels, 'useParallel','yes');


%% Extract HOG Features for Test Set

testSets = numel(test);         % number of categories (37)
testSetSize = sum([test.Count]);        % total number of test images

testLabelsMatrix = zeros(testSets, testSetSize);      % zeros matrix for labels
testFeatureCount = 1;

% loop through each image in the test set, extract the HOG Features and the
% labels
for i=1:testSets
    for j=1:test(i).Count
        testFeatures(:, testFeatureCount) = extractHOGFeatures(read(test(i),j));
        testLabelsMatrix(i, testFeatureCount) = 1;
        testFeatureCount = testFeatureCount + 1;
    end
end

%% Predict matching labels for all images in the test set
testOutputs = net(testFeatures);

% loop through the output from the network - the closest match is the index
% where the maximum value is per column. At the same time, get the actual
% labels for the test dataset.
for i = 1 : testSetSize
    [value testLabels(1,i)] = max(testOutputs(:,i));
    actualTestLabels(i) = find(testLabelsMatrix(:,i));
end

%% Calculate accuracy of test imageset
accuracy = sum(testLabels == actualTestLabels) / testSetSize;
