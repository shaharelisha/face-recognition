%% Load all images in the Faces folder
faceDatabase = imageSet('Faces','recursive');

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2], 'random');

%% Returns a bag of features object for the training set
bag = bagOfFeatures(training);

%% Extract Bag-of-Words Features for Training Set

training_sets = numel(training);    % number of categories (37)
training_set_size = sum([training.Count]);       % total number of training images

trainingFeatures = encode(bag,training).';      % feature matrix of visual words for the training set
trainingLabels = zeros(training_sets, training_set_size);   % zeros matrix for labels
featureCount = 1;

% loop through each image, and populates labels matrix by setting a 1 at 
% the index of the label number
for i=1:training_sets
    for j = 1:training(i).Count
        trainingLabels(i, featureCount) = 1;
        featureCount = featureCount + 1;
    end
end

%% Set up and train feedforward neural network

net = feedforwardnet(15, 'trainscg');
net = configure(net,trainingFeatures,trainingLabels);
net = train(net,trainingFeatures, trainingLabels);

%% Extract HOG Features for Test Set

testSets = numel(test);         % number of categories (37)
testSetSize = sum([test.Count]);        % total number of test images

testLabelsMatrix = zeros(testSets, testSetSize);      % zeros matrix for labels
testFeatureCount = 1;

% creates feature vector that represents a histogram of visual word 
% occurrences from the test set
testFeatures = encode(bag,test).';

% loop through each image in the test set and record the labels
for i=1:testSets
    for j=1:test(i).Count
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