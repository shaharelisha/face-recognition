%% Load all images in the Faces folder
faceDatabase = imageSet('Faces','recursive');

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2], 'random');

%% Create bag of visual words
bag = bagOfFeatures(training); 

%% Train a classifier with the Training Set
categoryClassifier = trainImageCategoryClassifier(training, bag); 

%% Evaluate the classifier using the Test Set
confMatrix = evaluate(categoryClassifier, test); 

% Compute average accuracy
accuracy = mean(diag(confMatrix));

