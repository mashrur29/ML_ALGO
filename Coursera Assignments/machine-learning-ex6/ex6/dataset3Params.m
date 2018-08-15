function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 10;

arr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 90; 270; 0.02; 0.04; 0.08; 1.6; 3.2; 20; 40; 80; 160; 320];
res = 10000000000000;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



for i = 1: length(arr),
	for j = i+1: length(arr),
		model = svmTrain(X, y, arr(i), @(x1, x2) gaussianKernel(x1, x2, arr(j)), 1e-3, 20);
		predictions = svmPredict(model, Xval);
		mean = (double(predictions ~= yval));
		if mean <= res,
			res = mean;
			C = arr(i);
			sigma = arr(j);
			% fprintf('lol -> %d %d\n', C, sigma);
		end;
	end;
end;

% fprintf('Finally lol -> %d %d\n', C, sigma);

% =========================================================================

end
