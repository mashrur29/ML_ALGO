function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


m = size(X, 1);
predictions = X * theta;
sqrError = (predictions - y) .^ 2;

thetaTemp = theta;
thetaTemp(1) = 0;

J = (1 / (2 * m)) * sum(sqrError);
J += (lambda / (2 * m)) * sum(thetaTemp .* thetaTemp);


for j = 1: size(theta),
	temp = 0;
	for i = 1: m,
		temp += ((X(i, :) * theta) - y(i)) * X(i, j);
	end;
	temp /= m;
	
	if(j>1) 
		temp += (lambda * theta(j)) / m;
	end;
	
	grad(j) = temp;
end;




% =========================================================================

grad = grad(:);

end
