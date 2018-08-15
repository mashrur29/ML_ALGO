function [J] = costFunction(X, y, theta, lambda)
	m = size(X, 1);
	J = 0;
	predictions = X * theta;
	sqrError = (predictions - y) .^ 2;
	J = (1 / (2 * m)) * sum(sqrError);
	thetaTemp = theta;
	thetaTemp(1) = 0;
	J = J + ((lambda / (2 * m)) * sum(thetaTemp .* thetaTemp));
end;