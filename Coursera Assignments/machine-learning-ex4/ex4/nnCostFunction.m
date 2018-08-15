function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Xtemp = [ones(m, 1) X];
Ytemp = zeros(m, num_labels);


for i = 1: m,
	Ytemp(i, y(i)) = 1;
end;

a1 = Xtemp;

pre_z2 = Xtemp * Theta1';


z2 = sigmoid(Xtemp * Theta1');
z2 = [ones(m, 1) z2];
pre_z3 = z2 * Theta2';
z3 = sigmoid(z2 * Theta2');

temp = 0;

for i = 1: m,
	for k = 1: num_labels,
		temp += Ytemp(i, k) * log(z3(i, k)) + (1 - Ytemp(i, k)) * log(1 - z3(i, k));
	end;
end;

temp1 = 0;

for i = 1: size(Theta1, 1),
	for j = 2: size(Theta1, 2),
		temp1 += (Theta1(i, j) * Theta1(i, j));
	end;
end;

for i = 1: size(Theta2, 1),
	for j = 2: size(Theta2, 2),
		temp1 += (Theta2(i, j) * Theta2(i, j));
	end;
end; 

J = -(1/m) * temp + ((lambda / (2 * m)) * temp1);


delta_3 = z3 - Ytemp;

delta_2 = delta_3 * Theta2;

delta_2 = delta_2 .* (z2 .* (1 .- z2));



del_1 = a1' * delta_2(:, [2: end]);
del_1 = del_1';

del_2 = z2' * delta_3;
del_2 = del_2';

for i = 1: size(Theta1, 1),
	for j = 1: 1,
		Theta1(i, j) = 0;
	end;
end;

for i = 1: size(Theta2, 1),
	for j = 1: 1,
		Theta2(i, j) = 0;
	end;
end; 

Theta1_grad = (del_1 ./ m) + ( (lambda / m) .* Theta1 );
Theta2_grad = (del_2 ./ m) + ( (lambda / m) .* Theta2 );



% -------------------------------------------------------------

% =========================================================================
	
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
