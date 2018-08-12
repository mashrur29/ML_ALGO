% creating a matrix

	A = [1 2; 2 3; 3 4]

% vector

	v = [1; 2; 3]

% creates a vector from 1 -> 2 with diff = 0.1

	v = 1:0.1:2

% creates v from 1 -> 6 with diff = 1

	v = 1:6

% 2 x 3 one matrix

	ones(2, 3)

% 2 x 3 zero matrix

	zeros(2, 3)

% random 3 x 3 matrix

	rand(3, 3)

% random 3 x 3 from gaussian distribution

	randn(3, 3)

% creates histrogram

	w = -6 + sqrt(10)*(randn(1, 10000))
	hist(w)
	hist(w, 50)

% 6x6 identity matrix

	eye(6)

% help

	help eye % (or anything else)

% size of matrix

	size(A)
	size(A, 1)
	size(A, 2)

% length

	length(v) % for matrix it returns the max size

% current path of octave

	pwd

% load file

	load sampleX.dat
	load('sampleX.dat')

% current variables in session

	who

% current variables with details

	whos

% delete a variable

	clear sampleX

% v is the first two elements of resultY

	v = resultY(1:2)

% save v in a file called hello.matrix

	save hello.mat v

% save as ascii

	save hello.txt v -ascii

% everything along second row

	A(2,:)

% everything along first column

	A(:,1)

% everything from 1st and 3rd row

	A([1, 3], :)

% assigning 2nd column

	A(:,2) = [1;2;3]

% append a column

	A = [A, [4; 5; 6]]

% append a row

	A = [A; [4 5 6]]

% turn matrix to vector

	A(:)

% append two matrix

	A = [A B]

% matrix operations

	A*C % multiplication
	A .* B % element wise multiplication
	A .^ B % element wise power
	1 ./ A % element wise inverse
	log(v) % element wise log
	exp(v) % element wise base e exp
	-v
	abs(v)
	A' % transpose
	max(A) % column wise max
	A < 3 % element wise comparison
	find(A<3) % returns indices by treating A as vector
	[r,c] = find(A<=3) % row + col index
	sum(v) % sum of vector
	prod(v) % product
	ceil(v)
	floor(v)
	max(rand(3), rand(3)) % element wise max of two random 3x3
	max(A, [], 1) % column wise max
	max(A, [], 2) % row wise max
	max(max(A)) % max in entire matrix
	sum(A, 1) % column wise sum
	sum(A, 2) % row wise
	flipud(A) % flips a matrix
	pinv(A) % pseudo inverse

% returns matrix whose sum along row/col/diagonal same

	magic(3)

% Don't close figure while making plots

% making plots

	t = [0: 0.01: 0.98]
	y1 = sin(2*pi*4*t)
	plot(t, y1) % along x-axis t and along y-axis sin()

% make one plot on another

	plot(t, y1);
	hold on;
	plot(t, y2, 'r'); % 'r' means color it with red

% labelling the axes

	xlabel('time')
	ylabel('value')

% adding legends

	legend('sin', 'cos')

% adding title

	title('my plot')

% save plot, make sure about the directory

	print -dpng 'plot.png' % type help plot for other formats

% naming plots

	figure(1); plot(t, y1);

% divides plot into 1x2 grid

	subplot(1, 2, 1); % accessing 1st element
	plot(t, y1) % the first element
	subplot(1, 2, 2); % accessing 2nd element
	plot(t, y2) % the second element

% changing scale

	axis([0.5 1 -1 1])

% clear figure

	clf;

% visualize a matrix

	A = magic(9)
	imagesc(A)
	imagesc(A), colorbar, colormap gray; % uses grayscale image with colorbar

							% control statements %

% for loop

for i = 1: 10,  % loop through 1 -> 10
	v(i) = 2^i;
end;

% Alternate for loop

indices = 1: 10
for i = indices,
	disp(i);
end;

% while loop

i =  1
while i <= 5,
	v(i) = 100;
	i += 1;
end;

% using if

i = 1;
while true,
	v(i) = 999;
	i += 1;
	if i==6,
		break;
	end;
end;

% else if

if v(1) == 1,
	disp('The value is 1');
elseif v(1) == 2,
	disp('The value is 2');
else
	disp(v(1));
end;

			% functions

% Basic: create file with function name

% in file squareThisNumber.m
function y = squareThisNumber(x)
	y = x ^ 2;
% from terminal
squareThisNumber(5)

% adding octave search path

addpath('C:\Users\Asus\Desktop')

% function with two return types
function [y1, y2] = squareAndCube(x1)
	y1 = x1 ^ 3;
	y2 = x1 ^ 2;

[a, b] = squareAndCube(5)

% cost function
X = [1, 1; 1, 2; 1, 3];
Y = [1; 2; 3];
theta = [0; 1];

function J = costFunction(X, y, theta)
	predictions = X*theta;
	sqrError = (predictions - y) .^ 2;
	m = size(X, 1);
	J = (1 / (2 * m)) * sum(sqrError);