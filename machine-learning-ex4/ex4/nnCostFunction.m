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

%ref: https://github.com/anirudhjayaraman/Machine-Learning/blob/master/Andrew%20Ng%20Stanford%20Coursera/Week%2005/ex4/nnCostFunction.m

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

%y_matrix = zeros(m, num_labels);
%for i = 1:m
%    y_matrix(i, y(i)) = 1;
%end;
%Feedforward and cost function
y_eye = eye(num_labels);
y_matrix = y_eye(y, :);
%fprintf("y matrix:\n");
%disp(y_matrix);

a1 = [ones(m, 1) X]; % 5000 X 401
z2 = a1 * Theta1'; %5000 X 25
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2]; %5000 X 26
z3 = a2 * Theta2';  %5000 X 10
a3 = sigmoid(z3);
h_theta_x = a3;  %5000 X 10
%fprintf("h_theta_x's size:(%d %d)", size(h_theta_x,1), size(h_theta_x, 2));
%fprintf("y's size(%d %d), y_matrix's size(%d, %d)",size(y, 1), size(y, 2), size(y_matrix, 1), size(y_matrix, 2));



J = 1 / m * sum(sum(log(h_theta_x) .* (-y_matrix) - log(1 - h_theta_x) .* (1 - y_matrix)));
%Regularized cost function

%fprintf("Theta1's size(%d %d)\n", size(Theta1, 1), size(Theta1, 2));
%fprintf("Theta2's size(%d %d)\n", size(Theta2, 1), size(Theta2, 2));
regularization_part = lambda / (2 * m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularization_part;

%Backpropagation
DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));
% test data in file checkNNGradients.m
% Matrix implementation
%{
d3 = a3 - y_matrix;                                             % has same dimensions as a3
d2 = (d3*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];     % has same dimensions as a2

D1 = d2(:,2:end)' * a1;    % has same dimensions as Theta1
D2 = d3' * a2;    % has same dimensions as Theta2

Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;
%}

% Vector implementation
for t = 1:m
    a1t = [1; X(t, :)']; %4 X 1
    z2t = Theta1 * a1t; %5 X 1
    a2t = sigmoid(z2t); %5 X 1
    a2t = [1; a2t]; % 6 X 1
    z3t = Theta2 * a2t; %3 X 1
    a3t = sigmoid(z3t); %3 X 1
    delta_3 = a3t - y_eye(:,y(t)); % 3 x 1
    % Theta2': 6 X 3     3 X 1   
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1;z2t]); % 6 X 1 including
    % 3 X 6    3 X 6      3 X 1   1 X 6  
    DELTA_2 = DELTA_2 + delta_3 * a2t'; % 3 X 6    
    % 5 X 4    5 X 4      5 X 1          1 X 4   
    DELTA_1 = DELTA_1 + delta_2(2:end) * a1t'; % 5 X 4
end;

Theta1_grad = 1/m * DELTA_1;
Theta2_grad = 1/m * DELTA_2;

% Regularized neural networks
reg_Theta1 = [zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
reg_Theta2 = [zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
Theta1_grad = Theta1_grad + reg_Theta1 * lambda / m;
Theta2_grad = Theta2_grad + reg_Theta2 * lambda / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%fprintf("grad's size:%d. grad:\n", size(grad, 1));
%disp(grad);
end
