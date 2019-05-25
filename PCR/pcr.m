function [T, P, Q] = pcr(X, Y, k)
[P, T, ~, ~] = pca(X); 
T = T(:, 1:k); P = P(:, 1:k);
Q = ((T' * T) \ (T') * Y)'; 
end