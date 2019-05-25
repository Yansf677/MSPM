function [T, W, Q, Kc, K] = kpcr(X, Y, k)
[n,~] = size(X);
options.KernelType = 'Gaussian'; options.t = sqrt(5000/2);
K = constructKernel(X, [], options);
s = ones(n, 1); I = eye(n); Kc = (I - s * s' / n) * K * (I - s * s' / n); 
[W, L_W] = eig(Kc./n); W = W * (L_W^(-0.5)); T = Kc*W;
T = T(:, 1:k); W = W(:, 1:k);
Q = ((T' * T) \ T' * Y)';
end