clc
clear
%% data preprocessing
load TEdata.mat; IDV = 2; 
X_train = data(:, [1:22,42:52], 22); Y_train = data(:, 35, 22);
X_test = data(:, [1:22,42:52], IDV); Y_test = data(:, 35, IDV);

[~, m] = size(X_train); [n, p] = size(Y_train);
[X_train, Xmean, Xstd] = zscore(X_train); [Y_train, Ymean, Ystd] = zscore(Y_train);
[N, ~] = size(X_test);
X_test = (X_test - repmat(Xmean, N, 1))./repmat(Xstd, N, 1); Y_test = (Y_test - repmat(Ymean, N, 1))./repmat(Ystd, N, 1);

%% offline training
% kpls
options.KernelType = 'Gaussian'; options.t = sqrt(5000/2);
% fold = 5; indices = crossvalind('Kfold', n, fold); RMSE = zeros(m,1);
% for i = 1:m
%    for j = 1:fold
%       test = (indices == j); train = ~test;
%       [t, u, Kc, K, q] = kpls(X_train(train,:), Y_train(train,:), i, options);
%       Y_pre = Kc * u * q';
%       RMSE(i) = RMSE(i) +  mse(Y_pre, Y_train(test,:)) / fold;
%    end
% end
% pc = find(RMSE==min(RMSE));
pc = 5; % base on the above crossvalidation
[t, u, Kc, K, ~] = kpls(X_train, Y_train, pc, options);

M = u(:,pc) / (t(:,pc)' * Kc * u(:,pc)) * t(:,pc)' * Y_train;
[Um, Sm, Vm] = svd(M * M');

s = ones(n,1); I = eye(n);
tempy = (Um(:,1)' * Kc' * Kc * Um(:,1)) / (n-1);
temp = (Um(:,2:end)' * Kc' * Kc * Um(:,2:end)) / (n-1);
tempx = pinv(temp);

Ty2 = zeros(n, 1); Tx2 = zeros(n, 1);
for i = 1:n
     Ktrain = constructKernel(X_train(i,:),X_train,options);
     Kp = (Ktrain - s' * K / n) * (I - s * s' / n); 
     Ty2(i) = Kp * Um(:,1) / tempy * Um(:,1)' * Kp';
     Tx2(i) = Kp * Um(:,2:end) * tempx * Um(:,2:end)' * Kp';
end

% control limit
ALPHA = 0.99;
Ty_ctrl = (n-1) * (n+1) * finv(ALPHA, 1, n-1) / (n * (n-1));
miu = mean(Tx2); S = var(Tx2); g = S / (2 * miu); h = 2 * miu * miu / S;
Tx_ctrl = g * chi2inv(ALPHA, h);

%% online testing
Ty2 = zeros(N, 1); Tx2 = zeros(N, 1);
for i = 1:N 
     Ktest = constructKernel(X_test(i,:), X_train, options);
     Kp = (Ktest - s' * K / n) * (I - s * s' / n); 
     Ty2(i) = Kp * Um(:,1) / tempy * Um(:,1)' * Kp';
     Tx2(i) = Kp * Um(:,2:end) * tempx * Um(:,2:end)' * Kp';
end

% type I and type II errors
FAR_Ty = 0; FDR_Ty = 0;
FAR_Tx = 0; FDR_Tx = 0;
for i = 1:160
    if Ty2(i) > Ty_ctrl
       FAR_Ty = FAR_Ty + 1;
    end                     
    if Tx2(i) > Tx_ctrl
       FAR_Tx = FAR_Tx + 1;
    end                     
end
for i = 161:960
    if Ty2(i) > Ty_ctrl
       FAR_Ty = FAR_Ty + 1;
    end                     
    if Tx2(i) > Tx_ctrl
       FAR_Tx = FAR_Tx + 1;
    end                   
end
FAR_Ty = FAR_Ty / 160; FAR_Tx = FAR_Tx / 160;
FDR_Ty = FDR_Ty / 800; FDR_Tx = FDR_Tx / 800;

% ROC curves including f1-score
class_1 = Ty2(1:160); class_2 = Ty2(161:960);
figure; roc_Ty = roc_curve(class_1, class_2);

class_1 = Tx2(1:160); class_2 = Tx2(161:960);
figure; roc_To = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,1,1);plot(Ty2,'k');title('MKPLS');hold on;plot(Ty_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('T^2');legend('statistics','threshold');hold off;
subplot(2,1,2);plot(Tx2,'k');title('MKPLS');hold on;plot(Tx_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('Q');legend('statistics','threshold');hold off;
