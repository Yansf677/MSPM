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
% pls
fold = 5; indices = crossvalind('Kfold', n, fold); RMSE = zeros(m,1);
for i = 1:m
   for j = 1:fold
      test = (indices == j); train = ~test;
      [~, R, Q, ~] = pls(X_train(train,:), Y_train(train,:), i);
      Y_pre = X_train(test,:) * R * Q';
      RMSE(i) = RMSE(i) +  mse(Y_pre, Y_train(test,:)) / fold;
   end
end
pc = find(RMSE==min(RMSE));
[T, R, Q, P] = pls(X_train, Y_train, pc);   

Y_e = T * Q'; [Uc, Dc, Vc] = svd(Y_e, 'econ'); l = pc; lc = size(Dc, 1); 
Qc = Vc * Dc; Rc = R * Q' * Vc * pinv(Dc); Yc = Y_train - Uc * Qc';
[Py, Ty, Latenty, Tsquarey] = pca(Yc); 
ly = 0;
for i = 1:size(Latenty,1)
    cpvy = sum(Latenty(1:i)) / sum(Latenty);
    if cpvy >= 0.9
        ly = i; break;
    end
end
Ty = Ty(:, 1:ly); Py = Py(:, 1:ly); Yresidual = Yc - Ty * Py';

Rcinv = pinv(Rc' * Rc) * Rc'; Xc = X_train - Uc * Rcinv;
[Px, Tx, Latentx, Taquarex] = pca(Xc); 
lx = 0;
for i=1:size(Latentx,1)
    cpvx = sum(Latentx(1:i)) / sum(Latentx);
    if cpvx >= 0.9
        lx = i; break;
    end
end
Tx = Tx(:, 1:lx); Px = Px(:, 1:lx); Xresidual = Xc - Tx * Px';

Tc2 = zeros(n, 1); Tx2 = zeros(n, 1); Qx = zeros(n, 1);
for i = 1:n
   Tc2(i) = (n-1) * (X_train(i,:)) * Rc * Rc' * (X_train(i,:))';
   xc = (X_train(i,:))' - Rcinv' * Rc' * (X_train(i,:))';
   Tx2(i) = xc' * Px * pinv((Tx' * Tx)/(n-1)) * Px' * xc;
   Qx(i) = xc' * (eye(m) - Px * Px') * xc;
end

% control limit
ALPHA=0.97;
Tc_ctrl = lc * (n-1) * (n+1) * finv(ALPHA, lc, n - lc) / (n * (n - lc));
Tx_ctrl = lx * (n-1) * (n+1) * finv(ALPHA, lx, n - lx) / (n * (n - lx));
miu = mean(Qx); S = var(Qx); g = S / (2 * miu); h = 2 * miu * miu / S;
Qx_ctrl = g * chi2inv(ALPHA, h);
% theta=zeros(1,3);
% for i=1:3
%     for j=(lx+1):m
%         theta(i) = Latentx(j)^i + theta(i);
%     end
% end
% h0=1-2*theta(1)*theta(3)/(3*theta(2)^2);
% Qx_ctrl=theta(1)*(norminv(ALPHA)*(2*theta(2)*h0^2)^0.5/theta(1)+1+theta(2)*h0*(h0-1)/theta(1)^2)^(1/h0);

%% online testing
Tc2 = zeros(N, 1); Tx2 = zeros(N, 1); Qx = zeros(N, 1);
for i = 1:N
   Tc2(i) = (n-1) * (X_test(i,:)) * Rc * Rc' * (X_test(i,:))';
   xc = (X_test(i,:))' - Rcinv' * Rc' * (X_test(i,:))';
   Tx2(i) = xc' * Px * pinv((Tx' * Tx)/(n-1)) * Px' * xc;
   Qx(i) = xc' * (eye(m) - Px * Px') * xc;
end

% type I and type II errors
FAR_Tc = 0; FDR_Tc = 0;
FAR_Tx = 0; FDR_Tx = 0;
FAR_Qx = 0; FDR_Qx = 0;
for i = 1:160
    if Tc2(i) > Tc_ctrl
       FAR_Tc = FAR_Tc + 1;
    end
    if Tx2(i) > Tx_ctrl
       FAR_Tx = FAR_Tx + 1;
    end
    if Qx(i) > Qx_ctrl
       FAR_Qx = FAR_Qx + 1;
    end                     
end
for i = 161:960
    if Tc2(i) > Tc_ctrl
       FDR_Tc = FDR_Tc + 1;
    end  
    if Tx2(i) > Tx_ctrl
       FDR_Tx = FDR_Tx + 1;
    end 
    if Qx(i) > Qx_ctrl
       FDR_Qx = FDR_Qx + 1;
    end                     
end
FAR_Tc = FAR_Tc / 160; FAR_Tx = FAR_Tx / 160; FAR_Qx = FAR_Qx / 160;
FDR_Tc = FDR_Tc / 800; FDR_Tx = FDR_Tx / 800; FDR_Qx = FDR_Qx / 800;

% ROC curves including f1-score
class_1 = Tc2(1:160); 
class_2 = Tc2(161:960);
figure;
roc_Tc = roc_curve(class_1, class_2);

class_1 = Tx2(1:160); 
class_2 = Tx2(161:960);
figure;
roc_Tx = roc_curve(class_1, class_2);

class_1 = Qx(1:160); 
class_2 = Qx(161:960);
figure;
roc_Qx = roc_curve(class_1, class_2);

% ¼ì²â½á¹û-----------------------------------------------------------------------------------------------------------------
figure;
subplot(3,1,1);plot(Tc2,'k');title('CPLS');hold on;plot(Tc_ctrl(1,1)*ones(1, N),'k--');xlabel('sample');ylabel('Tc^2');legend('statistics','threshold');hold off
subplot(3,1,2);plot(Tx2,'k');title('CPLS');hold on;plot(Tx_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Tx^2');legend('statistics','threshold');hold off
subplot(3,1,3);plot(Qx,'k');title('CPLS');hold on;plot(Qx_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Qx');legend('statistics','threshold');hold off
