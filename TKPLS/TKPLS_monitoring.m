clc
clear
%% data preprocessing
load TEdata.mat; IDV = 21; 
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
k = 8; % base on the above crossvalidation
[t, u, Kc, K, ~] = kpls(X_train, Y_train, k, options);
pc = 1:k;
s = ones(n,1); I = eye(n);
temp = t(:,pc)' * Kc * u(:,pc);
T = Kc * u(:,pc) / temp;
Ty = Kc * u(:,pc) / temp * t(:,pc)' * Y_train;
Zy = I - Ty / (Ty' * Ty) * Ty'; Q = Y_train' * T; Qy=1; Ao=7; Ar=6; 
Ko = Zy * t(:, pc) * t(:, pc)' * Kc * t(:, pc) * t(:, pc)' * Zy;
Kr = (I - t(:,pc) * t(:, pc)') * Kc * (I - t(:, pc) * t(:, pc)');

[Wo, L_Wo, ~]=svd(Ko);
for i = 1:n 
    Wo(:,i) = Wo(:,i) / sqrt(L_Wo(i,i));
end
L_Wo = L_Wo / n;
To = Ko * Wo(:, 1:Ao); Wo = Wo(:, 1:Ao);

[Wr, L_Wr, ~] = svd(Kr);
for i = 1:n 
    Wr(:,i) = Wr(:,i) / sqrt(L_Wr(i,i));
end
L_Wr = L_Wr / n;
Tr = Kr * Wr(:, 1:Ar); Wr = Wr(:,1:Ar);

Faiy = (u(:,pc) / temp * t(:,pc)' * Y_train)';
Faio = Wo' * Zy * t(:,pc) * t(:,pc)' * Kc * t(:,pc) / (u(:,pc)' * Kc * t(:,pc)) * u(:,pc)' - ...
       Wo' * Zy * t(:,pc) * t(:,pc)' * Kc * t(:,pc) * t(:,pc)' * Ty / (Ty'*Ty) * Faiy;
Fair = Wr' * (I - t(:,pc) * t(:,pc)') - Wr'  *(I - t(:,pc) * t(:,pc)') * Kc * t(:,pc) / ...
       (u(:,pc)' * Kc * t(:,pc)) * u(:,pc)';

Ty2 = zeros(n, 1); To2 = zeros(n, 1); Tr2 = zeros(n, 1); Qr = zeros(n, 1);
for i = 1:n
    Kpredict = constructKernel(X_train(i,:), X_train, options);
    Kp = (Kpredict - s' * K / n) * (I - s * s' / n);
    tnew = Kp * u(:,pc) / (t(:,pc)' * Kc * u(:,pc));
    tynew = Faiy * Kp';
    tonew = Faio * Kp';
    trnew = Fair * Kp';
    Qr(i) = 1 - 2 / n * sum(Kpredict) + 1 / n / n * sum(sum(K))...
              - 2 * (Kp * (I - t(:,pc) * t(:,pc)') * Wr * trnew - tnew * ...
              t(:,pc)' * Kc * (I - t(:,pc) * t(:,pc)') * Wr * trnew) + ...
              trnew' * Wr' * (I - t(:,pc) * t(:,pc)') * Kc * (I - t(:,pc) * t(:,pc)') * Wr * trnew;
    Tr2(i) = trnew' / L_Wr(1:Ar,1:Ar) * trnew;
    To2(i) = tonew' / L_Wo(1:Ao,1:Ao) * tonew;
    Ty2(i) = tynew'/(Ty' * Ty / (n-1)) * tynew;
end

% control limit
ALPHA = 0.99;
Ty_ctrl = (n*n-1) * finv(ALPHA,1,n-1) / (n*(n-1));
To_ctrl = Ao * (n*n-1) * finv(ALPHA,Ao,n-Ao) / (n*(n-Ao));
Tr_ctrl = Ar * (n*n-1) * finv(ALPHA,Ar,n-Ar) / (n*(n-Ar));
miu = mean(Qr); S = var(Qr); g = S / (2 * miu); h = 2 * miu * miu / S;
Qr_ctrl = g * chi2inv(ALPHA, h);

%% online testing
Ty2 = zeros(N, 1); To2 = zeros(N, 1); Tr2 = zeros(N, 1); Qr = zeros(N, 1);
for i = 1:N
    Kpredict = constructKernel(X_test(i,:), X_train, options);
    Kp = (Kpredict - s' * K/n) * (I - s * s'/n);
    tnew = Kp * u(:,pc) / (t(:,pc)' * Kc * u(:,pc));
    tynew = Faiy * Kp';
    tonew = Faio * Kp';
    trnew = Fair * Kp';
    Qr(i) = 1 - 2 / n * sum(Kpredict) + 1 / n / n * sum(sum(K)) ...
              - 2 * (Kp * (I - t(:,pc) * t(:,pc)') * Wr * trnew - tnew * ...
              t(:,pc)' * Kc * (I - t(:,pc) * t(:,pc)') * Wr * trnew) + ...
              trnew' * Wr' * (I - t(:,pc) * t(:,pc)') * Kc * (I - t(:,pc) * t(:,pc)') * Wr * trnew;
    Tr2(i) = trnew' / L_Wr(1:Ar,1:Ar) * trnew;
    To2(i) = tonew' / L_Wo(1:Ao,1:Ao) * tonew;
    Ty2(i) = tynew' / (Ty'*Ty / (n-1)) * tynew;
end

% type I and type II errors
FAR_Ty = 0; FDR_Ty = 0;
FAR_To = 0; FDR_To = 0;
FAR_Tr = 0; FDR_Tr = 0;
FAR_Qr = 0; FDR_Qr = 0;
for i = 1:160
    if Ty2(i) > Ty_ctrl
       FAR_Ty = FAR_Ty + 1;
    end
    if To2(i) > To_ctrl
       FAR_To = FAR_To + 1;
    end
    if Tr2(i) > Tr_ctrl
       FAR_Tr = FAR_Tr + 1;
    end
    if Qr(i) > Qr_ctrl
       FAR_Qr = FAR_Qr + 1;
    end                     
end
for i = 161:960
    if Ty2(i) > Ty_ctrl
       FDR_Ty = FDR_Ty + 1;
    end  
    if To2(i) > To_ctrl
       FDR_To = FDR_To + 1;
    end 
    if Tr2(i) > Tr_ctrl
       FDR_Tr = FDR_Tr + 1;
    end 
    if Qr(i) > Qr_ctrl
       FDR_Qr = FDR_Qr + 1;
    end                     
end
FAR_Ty = FAR_Ty / 160; FAR_To = FAR_To / 160; FAR_Tr = FAR_Tr / 160; FAR_Qr = FAR_Qr / 160;
FDR_Ty = FDR_Ty / 800; FDR_To = FDR_To / 800; FDR_Tr = FDR_Tr / 800; FDR_Qr = FDR_Qr / 800;

% ROC curves including f1-score
class_1 = Ty2(1:160); class_2 = Ty2(161:960);
figure; roc_Ty = roc_curve(class_1, class_2);

class_1 = To2(1:160); class_2 = To2(161:960);
figure; roc_To = roc_curve(class_1, class_2);

class_1 = Tr2(1:160); class_2 = Tr2(161:960);
figure; roc_Tr = roc_curve(class_1, class_2);

class_1 = Qr(1:160); class_2 = Qr(161:960);
figure; roc_Q = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,2,1);plot(Ty2,'k');title('TKPLS');hold on;plot(Ty_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Ty^2');legend('statistics','threshold');hold off
subplot(2,2,2);plot(To2,'k');title('TKPLS');hold on;plot(To_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('To^2');legend('statistics','threshold');hold off;
subplot(2,2,3);plot(Tr2,'k');title('TKPLS');hold on;plot(Tr_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Tr^2');legend('statistics','threshold');hold off
subplot(2,2,4);plot(Qr,'k');title('TKPLS');hold on;plot(Qr_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Q');legend('statistics','threshold');hold off

 