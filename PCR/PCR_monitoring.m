clc
clear
%% data preprocessing
load TEdata.mat; IDV = 1; 
X_train = data(:, [1:22,42:52], 22); Y_train = data(:, 35, 22);
X_test = data(:, [1:22,42:52], IDV); Y_test = data(:, 35, IDV);

[~, m] = size(X_train); [n, p] = size(Y_train);
[X_train, Xmean, Xstd] = zscore(X_train); [Y_train, Ymean, Ystd] = zscore(Y_train);
[N, ~] = size(X_test);
X_test = (X_test - repmat(Xmean, N, 1))./repmat(Xstd, N, 1); Y_test = (Y_test - repmat(Ymean, N, 1))./repmat(Ystd, N, 1);

%% offline training
% pcr
fold = 5; indices = crossvalind('Kfold', n, fold); RMSE = zeros(m,1);
for i = 1:m
   for j = 1:fold
      test = (indices == j); train = ~test;
      [T, P, Q] = pcr(X_train(train,:), Y_train(train,:), i);
      Y_pre = X_train(test,:) * P * Q';
      RMSE(i) = RMSE(i) +  mse(Y_pre, Y_train(test,:)) / fold;
   end
end
pc = find(RMSE==min(RMSE));
[T, P, Q] = pcr(X_train, Y_train, pc);
Y_e = T * Q';

[Qy, Ty, Latent_y, Tsquare_y] = pca(Y_e);
Py = (((Ty') * Ty) \ (Ty') * X_train)';

% subspace division
Xy = Ty * Py'; Xo = X_train - Xy;

[Po, To, Latento, Taquareo] = pca(Xo); 
ko=0;
for i = 1:size(Latento,1)
    cpvo = sum(Latento(1:i)) / sum(Latento);
    if cpvo >= 0.999
        ko = i; break;
    end
end
To = To(:, 1:ko); Po = Po(:, 1:ko);

% control limit
ALPHA=0.97;
Ty_ctrl = 1*(n-1)*(n+1) * finv(ALPHA, 1, n-1) / (n*(n-1));
To_ctrl = ko*(n-1)*(n+1) * finv(ALPHA, ko, n-ko) / (n*(n-ko));

%% online testing
Ty2 = zeros(N, 1); To2 = zeros(N, 1);
for i=1:N
   tynew = X_test(i,:) * P * Q';
   Ty2(i) = tynew * pinv(((Ty')*Ty) / (n-1)) * (tynew');
   tonew = (X_test(i,:) - X_test(i,:) * P * Q' * Qy * Py') * Po;
   To2(i) = tonew * pinv(((To')*To) / (n-1)) * (tonew');
end

% type I and type II errors
FAR_Ty = 0; FDR_Ty = 0;
FAR_To = 0; FDR_To = 0;
for i = 1:160
    if Ty2(i) > Ty_ctrl
       FAR_Ty = FAR_Ty + 1;
    end                     
    if To2(i) > To_ctrl
       FAR_To = FAR_To + 1;
    end                     
end
for i = 161:960
    if Ty2(i) > Ty_ctrl
       FDR_Ty = FDR_Ty + 1;
    end                     
    if To2(i) > To_ctrl
       FDR_To = FDR_To + 1;
    end                     
end
FAR_Ty = FAR_Ty / 160; FAR_To = FAR_To / 160;
FDR_Ty = FDR_Ty / 800; FDR_To = FDR_To / 800;

% ROC curves including f1-score
class_1 = Ty2(1:160); class_2 = Ty2(161:960);
figure; roc_Ty = roc_curve(class_1, class_2);

class_1 = To2(1:160); class_2 = To2(161:960);
figure; roc_To = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,1,1);plot(Ty2,'k');title('PCR');hold on;plot(Ty_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('Ty^2');legend('statistics','threshold');hold off;
subplot(2,1,2);plot(To2,'k');title('PCR');hold on;plot(To_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('To^2');legend('statistics','threshold');hold off;


 


