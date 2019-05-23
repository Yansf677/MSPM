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
% pls
E = X_train; F = Y_train; U(:,1) = Y_train;
for i = 1:m
        uold = U(:,i);
    while 1
        w = (uold' * E / (uold' * uold))';
        w = w / norm(w); R(:,i) = w;
        T(:,i) = E * w; Q(:,i) = F' * T(:,i) / (T(:,i)' * T(:,i));
        unew = F * Q(:,i) / Q(:,i)' * Q(:,i);
    if abs(norm(unew-uold)/norm(uold))<1e-3
        U(:,i+1) = unew;
        break;
    end
    uold = unew;
    end
    P(:,i) = (T(:,i)' * E / (T(:,i)' * T(:,i)))';
    E = E - T(:,i) * P(:,i)'; F = F - T(:,i) * Q(:,i)';
end

pc = 6; % base on the reference or cross validation
T = T(:, 1:pc); R = R(:,1:pc); P = P(:, 1:pc);

% control limit
ALPHA=0.97;
T_ctrl = pc*(n-1)*(n+1) * finv(ALPHA, pc, n-pc) / (n * (n-pc));
Q = zeros(n, 1);
for i=1:n
   Q(i)=(X_train(i,:) - T(i,:) * P') * (X_train(i,:) - T(i,:) * P')';
end
miu = mean(Q); S = var(Q); g = S / (2 * miu); h = 2 * miu * miu / S;
Q_ctrl = g * chi2inv(ALPHA, h);

%% online testing
T2 = zeros(N, 1); Q = zeros(N, 1);
for i=1:N
   tnew = X_test(i,:) * R;
   T2(i) = tnew * pinv((T' * T)/(n-1)) * tnew';
   xnew = X_test(i,:) * (eye(m) - R * P');
   Q(i) = xnew * xnew';
end

% type I and type II errors
FAR_T = 0; FDR_T = 0;
FAR_Q = 0; FDR_Q = 0;
for i = 1:160
    if T2(i) > T_ctrl
       FAR_T = FAR_T + 1;
    end                     
    if Q(i) > Q_ctrl
       FAR_Q = FAR_Q + 1;
    end                     
end
for i = 161:960
    if T2(i) > T_ctrl
       FDR_T = FDR_T + 1;
    end                     
    if Q(i) > Q_ctrl
       FDR_Q = FDR_Q + 1;
    end                     
end
FAR_T = FAR_T / 160; FAR_Q = FAR_Q / 160;
FDR_T = FDR_T / 800; FDR_Q = FDR_Q / 800;

% ROC curves including f1-score
class_1 = T2(1:160); 
class_2 = T2(161:960);
figure;
roc_Ty = roc_curve(class_1, class_2);

class_1 = Q(1:160); 
class_2 = Q(161:960);
figure;
roc_To = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,1,1);plot(T2,'k');title('PLS');hold on;plot(T_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('T^2');hold off;
subplot(2,1,2);plot(Q,'k');title('PLS');hold on;plot(Q_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('Q');hold off;
