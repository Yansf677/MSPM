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
T = T(:, 1:pc); R = R(:, 1:pc); P = P(:, 1:pc); Q = Q(:, 1:pc);

Ty = T * Q'; X_e = T * P'; Py = (X_e' * Ty) / (Ty'*Ty); Xo = X_e - Ty * Py';
[Po, To, Latento, Tsquareo] = pca(Xo); ko = pc-1;
To = To(:, 1:ko); Po = Po(:, 1:ko); E = X_train - X_e;
[Pr, Tr, Latentr, Tsquarer] = pca(E); kr=0;
for i = 1:size(Latentr, 1)
    cpvr = sum(Latentr(1:i)) / sum(Latentr);
    if cpvr >= 0.9
        kr=i; break;
    end
end
Tr = Tr(:, 1:kr); Pr = Pr(:, 1:kr); Er = E - Tr*Pr';

% control limit
ALPHA=0.97;
Ty_ctrl = 1 * (n-1) * (n+1) * finv(ALPHA, 1, n-1) / (n*(n-1));
To_ctrl = ko * (n-1) * (n+1) * finv(ALPHA, ko, n-ko) / (n*(n-ko));
Tr_ctrl = kr * (n-1) * (n+1) * finv(ALPHA, kr, n-kr) / (n*(n-kr));
% Qr = zeros(n, 1);
% for i=1:n
%    Qr(i) = Er(i,:) * Er(i,:)';
% end
% miu = mean(Qr); S = var(Qr); g = S / (2 * miu); h = 2 * miu * miu / S;
% Qr_ctrl = g * chi2inv(ALPHA, h);
theta=zeros(1,3);
for i=1:3
    for j = (kr+1):m
        theta(i) = Latentr(j)^i + theta(i);
    end
end
h0 = 1 - 2 * theta(1) * theta(3) / (3 * theta(2)^2);
Qr_ctrl = theta(1) * (norminv(ALPHA) * (2 * theta(2) * h0^2)^0.5 / theta(1) + 1 + theta(2) * h0 * (h0-1) / theta(1)^2)^(1/h0);
 
%% online testing
Ty2 = zeros(N, 1); To2 = zeros(N, 1); Tr2 = zeros(N, 1); Qr = zeros(N, 1);
for i=1:N
   tynew = Q * R' * (X_test(i,:))'; 
   Ty2(i) = tynew' * pinv(((Ty') * Ty)/(n-1)) * (tynew);
   tonew = Po' * (P - Py * Q) * R' * (X_test(i,:))';
   To2(i) = tonew' * pinv(((To') * To)/(n-1)) * (tonew); 
   trnew = Pr' * (eye(m) - P * R') * (X_test(i,:))';
   Tr2(i) = trnew' * pinv(((Tr') * Tr)./(n-1))*(trnew);
   xrnew = (eye(m)-Pr * Pr') * (eye(m)-P * R') * (X_test(i,:))';
   Qr(i) = xrnew' * xrnew;
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
class_1 = Ty2(1:160); 
class_2 = Ty2(161:960);
figure;
roc_Ty = roc_curve(class_1, class_2);

class_1 = To2(1:160); 
class_2 = To2(161:960);
figure;
roc_To = roc_curve(class_1, class_2);

class_1 = Tr2(1:160); 
class_2 = Tr2(161:960);
figure;
roc_Tr = roc_curve(class_1, class_2);

class_1 = Qr(1:160); 
class_2 = Qr(161:960);
figure;
roc_Q = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,2,1);plot(Ty2,'k');title('T-PLS Ty^2');hold on;plot(Ty_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Ty^2');%legend('statistics','threshold');
hold off
subplot(2,2,2);plot(To2,'k');title('T-PLS To^2');hold on;plot(To_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('To^2');%legend('statistics','threshold');
hold off;
subplot(2,2,3);plot(Tr2,'k');title('T-PLS Tr^2');hold on;plot(Tr_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Tr^2');%legend('statistics','threshold')
hold off
subplot(2,2,4);plot(Qr,'k');title('T-PLS Q');hold on;plot(Qr_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Q');%legend('statistics','threshold')
hold off

 