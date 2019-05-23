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

Y_e = T * Q'; [Uc, Dc, Vc] = svd(Y_e, 'econ'); l = pc; lc = size(Dc, 1); 
Qc = Vc * Dc; Rc = R * Q' * Vc * pinv(Dc); Yc = Y_test - Uc * Qc';
[Py, Ty, Latenty, Tsquarey] = pca(Yc); ly=0;
for i = 1:size(Latenty,1)
    cpvy = sum(Latenty(1:i)) / sum(Latenty);
    if cpvy >= 0.9
        ly = i; break;
    end
end

Ty = Ty(:, 1:ly); Py = Py(:, 1:ly); Yresidual = Yc - Ty * Py';
Rcinv = pinv(Rc' * Rc) * Rc'; Xc = X_test - Uc * Rcinv;
[Px, Tx, Latentx, Taquarex] = pca(Xc); lx=0;
for i=1:size(Latentx,1)
    cpvx = sum(Latentx(1:i)) / sum(Latentx);
    if cpvx >= 0.9
        lx = i; break;
    end
end
Tx = Tx(:, 1:lx); Px = Px(:, 1:lx); Xresidual = Xc - Tx * Px';

% control limit
ALPHA=0.97;
Tc_ctrl = lc * (n-1) * (n+1) * finv(ALPHA, lc, n - lc) / (n * (n - lc));
Tx_ctrl = lx * (n-1) * (n+1) * finv(ALPHA, lx, n - lx) / (n * (n - lx));
theta=zeros(1,3);
for i=1:3
    for j=(lx+1):m
        theta(i) = Latentx(j)^i + theta(i);
    end
end
h0=1-2*theta(1)*theta(3)/(3*theta(2)^2);
Qx_ctrl=theta(1)*(norminv(ALPHA)*(2*theta(2)*h0^2)^0.5/theta(1)+1+theta(2)*h0*(h0-1)/theta(1)^2)^(1/h0);

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
subplot(3,1,1);plot(Tc2,'k');title('CPLS');hold on;plot(Tc_ctrl(1,1)*ones(1, N),'k--');xlabel('sample');ylabel('Tc^2');legend('statistics','threshold')
hold off
subplot(3,1,2);plot(Tx2,'k');title('CPLS');hold on;plot(Tx_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Tx^2');legend('statistics','threshold')
hold off
subplot(3,1,3);plot(Qx,'k');title('CPLS');hold on;plot(Qx_ctrl*ones(1, N),'k--');xlabel('sample');ylabel('Qx');legend('statistics','threshold')
hold off
