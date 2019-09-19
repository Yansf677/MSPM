clc
clear

%% data preprocessing
load TEdata.mat; IDV = 1; 
X_train = data(:, [1:22,42:52], 22);
X_test = data(:, [1:22,42:52], IDV);

[n, m] = size(X_train); 
[X_train, Xmean, Xstd] = zscore(X_train); 

[N, ~] = size(X_test);
X_test = (X_test - repmat(Xmean, N, 1))./repmat(Xstd, N, 1);

%% offline training
% pca
[P, T, Latent, Tsquare] = pca(X_train);
pc = 0;
for i = 1:size(Latent,1)
    cpv = sum(Latent(1:i)) / sum(Latent);
    if cpv >= 0.9
        pc = i;break;
    end
end
T = T(:, 1:pc); P = P(:, 1:pc);

% control limit
ALPHA=0.97;
T_ctrl = pc * (n-1) * (n+1) * finv(ALPHA, pc, n - pc) / (n * (n - pc));
theta=zeros(1,3);
for i=1:3
    for j = (pc+1):m
        theta(i) = Latent(j)^i + theta(i);
    end
end
h0 = 1 - 2 * theta(1) * theta(3) / (3 * theta(2)^2);
Q_ctrl = theta(1) * (norminv(ALPHA) * (2 * theta(2) * h0^2)^0.5 / theta(1) + 1 + theta(2) * h0 * (h0 - 1) / theta(1)^2)^(1/h0);

%% online testing
T2 = zeros(N, 1); Q = zeros(N, 1);
for i = 1:N
   tnew = X_test(i,:) * P;
   T2(i) = tnew * diag(1./Latent(1:pc)) * tnew';
   Q(i) = ((eye(m) - P * P') * X_test(i,:)')' * ((eye(m) - P * P') * X_test(i,:)');
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
% class_1 = T2(1:160); class_2 = T2(161:960);
% figure; roc_Ty = roc_curve(class_1, class_2);
% 
% class_1 = Q(1:160); class_2 = Q(161:960);
% figure; roc_To = roc_curve(class_1, class_2);

% statistics plot
figure;
subplot(2,1,1);plot(T2,'k');title('PCA');hold on;plot(T_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('T^2');legend('statistics','threshold');hold off
subplot(2,1,2);plot(Q,'k');title('PCA');hold on;plot(Q_ctrl*ones(1,N),'k--');xlabel('sample');ylabel('Q');legend('statistics','threshold');hold off




