function [T, R, Q, P] = pls(X, Y, pc)

E = X; 
F = Y; 
U(:,1) = Y;
[~, m] = size(X);

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
    E = E - T(:,i) * P(:,i)'; 
    F = F - T(:,i) * Q(:,i)';
end
T = T(:, 1:pc); R = R(:,1:pc); Q = Q(:,1:pc); P = P(:, 1:pc);
end