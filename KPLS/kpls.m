function [t,u,Kc,K]=kpls(X,Y,m,options)
n=size(X,1);
K = constructKernel(X,[],options);
I=eye(n);s=ones(n,1);
Kb=(I-s*s'/n)*K*(I-s*s'/n);% 核函数的中心化
Kc=Kb;
% Kb,Y用作迭代
for i=1:m
    u(:,i)=Y; % 初始化u
    t(:,i)=Y; % 初始化t
end;
for i=1:m     % 非线性迭代求解t、u;
    while 1
    	told=t(:,i);
   	    t(:,i)=Kb*u(:,i);
    	t(:,i)=t(:,i)/norm(t(:,i));
    	q(:,i)=Y'*t(:,i);
    	u(:,i)=Y*q(:,i);
        u(:,i)=u(:,i)/norm(u(:,i));
    	if norm(told-t(:,i))/norm(told)<1e-3	%收敛判据
	       break;
    	end;
    end;
    Kb=(I-t(:,i)*t(:,i)')*Kb*(I-t(:,i)*t(:,i)');
    Y=Y-t(:,i)*t(:,i)'*Y;
end;