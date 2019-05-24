function [t,u,Kc,K]=kpls(X,Y,m,options)
n=size(X,1);
K = constructKernel(X,[],options);
I=eye(n);s=ones(n,1);
Kb=(I-s*s'/n)*K*(I-s*s'/n);
Kc=Kb;

for i=1:m
    u(:,i)=Y; 
    t(:,i)=Y;
end
for i=1:m    
    while 1
    	told=t(:,i);
   	    t(:,i)=Kb*u(:,i);
    	t(:,i)=t(:,i)/norm(t(:,i));
    	q(:,i)=Y'*t(:,i);
    	u(:,i)=Y*q(:,i);
        u(:,i)=u(:,i)/norm(u(:,i));
    	if norm(told-t(:,i))/norm(told)<1e-3	
	       break;
        end
    end
    Kb=(I-t(:,i)*t(:,i)')*Kb*(I-t(:,i)*t(:,i)');
    Y=Y-t(:,i)*t(:,i)'*Y;
end