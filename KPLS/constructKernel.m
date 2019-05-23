function [K] = constructKernel(fea_a,fea_b,options)
%	Usage:
%	K = constructKernel(fea_a,[],options)
%   K = constructKernel(fea_a,fea_b,options)
%	fea_a, fea_b  : Rows of vectors of data points. 
%   options       : Struct value in Matlab. The fields in options that can
%                   be set: 
%               KernelType      -  Choices are:
%               'Gaussian'      - e^{-(|x-y|^2)/2t^2}
%               'Polynomial'    - (x'*y)^d
%               'PolyPlus'      - (x'*y+1)^d
%
%               t                -  parameter for Gaussian
%               PolyDegree       -  parameter for Poly
%=================================================
switch lower(options.KernelType)
    case {lower('Gaussian')}       
        if isempty(fea_b)
            D = EuDist2(fea_a,[]);
%               D = pdist2(fea_a,fea_a,'euclidean');
%               D = D.*D;
        else
            D = EuDist2(fea_a,fea_b);
%               D = pdist2(fea_a,fea_b,'euclidean');
%               D = D.*D;
        end
        K = exp(-D/(2*options.t^2));
    case {lower('Polynomial')}     
        if isempty(fea_b)
            D = fea_a * fea_a';
        else
            D = fea_a * fea_b';
        end
        K = D.^options.PolyDegree;
    case {lower('PolyPlus')}     
        if isempty(fea_b)
            D = fea_a * fea_a';
        else
            D = fea_a * fea_b';
        end
        K = (D+1).^options.PolyDegree;
    otherwise
        error('KernelType does not exist!');
end
