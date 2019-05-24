function D = EuDist2(fea_a,fea_b)
% Euclidean Distance matrix
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%=================================== 
if  isempty(fea_b)
    [nSmp, nFea] = size(fea_a);
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    aa = full(aa);
    ab = full(ab);
    D = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
    D = D - diag(diag(D));
    D = abs(D);
else
    [nSmp_a, nFea] = size(fea_a);
    [nSmp_b, nFea] = size(fea_b);
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';
    aa = full(aa);
    bb = full(bb);
    ab = full(ab);
    D = repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab;
    D = abs(D);
end


