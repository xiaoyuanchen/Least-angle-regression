function [weights, Cp] = lars(features, target)
%
% Least angle regression algorithm
%   uses C_p Mallow's criterium
%
% Arguments
% Input
%   features - matrix of features, where rows are objects, and colums are feature vectors
%   target   - target feature vector

%   featuresRating - structure with rating for all features; has fields
%     isInformative - array of marks is particular feature informative (1) or not (0)
%     weight        - weight of particular feature id it is informative
%


X = features;
y = target;

[n,p] = size(X);
muA = zeros(n,1);  % current y-estimate
beta = zeros(p,1); % current parameter vector
b = [];  % list of models
A = [];  %current set of features
Cp = zeros(p,1); % Mallow's criterion 
SSE= zeros(p,1); % SSE criterion
AIC= zeros(p,1);  % Akaike's criterion
betaAll = lscov(X,y);
MSE = sumsqr(y - X*betaAll)/n;

for i = 1:p 
  c = X'*(y-muA);
  B = setdiff(1:p,A); % all indexes except indexes in A
  [C, idxC] = max(abs(c(B))); % find maximal value of correlation and corresponding index j of column in X
  idxC = B(idxC); % returning to indexes from X
  A = unique([A; idxC]);% add new index to the current set
  Sj  = sign(c(A)); %signes of correlations
  XA = X(:,A)*diag(Sj);
  G = pinv(XA'*XA);
  oA = ones(length(A),1);
  AA =(oA'*G*oA)^(-0.5);
  wA = AA*G*oA;
  uA = XA*wA; %unit vector
  a = X'*uA;
  % get the step value
  if i<p
      M = [(C-c(B))./(AA-a(B));(C+c(B))./(AA+a(B))];
      M(find(M<=0))=+Inf;
      gamma = min(M);
  else
      gamma  = C/AA;
  end
  muA = muA + gamma*uA; %update y-approximation
  beta(A) = beta(A) + gamma*diag(Sj)*wA; % update parameters vector
  b= cat(1, b, beta');
  Cp(i) = sumsqr(y - X*beta)/MSE +2*i - n;% Mallows criterium
  SSE(i)= sumsqr(y - X*beta);
  AIC(i)= 2*i+n*log(sumsqr(y - X*beta)/(n-2));
end
[minCp,idx] = min(Cp);% find the best model
bestModel = b(idx,:);
weights= bestModel;
