function [dist] = LM_construction2(local_patterns,pattern,Weight,num_NN)
%compute weighted local manifold distance by Quadratic constrint
%Weight = ones(size(pattern,1),1);
lambda = 1e-3;
N = size(local_patterns,2);
dim =length(Weight);
distance =  (((local_patterns - repmat(pattern,1,N)).^2).*repmat(Weight,1,N)) ;
Dis =sum(distance);
[sorted_Dis, ind] = sort(Dis); 
num = min(num_NN,size(local_patterns,2));
ind_NN = ind(1:num);
NN_dist =  distance(:,ind_NN);
V = local_patterns(:,ind_NN(2:end)); % remove the first sample
W = diag(Weight);
 A = V'*W*V;
H = A+A'+lambda*eye(size(A));  f = -2*pattern'*W*V;
Aeq = ones(1,size(H,1));rhAeq =1;
%%%%%%%%%%%%%%%%%%%%%
%Aeq = [];rhAeq = [];
lb = zeros(1,size(H,1));
% options = optimset('TolFun',1e-4,'LargeScale','off','Algorithm','sqp-legacy','MaxIter',800);
options = optimset('TolFun',1e-4,'LargeScale','off','MaxIter',800);
[alpha,fval,exitflag,output] =  quadprog(H,f,[],[],Aeq,rhAeq,lb,[],[],options);
% if ~exitflag
%     keyboard;
% end
dist = sqrt((pattern - V*alpha).^2.* Weight);
