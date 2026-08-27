% mRMR by Ding, 2005
clear; clc

cd('../data/')
X = readtable('Xtrain.csv', 'ReadRowNames', true);
X = table2array(X);
y = readtable('ytrain.csv', 'ReadRowNames', true);
y = table2array(y);
cd('../matlabFS/')

% mRMR
tStart_mRMR = cputime;
mRMR_idx = fscmrmr(X, y);
tEnd_mRMR = cputime - tStart_mRMR;
tmRMR = (tEnd_mRMR);
writematrix(mRMR_idx, "Epilepsy_mRMR_ranks.csv");
writematrix(tmRMR, "Epilepsy_tmRMR.csv");

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [7];

% LH-Relief
tStart_LH = cputime;
[Weight_LM, ~] = LHR(X', y, Para4IRelief);
tEnd_LH = cputime - tStart_LH;
t_LM = (tEnd_LH);
writematrix(Weight_LM, "Epilepsy_LHReliefscores.csv");
writematrix(t_LM, "Epilepsy_tLH.csv");

% I-Relief
tStart_I = cputime;
[Weight_I, ~] = IMRelief_1(X', y, Para4IRelief);
tEnd_I = cputime - tStart_I;
t_I = (tEnd_I);
writematrix(Weight_I, "Epilepsy_IReliefscores.csv");
writematrix(t_I, "Epilepsy_tI.csv");
