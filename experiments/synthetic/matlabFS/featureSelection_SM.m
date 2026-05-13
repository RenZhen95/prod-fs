% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
[wdir, ~] = fileparts(mfilename("fullpath"));
nClass2_idxs = [16; 43; 70; 97];
nClass3_idxs = [17; 44; 71; 98];
nClass4_idxs = [18; 45; 72; 99];

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [7];

% Only considering 3 different dimensions, 20 genes per dimension
%  - 4060 features | 3 number of class x 4 iterations per dataset type
Weights_LM_nClass2 = zeros(4060, 4);
Weights_I_nClass2  = zeros(4060, 4);

Weights_LM_nClass3 = zeros(4060, 4);
Weights_I_nClass3  = zeros(4060, 4);

Weights_LM_nClass4 = zeros(4060, 4);
Weights_I_nClass4  = zeros(4060, 4);

t_LM = zeros(3, 4);
t_I  = zeros(3, 4);

for i=1:4
    % Adjust path accordingly
    cd('../SM/SM-Datasets/')
    X_nClass2 = readmatrix("X/" + nClass2_idxs(i) + "_X.csv");
    y_nClass2 = readmatrix("y/" + nClass2_idxs(i) + "_y.csv");
    y_nClass2 = y_nClass2 + 1;

    X_nClass3 = readmatrix("X/" + nClass3_idxs(i) + "_X.csv");
    y_nClass3 = readmatrix("y/" + nClass3_idxs(i) + "_y.csv");
    y_nClass3 = y_nClass3 + 1;

    X_nClass4 = readmatrix("X/" + nClass4_idxs(i) + "_X.csv");
    y_nClass4 = readmatrix("y/" + nClass4_idxs(i) + "_y.csv");
    y_nClass4 = y_nClass4 + 1;

    cd(wdir)
    % LH-Relief
    tStart_LH_nClass2 = cputime;
    [Weight_LM_nClass2, ~] = LHR(X_nClass2', y_nClass2, Para4IRelief);
    Weights_LM_nClass2(:,i) = Weight_LM_nClass2;
    tEnd_LH_nClass2 = cputime - tStart_LH_nClass2;
    t_LM(1, i) = tEnd_LH_nClass2;

    tStart_LH_nClass3 = cputime;
    [Weight_LM_nClass3, ~] = LHR(X_nClass3', y_nClass3, Para4IRelief);
    Weights_LM_nClass3(:,i) = Weight_LM_nClass3;
    tEnd_LH_nClass3 = cputime - tStart_LH_nClass3;
    t_LM(2, i) = tEnd_LH_nClass3;

    tStart_LH_nClass4 = cputime;
    [Weight_LM_nClass4, ~] = LHR(X_nClass4', y_nClass4, Para4IRelief);
    Weights_LM_nClass4(:,i) = Weight_LM_nClass4;
    tEnd_LH_nClass4 = cputime - tStart_LH_nClass4;
    t_LM(3, i) = tEnd_LH_nClass4;

    % I-Relief
    tStart_I_nClass2 = cputime;
    [Weight_I_nClass2, ~] = IMRelief_1(X_nClass2', y_nClass2, Para4IRelief);
    Weights_I_nClass2(:,i) = Weight_I_nClass2;
    tEnd_I_nClass2 = cputime - tStart_I_nClass2;
    t_I(1, i) = tEnd_I_nClass2;

    tStart_I_nClass3 = cputime;
    [Weight_I_nClass3, ~] = IMRelief_1(X_nClass3', y_nClass3, Para4IRelief);
    Weights_I_nClass3(:,i) = Weight_I_nClass3;
    tEnd_I_nClass3 = cputime - tStart_I_nClass3;
    t_I(2, i) = tEnd_I_nClass3;

    tStart_I_nClass4 = cputime;
    [Weight_I_nClass4, ~] = IMRelief_1(X_nClass4', y_nClass4, Para4IRelief);
    Weights_I_nClass4(:,i) = Weight_I_nClass4;
    tEnd_I_nClass4 = cputime - tStart_I_nClass4;
    t_I(3, i) = tEnd_I_nClass4;
end

% Save scores and elapsed times
cd(wdir)
writematrix(Weights_LM_nClass2, "WeightsLM_nClass2.csv");
writematrix(Weights_LM_nClass3, "WeightsLM_nClass3.csv");
writematrix(Weights_LM_nClass4, "WeightsLM_nClass4.csv");
writematrix(Weights_I_nClass2, "WeightsI_nClass2.csv");
writematrix(Weights_I_nClass3, "WeightsI_nClass3.csv");
writematrix(Weights_I_nClass4, "WeightsI_nClass4.csv");
writematrix(t_LM, "tLM.csv");
writematrix(t_I, "tI.csv");