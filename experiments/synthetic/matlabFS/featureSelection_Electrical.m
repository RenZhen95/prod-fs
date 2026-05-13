% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc
[wdir, ~] = fileparts(mfilename("fullpath"));
% Assuming the pyenv environment is accessible my MATLAB
% Check my typing pyenv in the Command Window
pickle = py.importlib.import_module('pickle');

% Modify file location accordingly
cd('../Electrical/')

% Modify dataset name accordingly
datasetname = "ANDORdiscrete";
%datasetname = "ADDERdiscrete";

%datasetname = "ANDORcontinuous";
%datasetname = "ADDERcontinuous";

handle = py.open(datasetname + "_datasets.pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd(wdir)

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [7];

processedDatasets = dictionary(processedDatasets_dict);
datasets_30obs = dictionary(processedDatasets(30));
datasets_50obs = dictionary(processedDatasets(50));
datasets_70obs = dictionary(processedDatasets(70));

Weights_LM_30 = zeros(100, 50);
Weights_LM_50 = zeros(100, 50);
Weights_LM_70 = zeros(100, 50);
Weights_I_30 = zeros(100, 50);
Weights_I_50 = zeros(100, 50);
Weights_I_70 = zeros(100, 50);
t_LM = zeros(3, 50);
t_I  = zeros(3, 50);

for i=0:49
    dataset_iteration30 = dictionary(datasets_30obs(i));
    dataset_iteration50 = dictionary(datasets_50obs(i));
    dataset_iteration70 = dictionary(datasets_70obs(i));

    X30 = double(dataset_iteration30('X'));
    y30 = double(dataset_iteration30('y'))';

    X50 = double(dataset_iteration50('X'));
    y50 = double(dataset_iteration50('y'))';

    X70 = double(dataset_iteration70('X'));
    y70 = double(dataset_iteration70('y'))';

    % LH-Relief
    tStart_LH_30 = cputime;
    [Weight_LM_30, ~] = LHR(X30', y30, Para4IRelief);
    Weights_LM_30(:,i+1) = Weight_LM_30;
    tEnd_LH_30 = cputime - tStart_LH_30;
    t_LM(1, i+1) = tEnd_LH_30;

    tStart_LH_50 = cputime;
    [Weight_LM_50, ~] = LHR(X50', y50, Para4IRelief);
    Weights_LM_50(:,i+1) = Weight_LM_50;
    tEnd_LH_50 = cputime - tStart_LH_50;
    t_LM(2, i+1) = tEnd_LH_50;

    tStart_LH_70 = cputime;
    [Weight_LM_70, ~] = LHR(X70', y70, Para4IRelief);
    Weights_LM_70(:,i+1) = Weight_LM_70;
    tEnd_LH_70 = cputime - tStart_LH_70;
    t_LM(3, i+1) = tEnd_LH_70;

    % I-Relief
    tStart_I_30 = cputime;
    [Weight_I_30, ~] = IMRelief_1(X30', y30, Para4IRelief);
    Weights_I_30(:,i+1) = Weight_I_30;
    tEnd_I_30 = cputime - tStart_I_30;
    t_I(1, i+1) = tEnd_I_30;

    tStart_I_50 = cputime;
    [Weight_I_50, ~] = IMRelief_1(X50', y50, Para4IRelief);
    Weights_I_50(:,i+1) = Weight_I_50;
    tEnd_I_50 = cputime - tStart_I_50;
    t_I(2, i+1) = tEnd_I_50;

    tStart_I_70 = cputime;
    [Weight_I_70, ~] = IMRelief_1(X70', y70, Para4IRelief);
    Weights_I_70(:,i+1) = Weight_I_70;
    tEnd_I_70 = cputime - tStart_I_70;
    t_I(3, i+1) = tEnd_I_70;
end

% Save scores and elapsed times
writematrix(Weights_LM_30, datasetname + "WeightsLM_30.csv");
writematrix(Weights_LM_50, datasetname + "WeightsLM_50.csv");
writematrix(Weights_LM_70, datasetname + "WeightsLM_70.csv");
writematrix(Weights_I_30, datasetname + "WeightsI_30.csv");
writematrix(Weights_I_50, datasetname + "WeightsI_50.csv");
writematrix(Weights_I_70, datasetname + "WeightsI_70.csv");
writematrix(t_LM, datasetname + "_tLM.csv");
writematrix(t_I, datasetname + "_tI.csv");