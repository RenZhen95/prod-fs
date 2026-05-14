% mRMR by Ding, 2005
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

rank_nObs30 = zeros(100, 5);
rank_nObs50 = zeros(100, 5);
rank_nObs70 = zeros(100, 5);

t_mRMR = zeros(3, 50);

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

    % === === === ===
    % 30
    tStart_30 = cputime;
    idx_30 = fscmrmr(X30, y30);
    tEnd_30 = cputime - tStart_30;
    rank_nObs30(:,i+1) = idx_30;
    t_mRMR(1, i+1) = tEnd_30;

    % === === === ===
    % 50
    tStart_50 = cputime;
    idx_50 = fscmrmr(X50, y50);
    tEnd_50 = cputime - tStart_50;
    rank_nObs50(:,i+1) = idx_50;
    t_mRMR(2, i+1) = tEnd_50;

    % === === === ===
    % 70
    tStart_70 = cputime;
    idx_70 = fscmrmr(X70, y70);
    tEnd_70 = cputime - tStart_70;
    rank_nObs70(:,i+1) = idx_70;
    t_mRMR(3, i+1) = tEnd_70;
end

% Save scores and elapsed times
cd(wdir)

writematrix(rank_nObs30, datasetname + "mRMR_nObs30.csv");
writematrix(rank_nObs50, datasetname + "mRMR_nObs50.csv");
writematrix(rank_nObs70, datasetname + "mRMR_nObs70.csv");
writematrix(t_mRMR, datasetname + "tmRMR.csv");
