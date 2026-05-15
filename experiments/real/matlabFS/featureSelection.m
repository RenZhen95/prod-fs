% Feature selection of I-RELIEF and LHR
% The coded is implemented based on Y.J, Sun's IRELIEF.
clear; clc

pickle = py.importlib.import_module('pickle');
cd('../')

handle = py.open("ProcessedDatasets.pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('matlabFS/')

% Parameters of IRELIEF taken from Y.J, Sun
it = 15;
Para4IRelief.it = it;
Para4IRelief.distance = 'euclidean';
Para4IRelief.kernel = 'exp';
Para4IRelief.Outlier = 0;
Para4IRelief.sigma = 0.5;
Para4IRelief.Prob = 'yes';
Para4IRelief.NN = [7];

processedDatasets = struct(processedDatasets_dict);
datasetKeys = fieldnames(processedDatasets);

% Measuring the elapsed time for both methods
LHRelief_timefile = fopen("LHRelief_time.csv", "w");
IRelief_timefile = fopen("IRelief_time.csv", "w");

for ds_i=1:length(datasetKeys)
  ds_name = datasetKeys{ds_i};
  dataset = processedDatasets_dict.get(ds_name);

  X_ = dataset.get('X');
  Xflat = double(X_.flatten().tolist());
  X = reshape(Xflat, int32(X_.shape(1)), int32(X_.shape(2)));

  y = double(dataset.get('y').tolist())';

  if ds_name == "geneExpressionCancerRNA"
      y = y+1;
  end
 
  % LH-Relief
  disp(ds_name + " (LH-Relief) ...")
  tStart_LH = cputime;
  [Weight_LM, Theta_LM] = LHR(X', y, Para4IRelief);
  writematrix(Weight_LM, strcat(ds_name, '_ReliefLM.csv'));
  tEnd_LH = cputime - tStart_LH;

  % Standard I-Relief
  disp(ds_name + " (I-Relief) ...")
  tStart_I = cputime;
  [Weight_I, Theta_I] = IMRelief_1(X', y, Para4IRelief);
  writematrix(Weight_I, strcat(ds_name, '_ReliefI.csv'));
  tEnd_I = cputime - tStart_I;

  % Recording elapsed time
  fprintf(LHRelief_timefile, ds_name + "," + tEnd_LH + "\n");
  fprintf(IRelief_timefile, ds_name + "," + tEnd_I + "\n");
end

fclose(LHRelief_timefile);
fclose(IRelief_timefile);