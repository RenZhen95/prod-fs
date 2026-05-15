% Feature selection of mRMR
clear; clc

pickle = py.importlib.import_module('pickle');
cd('../')

handle = py.open("ProcessedDatasets.pkl", 'rb');
processedDatasets_dict = pickle.load(handle);
handle.close();
cd('matlabFS/')

processedDatasets = struct(processedDatasets_dict);
datasetKeys = fieldnames(processedDatasets);

% Measuring the elapsed time
mRMR_timefile = fopen("mRMR_time.csv", "w");

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

  % mRMR
  tStart_mRMR = cputime;
  idx_rank = fscmrmr(X, y);
  tEnd_mRMR = cputime - tStart_mRMR;

  % Recording elapsed time
  fprintf(mRMR_timefile, ds_name + "," + tEnd_mRMR + "\n");
  % Write out ranks
  writematrix(idx_rank, strcat(ds_name, '_mRMR.csv'));
end

fclose(mRMR_timefile);