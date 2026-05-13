% mRMR by Ding, 2005
clear; clc
[wdir, ~] = fileparts(mfilename("fullpath"));
nClass2_idxs = [16; 43; 70; 97];
nClass3_idxs = [17; 44; 71; 98];
nClass4_idxs = [18; 45; 72; 99];

% Only considering 3 different dimensions, 20 genes per dimension
%  - 4060 features | 3 number of class x 4 iterations per dataset type
rank_nClass2 = zeros(4060, 4);
rank_nClass3 = zeros(4060, 4);
rank_nClass4 = zeros(4060, 4);

t_mRMR = zeros(3, 4);

% Adjust path accordingly
cd('../SM/SM-Datasets/')

for i=1:4
    % nClass = 2
    X_nClass2 = readmatrix("X/" + nClass2_idxs(i) + "_X.csv");
    y_nClass2 = readmatrix("y/" + nClass2_idxs(i) + "_y.csv");
    y_nClass2 = y_nClass2 + 1;

    Xmean_nClass2    = mean(X_nClass2);
    Xstd_nClass2     = std(X_nClass2);
    upperEnd_nClass2 = Xmean_nClass2 + Xstd_nClass2;
    lowerEnd_nClass2 = Xmean_nClass2 - Xstd_nClass2;

    X_nClass2discrete = zeros(size(X_nClass2));

    for c=1:size(X_nClass2discrete, 2)
        X_nClass2discrete(:,c) = discretize_feature(X_nClass2(:,c), upperEnd_nClass2(c), lowerEnd_nClass2(c));
    end
    tStart_nClass2 = cputime;
    idx_nClass2 = fscmrmr(X_nClass2discrete, y_nClass2);
    tEnd_nClass2 = cputime - tStart_nClass2;
    rank_nClass2(:,i) = idx_nClass2;
    t_mRMR(1, i) = tEnd_nClass2;

    % === === === ===
    % nClass = 3
    X_nClass3 = readmatrix("X/" + nClass3_idxs(i) + "_X.csv");
    y_nClass3 = readmatrix("y/" + nClass3_idxs(i) + "_y.csv");
    y_nClass3 = y_nClass3 + 1;

    Xmean_nClass3    = mean(X_nClass3);
    Xstd_nClass3     = std(X_nClass3);
    upperEnd_nClass3 = Xmean_nClass3 + Xstd_nClass3;
    lowerEnd_nClass3 = Xmean_nClass3 - Xstd_nClass3;

    X_nClass3discrete = zeros(size(X_nClass3));

    for c=1:size(X_nClass3discrete, 2)
        X_nClass3discrete(:,c) = discretize_feature(X_nClass3(:,c), upperEnd_nClass3(c), lowerEnd_nClass3(c));
    end
    tStart_nClass3 = cputime;
    idx_nClass3 = fscmrmr(X_nClass3discrete, y_nClass3);
    tEnd_nClass3 = cputime - tStart_nClass3;
    rank_nClass3(:,i) = idx_nClass3;
    t_mRMR(2, i) = tEnd_nClass3;

    % === === === ===
    % nClass = 4
    X_nClass4 = readmatrix("X/" + nClass4_idxs(i) + "_X.csv");
    y_nClass4 = readmatrix("y/" + nClass4_idxs(i) + "_y.csv");
    y_nClass4 = y_nClass4 + 1;

    Xmean_nClass4    = mean(X_nClass4);
    Xstd_nClass4     = std(X_nClass4);
    upperEnd_nClass4 = Xmean_nClass4 + Xstd_nClass4;
    lowerEnd_nClass4 = Xmean_nClass4 - Xstd_nClass4;

    X_nClass4discrete = zeros(size(X_nClass4));

    for c=1:size(X_nClass4discrete, 2)
        X_nClass4discrete(:,c) = discretize_feature(X_nClass4(:,c), upperEnd_nClass4(c), lowerEnd_nClass4(c));
    end
    tStart_nClass4 = cputime;
    idx_nClass4 = fscmrmr(X_nClass4discrete, y_nClass4);
    tEnd_nClass4 = cputime - tStart_nClass4;
    rank_nClass4(:,i) = idx_nClass4;
    t_mRMR(3, i) = tEnd_nClass4;
end

% Save scores and elapsed times
cd(wdir)

writematrix(rank_nClass2, "mRMR_nClass2.csv");
writematrix(rank_nClass3, "mRMR_nClass3.csv");
writematrix(rank_nClass4, "mRMR_nClass4.csv");
writematrix(t_mRMR, "tmRMR.csv");

function featureCol = discretize_feature(col, uE, lE)
  featureCol = zeros(size(col));
  for r=1:size(col,1)
      if col(r) > uE
          featureCol(r) = 1;
      elseif col(r) < lE
          featureCol(r) = -1;
      else
          featureCol(r) = 0;
      end
  end
end