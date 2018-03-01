%%
clear all;
close all;
clc;

%% configuration
cfg.maxNodes = 50*ones(1,2);
cfg.maxWeightNorms = 10*ones(size(cfg.maxNodes)); %capital lambda
cfg.pnorm = 2;
cfg.maxWeightMag = 10*ones(size(cfg.maxNodes));
cfg.maxBiasMag = 10;
cfg.complexityRegWeight = logspace(-4,-3,length(cfg.maxNodes));
cfg.normRegWeight = 0.1*ones(size(cfg.maxNodes)); %beta
cfg.augment = true;
cfg.augmentLayers = true;
cfg.activationFunction = 'relu';
cfg.numEpochs = 30;
cfg.big_lambda = 1.01;
cfg.featureMap = [];

cfg.lossFunction = 'binary';
cfg.surrogateLoss = 'logistic';
cfg.javier = true
%% load data
[X_train, y_train, X_test, y_test] = load_data('deer', 'horse'); 

%% train
[adaParams,history] = adanet(X_train, double(y_train), cfg);

%% predict
preds = zeros(size(y_test,1),1);
for i = 1:size(y_test,1)
    preds(i,1) = adanet_predict(adaParams, X_test(i,:));
end

accuracy = sum(preds==y_test)/size(y_test,1)



