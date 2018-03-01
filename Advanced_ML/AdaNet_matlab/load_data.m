function [X_train, y_train, X_test, y_test] = load_data(pair1, pair2)
file = ['../data/CIFAR10_pair_', pair1, '_', pair2, '.mat'];
load(file)