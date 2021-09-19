clc;
clear;
close all;

%% input_args:
%   dataSet : Data sets to be clustered
%   K       : The number of clusters
dataSet = importdata('Sticks.mat');
K = 4;

%% NTHC clustering
[ clusterLabel ] = NTHC_clustering( dataSet, K );

%% clustering results on 2D data sets
if size(dataSet, 2) == 2 && K < 8
    plot_2d_Data( dataSet, clusterLabel )
end
