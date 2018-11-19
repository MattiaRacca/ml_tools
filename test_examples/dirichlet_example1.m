set(0,'DefaultFigureWindowStyle','docked');
close all; clc;


%% Usage and test of DirichletDistribution
D1 = DirichletDistribution(3,[1.5 1.5 1.5]');
D2 = DirichletDistribution(3,[5 5 5]');

% Test plotting
figure;
D2.plotDistribution();

% sample from D1
samples = D1.sampleDistribution(10);
% test samples from D1 to D2
logprob = D2.logpdf(samples);

%% MLE estimate
samples = D1.sampleDistribution(100);

D3 = DirichletDistribution(3);
D3.MLE(samples);

%% Test Moments and Entropy
D4 = DirichletDistribution(3,[3 10 1]');
D4.mean
D4.mode
D4.variance
D4.entropy

%% Weighted MLE estimate
D5 = DirichletDistribution(3,[1 10 1]');
samples = D5.sampleDistribution(1000);
weights = weightMask(samples(2,:)');

D6 = DirichletDistribution(3);
D6.weightedMLE(samples, weights);

%% KL divergence

D7 = DirichletDistribution(3,[3 10 1]');
D8 = DirichletDistribution(3,[12 2 2]');

KL = computeKLDivergence(D7, D8);