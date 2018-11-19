set(0,'DefaultFigureWindowStyle','docked');
close all; clc; clear

%% Usage and test of CategoricalDistribution

D_prior = DirichletDistribution(3,[1 2 1]');
Cat_learn = CategoricalDistribution(3, ones(3,1)/3,copy(D_prior));

D_skewed = DirichletDistribution(3, [7, 3, 3]);
Cat_real = CategoricalDistribution(3, D_skewed.sampleDistribution(1));

%% Sample the real Cat and compute the posterior for the learning Cat

samples = Cat_real.sampleDistribution(10);
Cat_learn.updatePosterior(samples);
disp(Cat_learn.posterior.alpha);

Cat_learn.MAP();
Cat_learn.updatePrior();
