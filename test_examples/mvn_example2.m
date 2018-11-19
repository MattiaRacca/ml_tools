set(0,'DefaultFigureWindowStyle','docked');
close all; clc;
rng('Default'); % for testing purposes

%% Usage of MVN Class 2 - Posterior

dimension = 50;

%% Generate covariance matrix and mean vector
% eigenvalues = rand(dimension,1);
% eigenvalues = exprnd(ones(dimension,1));
% eigenvalues = 1./(1:dimension);
% eigenvalues = eigenvalues/sum(eigenvalues)*dimension;
% covariance = gallery('randcorr', eigenvalues);

covariance = gallery('randsvd',dimension,-10);
mu = rand(dimension,1)*20;
mvn1 = MultivariateNormalDistribution(dimension,mu,covariance);

%% Sample mvn1
N = round(dimension*1.5);
prior_samples = mvn1.sampleDistribution(N);

%% Compute uninformative prior
% mvn_prior = MultivariateNormalDistribution();
% mvn_prior.MLE(prior_samples);
% 
% nu0 = dimension + 2;
% S0 = diag(diag(mvn_prior.sigma))*nu0;
% m0 = mvn_prior.mu;
% kappa0 = 0.01;

%% Compute informative prior
mvn_prior = MultivariateNormalDistribution();
mvn_prior.MLE(prior_samples);

nu0 = N;
S0 = diag(diag(mvn_prior.sigma))*nu0;
m0 = mvn_prior.mu;
kappa0 = N;

%% Create new distributions

% mvn2 will do only MLE
mvn2 = MultivariateNormalDistribution();
% mvn3 will have a prior and do posterior analysis
mvn3 = MultivariateNormalDistribution(dimension, m0, S0, kappa0, nu0);

%% Sample mvn1 again
N = dimension;
samples = mvn1.sampleDistribution(N);

mvn2.MLE([samples, prior_samples]);
mvn3.MAP(samples);

%% Compute likelihood
N = 1;
samples_test = mvn1.sampleDistribution(N);

lik1 = mvn1.logpdf(samples_test);
lik2 = mvn2.logpdf(samples_test);
lik3 = mvn3.logPosteriorPredictive(samples_test);

%% Plot eigenvalues spectrum
figure; hold on;
eig1 = sort(eig(mvn1.sigma), 'descend');
eig2 = sort(eig(mvn2.sigma), 'descend');
eig3 = sort(eig(mvn3.sigma), 'descend');

plot(eig1);
plot(eig2);
plot(eig3);

text(1, eig1(1), num2str(cond(mvn1.sigma),4));
text(1, eig2(1), num2str(cond(mvn2.sigma),4));
text(1, eig3(1), num2str(cond(mvn3.sigma),4));
legend('Real', 'MLE', 'MAP')