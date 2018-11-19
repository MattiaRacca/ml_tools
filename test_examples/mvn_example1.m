set(0,'DefaultFigureWindowStyle','docked');
close all; clc;

%% Usage of MVN Class
covariance = gallery('randsvd',2, -2);
mvn1 = MultivariateNormalDistribution(2,[2;-1],covariance);

%% Sample mvn1
N = 1000;
samples = mvn1.sampleDistribution(N);
figure;
subplot(1,2,1); hold on;
axis equal;
scatter(samples(1,:),samples(2,:));
mvn1.plot2d(2,2);

%% Draw PDF
[X,Y] = meshgrid(0:0.1:4,-3:0.1:1);
P = mvn1.pdf([X(:)'; Y(:)']);
subplot(1,2,2)
axis equal;
surf(X,Y,vec2mat(P, size(X,2))');

%% Draw Gradient
G = mvn1.evaluateGradient([X(:)'; Y(:)']);
figure; hold on;
axis equal;
contour(X,Y,vec2mat(P, size(X,2))');
quiver(X,Y, vec2mat(G(1,:), size(X,2))', vec2mat(G(2,:), size(X,2))');


%% MLE test
sample_size = 10:10:1000;
mu_history = zeros(2,length(sample_size));
mvn2 = MultivariateNormalDistribution();

figure; hold on;
title('MLE mean estimation')
for i=1:length(sample_size)
  samples = mvn1.sampleDistribution(sample_size(i));
  mvn2.MLE(samples);
  mu_history(:,i) = mvn2.mu;
end

plot(sample_size, mu_history(1,:));
plot(sample_size, mu_history(2,:));

%% Likelihood test
N = 1;
samples = mvn1.sampleDistribution(N);
lik1 = 1;

for i=1:N
  lik1 = lik1 * mvn1.pdf(samples(:,i));
end

lik2 = mvn1.likelihood(samples);


%% logLikelihood test
N = 1;
samples = mvn1.sampleDistribution(N);
loglik1 = 0;
non_lik1 = 1;

for i=1:N
  non_lik1 = non_lik1 * mvn1.pdf(samples(:,i));
  loglik1 = loglik1 + mvn1.logpdf(samples(:,i));
end

loglik2 = mvn1.loglikelihood(samples);

%% Entropy computation
entropy1 = mvn1.computeEntropy();
entropy2 = mvn2.computeEntropy();

