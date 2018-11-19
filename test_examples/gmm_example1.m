set(0,'DefaultFigureWindowStyle','docked');
close all; clc;

%% Usage and tests of GMM Class

% create a 2D 2-component GMM

K = 3;
D = 2;
w = [.4 .4 .2]';
mvns(1) = MultivariateNormalDistribution(D, [+2,+1]', eye(D));
mvns(2) = MultivariateNormalDistribution(D, [-1,-2]', eye(D));
mvns(3) = MultivariateNormalDistribution(D, [-2,+1]', eye(D));

gmm1 = GaussianMixtureModel(K, w, mvns);

%% Check if logpdf and pdf give same result
pdf = gmm1.pdf(gmm1.components(1).mu);
logpdf = gmm1.logpdf(gmm1.components(1).mu);

%% Plot GMM pdf

[X,Y] = meshgrid(-5:0.1:5,-5:0.1:5);
P = gmm1.pdf([X(:)'; Y(:)']);
figure; 
subplot(1,2,1);hold on;
surf(X,Y,vec2mat(P, size(X,2))');
xlabel('x')
ylabel('y')
zlabel('z')

subplot(1,2,2);hold on; axis equal;
N = 200;
samples = gmm1.sampleMixture(N);
contour(X,Y, vec2mat(P, size(X,2))');
scatter(samples(1,:),samples(2,:),'.');
xlabel('x')
ylabel('y')

%% Test gradient evaluation
% create a 2D 2-component GMM
K = 3;
D = 2;
w = [.4 .4 .2]';
mvns(1) = MultivariateNormalDistribution(D, [+2,+1]', eye(D));
mvns(2) = MultivariateNormalDistribution(D, [-1,-2]', eye(D));
mvns(3) = MultivariateNormalDistribution(D, [-2,+1]', eye(D));

gmm2 = GaussianMixtureModel(K, w, mvns);

%% Plot gradient

[X,Y] = meshgrid(-5:0.3:5,-5:0.3:5);
P = gmm2.pdf([X(:)'; Y(:)']);
G = gmm2.evaluateGradient([X(:)'; Y(:)']);

figure; hold on; axis equal;
contour(X,Y, vec2mat(P, size(X,2))');
quiver(X,Y, vec2mat(G(1,:), size(X,2))', vec2mat(G(2,:), size(X,2))');

