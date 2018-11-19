set(0,'DefaultFigureWindowStyle','docked');
clear all; close all; clc;
rng('Default'); % for testing purposes

%% Test class for Multivariate Student-T

%% 1-D (plus comparison with Gaussian)
mu = 4;
covariance = 2;

for nu=1:5
  mvst(nu) = MultivariateTStudentDistribution(1, mu, covariance, nu);
end

figure; hold on;
for i=1:length(mvst)
  ev = mvst(i).pdf([-4:0.1:12]);
  plot([-4:0.1:12], ev, 'b');
end
mvn = MultivariateNormalDistribution(1, mu, covariance);
ev = mvn.pdf([-4:0.1:12]);
plot([-4:0.1:12], ev, 'r');

%% 2-D
% covariance = gallery('randsvd',2, -2);
covariance = eye(2)*0.1;
mvst1 = MultivariateTStudentDistribution(2,[0;0],covariance, 2);
mvn1 = MultivariateNormalDistribution(2,[0;0],covariance);

%% Draw PDF
figure; hold on;
[X,Y] = meshgrid(-0.5:0.01:0.5,-0.5:0.01:0.5);
% [X,Y] = meshgrid(-10:0.1:10,-10:0.1:10);

Pst = mvst1.pdf([X(:)'; Y(:)']);
Pn = mvn1.pdf([X(:)'; Y(:)']);

% subplot(1,2,1)
% axis equal;
surf(X,Y,vec2mat(Pn, size(X,2))');
alpha(0.25);
% subplot(1,2,2)
axis equal;
surf(X,Y,vec2mat(Pst, size(X,2))');