set(0,'DefaultFigureWindowStyle','docked');
close all; clc;
rng('Default'); % for testing purposes

%% Test Entropy Approximation
% from Huber08, Section VI.B


%% Initialization
K = 5;
D = 2;
w = ones(5,1)*.2;

eval_point = 100;
c = linspace(-3,3, eval_point); % scalar parameter that varies 5° component mean
H_approx = zeros(eval_point,1);
H_true = zeros(eval_point,1);
H_LB = zeros(eval_point,1);
H_UB = zeros(eval_point,1);

mvn(1) = MultivariateNormalDistribution(D, [0,0]', diag([.16,1]));
mvn(2) = MultivariateNormalDistribution(D, [3,2]', diag([1,.16]));
mvn(3) = MultivariateNormalDistribution(D, [1,-0.5]', eye(2)*.5);
mvn(4) = MultivariateNormalDistribution(D, [2.5,1.5]', eye(2)*.5);

%% Entropy computation
if (~exist('data/GMM_entropy.mat','file'))
  for i = 1:length(c)
    % create/shift 5° component mean
    mvn(5) = MultivariateNormalDistribution(D, [1,1]'*c(i), eye(2)*.5);
    % create gmm
    gmm = GaussianMixtureModel(K, w, mvn);
    
    % compute entropy (with MC sampling)
    H_true(i) = gmm.computeEntropyMC(10000);
  end
  save('data/GMM_entropy.mat', 'H_true');
else
  load('data/GMM_entropy.mat', 'H_true');
end

for i = 1:length(c)
  % create/shift 5° component mean
  mvn(5) = MultivariateNormalDistribution(D, [1,1]'*c(i), eye(2)*.5);
  % create gmm
  gmm = GaussianMixtureModel(K, w, mvn);

  % compute entropy approximation
  H_approx(i) = gmm.computeEntropyApproximation();
  
  % compute entropy LB
  H_LB(i) = gmm.computeEntropyLB();
  
  % compute entropy loose UB
  H_UB(i) = gmm.computeEntropyLooseUB();
end

%% Plotting
figure; hold on;
plot(c, H_approx, 'g--');
plot(c, H_true, 'k');
plot(c, H_LB, 'k--');
plot(c, H_UB, 'k--');
legend('Huber08 entropy approximation', 'MC sampling entropy',...
  'Lower Bound', 'Loose Upper Bound')
