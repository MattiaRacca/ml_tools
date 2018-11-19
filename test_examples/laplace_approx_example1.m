set(0,'DefaultFigureWindowStyle','docked');
close all; clc;
rng('Default'); % for testing purposes

%% Laplacian approximation of Student-t distribution

%% 1-D case
mu = 4;
covariance = 1;
nu = 3;

mvst = MultivariateTStudentDistribution(1, mu, covariance, nu);
mvn1 = mvst.laplaceApproximation();
mvn2 = MultivariateNormalDistribution(1, mu, covariance);

figure; hold on;
ev = mvst.pdf(-4:0.1:12);
plot(-4:0.1:12, ev, 'b');

ev = mvn1.pdf(-4:0.1:12);
plot(-4:0.1:12, ev, 'r');

ev = mvn2.pdf(-4:0.1:12);
plot(-4:0.1:12, ev, 'g');

legend('T distribution', 'Laplacian approx. Normal', 'Limit Normal')

%% KL divergence (with MC sampling)

disp('KL divergence: unidimension case');

% sample the normal distribution
% N = 100000;
% samples = mvn1.sampleDistribution(N);
% 
% sum_n = mvn1.loglikelihood(samples);
% sum_st= sum(mvst.logpdf(samples));
% 
% KL1 = (sum_n - sum_st)/N; disp(KL1)
% 
% samples = mvn2.sampleDistribution(N);
% 
% sum_n = mvn2.loglikelihood(samples);
% sum_st= sum(mvst.logpdf(samples));
% 
% KL2 = (sum_n - sum_st)/N; disp(KL2)

%% 2-D case

N = 2;
mu = [-9, 5]';
covariance = [1 0.4; 0.4 1];
nu = 3;
lambda = inv(covariance);

mvst = MultivariateTStudentDistribution(N,mu,covariance, nu);
mvn1 = mvst.laplaceApproximation();
mvn2 = MultivariateNormalDistribution(N,mu,covariance);

%% Draw PDF
[X,Y] = meshgrid(mu(1)-2:0.05:2+mu(1),mu(2)-2:0.05:2+mu(2));

Pst = mvst.logpdf([X(:)'; Y(:)']);
Pn_approx = mvn1.logpdf([X(:)'; Y(:)']);
Pn_same = mvn2.logpdf([X(:)'; Y(:)']);

figure; hold on;
% axis equal;
surf(X,Y,vec2mat(Pn_approx, size(X,2))');
alpha(0.25);
axis equal;
surf(X,Y,vec2mat(Pst, size(X,2))');
legend('Laplace approximation (MVN)', 'Multivariate T distr.')

figure; hold on;
surf(X,Y,vec2mat(Pn_same, size(X,2))');
alpha(0.25);
axis equal;
surf(X,Y,vec2mat(Pst, size(X,2))');
legend('MVN same mu and cov', 'T')

%% KL divergence (with MC sampling)

% disp('KL divergence: bidimensional case');
% 
% % sample the normal distribution
% N = 100000;
% samples = mvn1.sampleDistribution(N);
% 
% sum_n = mvn1.loglikelihood(samples);
% sum_st= sum(mvst.logpdf(samples));
% 
% KL1 = (sum_n - sum_st)/N; disp(KL1)
% 
% samples = mvn2.sampleDistribution(N);
% 
% sum_n = mvn2.loglikelihood(samples);
% sum_st= sum(mvst.logpdf(samples));
% 
% KL2 = (sum_n - sum_st)/N; disp(KL2)

%% KL divergence test: MVN approximation vs Laplace approximation

N = 2;
mu = [0, 0]';
covariance = eye(2);
nu_vect = [2:10, 20:10:100];
lambda = inv(covariance);
N_samples = 1000000;
kl_laplace = zeros(length(nu_vect),1);
kl_limit = zeros(length(nu_vect),1);

if (~exist('data/kl_test.mat','file'))
  for i=1:length(nu_vect)
    
    mvst_test = MultivariateTStudentDistribution(N,mu,covariance, nu_vect(i));
    mvn1 = mvst_test.laplaceApproximation();
    mvn2 = MultivariateNormalDistribution(N,mu,covariance);
    
    fprintf('KL diverge test. nu = %d\n', nu_vect(i));
    
    % KL(mvn || mvt)
    samples1 = mvn1.sampleDistribution(N_samples);
    
    sum_n = mvn1.loglikelihood(samples1);
    sum_st= sum(mvst_test.logpdf(samples1));
    
    kl_laplace(i) = (sum_n - sum_st)/N_samples; disp(kl_laplace(i))
    
    samples2 = mvn2.sampleDistribution(N_samples);
    
    sum_n = mvn2.loglikelihood(samples2);
    sum_st= sum(mvst_test.logpdf(samples2));
    
    kl_limit(i) = (sum_n - sum_st)/N_samples; disp(kl_limit(i))
    
  end
  
  save('data/kl_test.mat', 'kl_limit', 'kl_laplace');
else
  load('data/kl_test.mat',  'kl_limit', 'kl_laplace');
end

%% Plot KL divergence evolution

figure; hold on;
plot(2:10, kl_laplace(1:9));
plot(2:10, kl_limit(1:9));
legend('Laplace approximation', 'Limit approximation')
xlabel('nu')
ylabel('KL(mvn||mvt)')
figure; hold on;
plot(20:10:100, kl_laplace(10:end));
plot(20:10:100, kl_limit(10:end));
legend('Laplace approximation', 'Limit approximation')
xlabel('nu')
ylabel('KL(mvn||mvt)')