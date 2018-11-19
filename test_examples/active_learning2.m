set(0,'DefaultFigureWindowStyle','docked');
close all; clc;
rng('Default'); % for testing purposes

%% Active Information Gathering for Task Modeling

%% create underlying HMM of the task
number_states = 4;
p = .95;
transition = [p (1-p) 0 0; 0 p (1-p) 0; 0 0 p (1-p); 0 0 0 1];

% Each state has an observartion probability of 2-d position of 3 objects
number_item = 3;

covariance_matrix{1} = blkdiag(eye(2)*.3, eye(2)*.3, eye(2)*.7);
covariance_matrix{2} = blkdiag(eye(2)*.3, eye(2), eye(2)*.5);
covariance_matrix{3} = blkdiag(eye(2), eye(2)*.5, eye(2));
covariance_matrix{4} = blkdiag(eye(2)*.3, eye(2)*.5, eye(2)*.5);

sq = 10; % square length

mean{1} = [0 0 0 0 sq sq]';
mean{2} = [0 0 0 sq sq sq]';
mean{3} = [sq sq 0 sq sq 0]';
mean{4} = [sq sq sq sq 0 0]';

for i=1:number_states
  observation(i) = MultivariateNormalDistribution(number_item*2, mean{i},...
    covariance_matrix{i});
end

prior = [1 0 0 0]';

% the real hmm
R_hmm = HiddenMarkovModel(number_states, transition, observation, prior);

%% Active Training for HMM

% initialize hmm_batch with dummy parameters
number_states = 4;
transition = ones(4)*.25;
mu_start = rand(6,4)*(sq + 1);
observation(1) = MultivariateNormalDistribution(number_item*2,...
  mu_start(:,1),eye(6)*2);
observation(2) = MultivariateNormalDistribution(number_item*2,...
  mu_start(:,2),eye(6)*2);
observation(3) = MultivariateNormalDistribution(number_item*2,...
  mu_start(:,3),eye(6)*2);
observation(4) = MultivariateNormalDistribution(number_item*2,...
  mu_start(:,4),eye(6)*2);
prior = [1 0 0 0]; 
% ASSUMPTION: we know that the task starts always on the first state

hmm_batch = HiddenMarkovModel(number_states,...
  transition, observation, prior);

