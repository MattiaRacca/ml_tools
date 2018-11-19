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

%% Visualize for each state the object configuration (by sampling)

figure;

for i=1:number_states
  % sample MVN from state i
  samples = R_hmm.O(i).sampleDistribution(50);
  
  subplot(2,2,i); hold on; axis equal;
  scatter(samples(1,:), samples(2,:),[], 'b', 'filled');
  scatter(samples(3,:), samples(4,:),[], 'g', 'filled');
  scatter(samples(5,:), samples(6,:),[], 'm', 'filled');
  axis([-2, (sq+2), -2, (sq+2)]);
  t = sprintf('Item configuration in state %d', i);
  title(t);
end

%% Train HMM with Batch BW
N = 100;
S = 8;
obs_s_training = cell(S,1);
state_s_training = cell(S,1);

for s=1:S
  [obs_s_training{s}, state_s_training{s}] = R_hmm.sampleHMM(N);
end

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
prior = ones(4,1)*.25;

hmm_batch = HiddenMarkovModel(number_states,...
  transition, observation, prior);

loglik = hmm_batch.baumwelch(obs_s_training);

% perform Viterbi on the trained hmm
estimated_path_batch = hmm_batch.viterbi(obs_s_training{1});

%% Visualize for each state the object configuration (CASE: trained batch)
figure
plot(loglik(2:end))
title('log p(d|\theta) for the batch BW case')

figure;

for i=1:number_states
  state_samples = obs_s_training{1}(:, estimated_path_batch == i);
  subplot(2,2,i); hold on; axis equal;
  scatter(state_samples(1,:), state_samples(2,:),[], 'b', 'filled');
  scatter(state_samples(3,:), state_samples(4,:),[], 'g', 'filled');
  scatter(state_samples(5,:), state_samples(6,:),[], 'm', 'filled');
  axis([-2, (sq+2), -2, (sq+2)]);
  t = sprintf('Item configuration in state %d', i);
  title(t);
end

% The batch trained HMM inverts the states (state 1 is mistaken for 3,...)
% This could be solved in this case by constraining the initial state to 1
% in the prior over state.
% However, this 'trick' works for simple model like this left-right hmm
% while might fail for more complex model (complex in sense of transition
% probabilities, prior over state and observation probability)