set(0,'DefaultFigureWindowStyle','docked');
close all; clc;
% rng('Default'); % for testing purposes

%% Initialize HMM
number_states = 4;
p = .2;
transition = [p (1-p) 0 0; 0 p (1-p) 0; 0 0 p (1-p); (1-p) 0 0 p];
observation(1) = MultivariateNormalDistribution(2,[0; 0],eye(2)/4);
observation(2) = MultivariateNormalDistribution(2,[0 ; 5],eye(2)/4);
observation(3) = MultivariateNormalDistribution(2,[5; 5],eye(2)/4);
observation(4) = MultivariateNormalDistribution(2,[5 ; 0],eye(2)/4);
prior = [1 0 0 0]';

hmm1 = HiddenMarkovModel(number_states, transition, observation, prior);

%% Sample HMM and plot
N = 25;
[obs_s, state_s] = hmm1.sampleHMM(N);

figure; 
subplot(1,2,1); hold on; axis equal;
scatter(obs_s(1,:), obs_s(2,:),[], state_s, 'filled');
plot(obs_s(1,:), obs_s(2,:))
for i=1:hmm1.N
  hmm1.O(i).plot2d(2,i);
end
axis([-2 7 -2 7])
%% Test Viterbi

estimated_path = hmm1.viterbi(obs_s);
label = num2str(estimated_path); c = cellstr(label);
text(obs_s(1,:)+.1, obs_s(2,:)+.1, c);
title('Original HMM')

%% Test Baum Welch

N = 100;
S = 8;
obs_s_training = cell(S,1);
state_s_training = cell(S,1);

for s=1:S
  [obs_s_training{s}, state_s_training{s}] = hmm1.sampleHMM(N);
end

% initialize hmm2 with dummy parameters
number_states = 4;
transition = ones(4)*.25;
mu_start = rand(2,4)*11 - 3;
observation(1) = MultivariateNormalDistribution(2,mu_start(:,1),eye(2));
observation(2) = MultivariateNormalDistribution(2,mu_start(:,2),eye(2));
observation(3) = MultivariateNormalDistribution(2,mu_start(:,3),eye(2));
observation(4) = MultivariateNormalDistribution(2,mu_start(:,4),eye(2));
prior = ones(4,1)*.25;

hmm2 = HiddenMarkovModel(number_states, transition, observation, prior);

loglik = hmm2.baumwelch(obs_s_training, 0, 50);

% plot info from trained hmm

subplot(1,2,2); hold on; axis equal;
scatter(obs_s(1,:), obs_s(2,:),[], state_s, 'filled');
plot(obs_s(1,:), obs_s(2,:))
for i=1:hmm2.N
  hmm2.O(i).plot2d(2,i);
end
axis([-2 7 -2 7])
% perform Viterbi on the trained hmm
estimated_path2 = hmm2.viterbi(obs_s);
label = num2str(estimated_path2); c = cellstr(label);
text(obs_s(1,:)+.1, obs_s(2,:)+.1, c);
title('Trained HMM')

figure
plot(loglik(2:end))
title('log p(d|\theta) for the standard BW case')

%% Test PO Baum Welch

% initialize hmm2 with dummy parameters
number_states = 4;
transition = ones(4)*.25;
observation(1) = MultivariateNormalDistribution(2,mu_start(:,1),eye(2));
observation(2) = MultivariateNormalDistribution(2,mu_start(:,2),eye(2));
observation(3) = MultivariateNormalDistribution(2,mu_start(:,3),eye(2));
observation(4) = MultivariateNormalDistribution(2,mu_start(:,4),eye(2));
prior = ones(4,1)*.25;

hmm2 = HiddenMarkovModel(number_states, transition, observation, prior);

% make the state sequence only partially known

% probability of not observing a state
partial_state_s_training = cell(length(state_s_training));
prob_not_observable = 0.8;

for s=1:length(state_s_training)
  partial_state_s_training{s} = state_s_training{s};
  mask = binornd(ones(size(state_s_training{s})), prob_not_observable);
  partial_state_s_training{s}(mask == 1) = NaN;
end

% create state evidence
state_evidence = cell(length(state_s_training), 1);
for s=1:length(partial_state_s_training)
  state_evidence{s} = zeros(number_states,...
    length(partial_state_s_training{s}));
  for t=1:length(state_evidence{s})
    if isnan(partial_state_s_training{s}(t))
      state_evidence{s}(:,t) = normalize(ones(number_states,1));
    else
      state_evidence{s}(partial_state_s_training{s}(t),t) = 1;
    end
  end
end

% PObaumwelch training
loglik = hmm2.PObaumwelch(obs_s_training, state_evidence, 0, 50);

% plot info from original hmm
figure;
subplot(1,2,1); hold on; axis equal;
scatter(obs_s(1,:), obs_s(2,:),[], state_s, 'filled');
plot(obs_s(1,:), obs_s(2,:))
for i=1:hmm1.N
  hmm1.O(i).plot2d(2,i);
end
axis([-2 7 -2 7])
label = num2str(estimated_path); c = cellstr(label);
text(obs_s(1,:)+.1, obs_s(2,:)+.1, c);
title('Original HMM')

% plot info from trained hmm
subplot(1,2,2); hold on; axis equal;
scatter(obs_s(1,:), obs_s(2,:),[], state_s, 'filled');
plot(obs_s(1,:), obs_s(2,:))
for i=1:hmm2.N
  hmm2.O(i).plot2d(2,i);
end
axis([-2 7 -2 7])
% perform Viterbi on the trained hmm
estimated_path2 = hmm2.viterbi(obs_s);
label = num2str(estimated_path2); c = cellstr(label);
text(obs_s(1,:)+.1, obs_s(2,:)+.1, c);
title('HMM trained with partial labeled observations')

figure
plot(loglik(2:end))
title('log p(d|\theta) for the Partially Observable BW case')