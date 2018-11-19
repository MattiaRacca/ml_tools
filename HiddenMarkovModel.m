classdef HiddenMarkovModel < handle
  %HIDDENMARKOVMODEL Class implementation of HMM
  
  properties
    N   % Number of States
    A   % Transition Probabilities
    O   % Observation Probabilities (Set of distributions)
    P   % State priors (column vector)
  end
  
  methods
    %% Constructor
    function obj = HiddenMarkovModel(number_states, transition,...
        observation, prior)
      if(nargin > 0)
        obj.N = number_states;
        obj.A = transition;
        obj.O = observation;
        obj.P = prior;
      end
    end
    
    %% Setters
    function set.N(obj, number_states)
      if number_states > 0
        obj.N = number_states;
      else
        error('Number of states should be stricly positive')
      end
    end
    function set.A(obj, transition)
      if (prod(size(transition) == [obj.N obj.N]))
        if size(uniquetol([sum(transition,2)'; ones(1,obj.N)],...
            'ByRows', true),1) == 1
          % check if sum over rows == 1 for each row, within tolerance
          obj.A = normalize(transition,2);
        else
          error('transition matrix rows must sum to one');
        end
      else
        error('Size of transition matrix must be NxN')
      end
    end
    function set.O(obj, observation)
      if(length(observation) == obj.N)
        obj.O = observation;
      else
        error('Size of observation distribution set must be N')
      end
    end
    function set.P(obj, prior)
      if(length(prior) == obj.N && ...
          length(uniquetol([sum(prior); 1], 'ByRows', true)) == 1)
        % are the prior summing to 1? within machine tolerance
        obj.P = normalize(prior);
      else
        error('Size of prior over state must be N and sum to 1');
      end
    end
    
    %% HMM Sampling
    function [obs_seq, state_seq] = sampleHMM(obj, samples_number)
      % obs_seq and state_seq are DxN matrix, where D is the dimensionality
      % of the observation set (D of the MVNs) and N is the samples_number
      
      obs_seq = zeros(obj.O(obj.N).D, samples_number);
      state_seq = zeros(samples_number,1);
      if (samples_number > 1)
        % sample the starting state from state prior (multinoulli sample)
        state_seq(1) = find(mnrnd(1, obj.P/sum(obj.P)));
        for t=1:(samples_number-1)
          % generate observation
          obs_seq(:,t) = obj.O(state_seq(t)).sampleDistribution(1);
          % transit to new state according to A
          state_seq(t+1) = find(mnrnd(1, obj.A(state_seq(t),:)));
        end
        obs_seq(:,samples_number) = obj.O(state_seq(samples_number))...
          .sampleDistribution(1);
      end
    end
    
    %% Compute logp(data|model) (data log probabily for each state)
    function [logb, scale] = computeLogB(obj, O, map)
      logb = zeros(obj.N,size(O,2));
      if map
        for i=1:obj.N
          logb(i,:) = obj.O(i).logPosteriorPredictive(O);
        end
      else
        for i=1:obj.N
          logb(i,:) = obj.O(i).logpdf(O);
        end
      end
      [logb, scale] = normalizeLogspace(logb);
    end
    
    %% Scaled Forward Procedure
    % from Murphy12, 17.4.2
    function [loglik, alpha] = scaledForwardProcedure(obj, obs_seq, p_obs)
      loglik = 0;
      if(~isempty(obs_seq))
        alpha = zeros(obj.N,length(obs_seq));
        s = zeros(length(obs_seq), 1);
        
        % compute p(data|state) if not provided
        if(nargin < 3)
          % use logprob to avoid numerical underflow
          % inspired by https://github.com/probml/pmtk3/blob/master/...
          %  toolbox/LatentVariableModels/hmm/sub/hmmFitEm.m
          [logp_obs, scale] = obj.computeLogB(obs_seq, 0);
          loglik = loglik + sum(scale);
          p_obs = exp(logp_obs);
        end
        
        % initialize forward variable alpha
        [alpha(:,1), s(1)] = normalize(obj.P.*p_obs(:,1));
        
        % induction
        for t=2:length(obs_seq)
          % Eq. 17.48
          [alpha(:,t), s(t)] = normalize(p_obs(:,t).*...
            (obj.A'*alpha(:,t-1)));
        end
        
        % loglik data computation
        % Eq. 17.49
        loglik = loglik + sum(log(s + eps));
      end
    end
    %% Scaled Evidence Forward Procedure
    % from Murphy12, 17.4.2 and Ramasso14
    function [loglik, alpha] = scaledEvidenceForwardProcedure(obj,...
        obs_seq, p_obs, st_ev)
      loglik = 0;
      if(~isempty(obs_seq))
        alpha = zeros(obj.N,length(obs_seq));
        s = zeros(length(obs_seq), 1);
        
        % compute p(data|state) if not provided
        if(nargin < 3)
          % use logprob to avoid numerical underflow
          % inspired by https://github.com/probml/pmtk3/blob/master/...
          %  toolbox/LatentVariableModels/hmm/sub/hmmFitEm.m
          [logp_obs, scale] = obj.computeLogB(obs_seq, 0);
          loglik = loglik + sum(scale);
          p_obs = exp(logp_obs);
        end
        
        % initialize forward variable alpha (Ramasso14, Eq. 28a)
        [alpha(:,1), s(1)] = normalize(obj.P.*st_ev(:,1).*p_obs(:,1));
        
        % induction
        for t=2:length(obs_seq)
          % Ramasso14, Eq. 28b
          [alpha(:,t), s(t)] = normalize(p_obs(:,t).*st_ev(:,t).*...
            (obj.A'*alpha(:,t-1)));
          if alpha(:,t) == zeros(obj.N, 1)
            warning('alpha instance is zeroed');
          end
        end
        
        % loglik data computation
        % Eq. 17.49
        loglik = loglik + sum(log(s + eps));
      end
    end
    
    %% Scaled Backward Procedure
    % from Murphy12, 17.4.3.1
    function [beta] = scaledBackwardProcedure(obj, obs_seq, p_obs)
      if(~isempty(obs_seq))
        beta = zeros(obj.N, length(obs_seq));
        
        % TODO: this is not compliant with the rest
        % it is using the old prob (not log version)
        % compute p(data|state) if not provided
        if(nargin < 3)
          p_obs = obj.computeB(obs_seq, 0);
        end
        
        % initialize beta Eq. 17.60
        beta(:,end) = ones(obj.N,1);
        
        % induction
        for t=length(obs_seq)-1:-1:1
          % from Murphy Eq. 17.59
          beta(:,t) = normalize(obj.A*(p_obs(:,t+1).*beta(:,t+1)));
        end
      else
        beta = 0;
      end
    end
    %% Scaled Evidence Backward Procedure
    % from Murphy12, 17.4.3.1 and Ramasso14
    function [beta] = scaledEvidenceBackwardProcedure(obj,...
        obs_seq, p_obs, st_ev)
      if(~isempty(obs_seq))
        beta = zeros(obj.N, length(obs_seq));
        
        % TODO: this is not compliant with the rest
        % it is using the old prob (not log version)
        % compute p(data|state) if not provided
        %         if(nargin < 3)
        %           p_obs = obj.computeB(obs_seq, 0);
        %         end
       
        % Ramasso14, Eq 29a
        beta(:,end) = ones(obj.N,1);
        
        % induction
        for t=length(obs_seq)-1:-1:1
          % Ramasso14, Eq 29b
          beta(:,t) = normalize(obj.A*(...
            st_ev(:,t+1).*p_obs(:,t+1).*beta(:,t+1)));
          if beta(:,t) == zeros(obj.N, 1)
            warning('beta instance is zeroed');
          end
        end
      else
        beta = 0;
      end
    end
    
    %% Viterbi Algorithm
    % from Rabiner89
    % log version, to avoid underflow (see Murphy12 17.4.4.2)
    % TODO: make it work for several observation sequences
    function state_seq_star = viterbi(obj, obs_seq)
      if(~isempty(obs_seq))
        T = length(obs_seq);
        delta = zeros(obj.N, T);
        psi = zeros(obj.N, T);
        state_seq_star = zeros(T,1);
        
        % compute all logprob
        logp = obj.computeLogB(obs_seq, 0);
        logA = log(obj.A);
        
        % initialization (Eq. 32)
        delta(:,1) = log(obj.P) + logp(:,1);
        
        % recursion (Eq. 33)
        for t=2:T
          delta(:,t) = max(logA + repmat(delta(:,t-1), 1, obj.N))'...
            + logp(:,t);
          for j=1:obj.N
            [~, psi(j,t)] = max(logA(:,j) + delta(:,t-1));
          end
        end
        
        % termination (Eq. 34b)
        [~, state_seq_star(T)] = max(delta(:,T));
        
        % path backtracking
        for t=T-1:-1:1
          state_seq_star(t) = psi(state_seq_star(t+1),t+1);
        end
      end
    end
    
    %% Baum-Welch Algorithm
    % from Rabiner89 and Murphy12
    function [loglik] = baumwelch(obj, obs, MAP, MAX_ITER)
      if(~isempty(obs))
        %% Process input
        if(~iscell(obs))
          % enable to handle the single obs sequence case with the multiple
          % case implementation
          tmp = obs;
          obs = cell(1);
          obs{1} = tmp;
        end
        
        % By default perform ML estimation in the E-step
        if nargin >= 3
          map = MAP;
        else
          map = 0;
        end
        % By default runs until convergence (no max_iter)
        if nargin >= 4
          max_iter = MAX_ITER;
        else
          max_iter = Inf;
        end
        
        %% Start Algorithm
        
        iter = 1; % iteration counter
        loglik(iter) = -Inf; iter = iter + 1;
        
        %% E step    
        [loglik(iter), xi_sum, gamma, gamma_start] = obj.EStep(obs, map);  
        
        % Iterate until convergence
        while and(loglik(iter) - loglik(iter-1) > eps,iter < max_iter)
          iter = iter + 1;
          
          %% M step
          obj.MStep(xi_sum, gamma, gamma_start, obs, map);
          %% E step
          [loglik(iter), xi_sum, gamma, gamma_start] = obj.EStep(obs, map);       
        end
      else
        % Empty observation sequence case...
        loglik = -Inf;
      end
    end
    
    %% Baum-Welch Algorithm for partially labeled observation sequences
    % from Rabiner89, Murphy12 and Ramasso14
    function [loglik] = PObaumwelch(obj, obs, state_evidence, ...
        MAP, MAX_ITER)
      if(~isempty(obs))
        %% Process input
        if(~iscell(obs))
          % enable to handle the single obs sequence case with the multiple
          % case implementation
          tmp = obs;
          tmp2 = state_evidence;
          obs = cell(1);
          state_evidence = cell(1);
          obs{1} = tmp;
          state_evidence{1} = tmp2;
        end
        
        % TODO: control size of state_evidence{n}
        
        % By default perform ML estimation in the E-step
        if nargin >= 4
          map = MAP;
        else
          map = 0;
        end
        % By default runs until convergence (no max_iter)
        if nargin >= 5
          max_iter = MAX_ITER;
        else
          max_iter = Inf;
        end
        
        %% Start Algorithm
        
        iter = 1; % iteration counter
        loglik(iter) = -Inf; iter = iter + 1;
        
        %% E step    
        [loglik(iter), xi_sum, gamma, gamma_start] = obj.EStep(obs,...
          map, state_evidence);  
        
        % Iterate until convergence
        while and(loglik(iter) - loglik(iter-1) > eps,iter < max_iter)
          iter = iter + 1;
          
          %% M step
          obj.MStep(xi_sum, gamma, gamma_start, obs, map);
          %% E step
          [loglik(iter), xi_sum, gamma, gamma_start] = obj.EStep(obs,...
            map, state_evidence);       
        end
      else
        % Empty observation sequence case...
        loglik = -Inf;
      end
    end
    
    %% E step
    % from Murphy12, 17.5.2.1
    function [loglik, xi_sum, gamma, gamma_start] = EStep(obj, obs,...
        map, state_evidence)
      loglik = 0;
      alpha = cell(length(obs),1);
      p_obs = cell(length(obs),1);
      logp_obs = cell(length(obs),1);
      beta = cell(length(obs),1);
      gamma = cell(length(obs),1);
      xi_sum = zeros(obj.N,obj.N);
      gamma_start = zeros(obj.N,1);
      
      evidence = 0;
      if nargin > 3
        % the evidence for each state at each timestep is provided
        evidence = 1;
      end
      
      % compute p(data|state)
      for n=1:length(obs)
        [logp_obs{n}, scale] = obj.computeLogB(obs{n}, map);
        loglik = loglik + sum(scale);
        p_obs{n} = exp(logp_obs{n});
      end
      
      % forward pass
      for n=1:length(obs)
        if not(evidence) 
          [loglik_partial, alpha{n}] = scaledForwardProcedure(obj, ...
            obs{n}, p_obs{n});
        else
          [loglik_partial, alpha{n}] = scaledEvidenceForwardProcedure...
            (obj, obs{n}, p_obs{n}, state_evidence{n});
        end
        loglik = loglik + loglik_partial;
      end
      
      % backward pass and expected counts
      for n=1:length(obs)
        if not(evidence)
          [beta{n}] = scaledBackwardProcedure(obj, obs{n}, p_obs{n});
        else
          [beta{n}] = scaledEvidenceBackwardProcedure(obj, obs{n},...
            p_obs{n}, state_evidence{n});
        end
        [xi_s, gamma{n}] = obj.computeXiGamma(alpha{n}, beta{n}, ...
          obs{n}, p_obs{n});
        xi_sum = xi_sum + xi_s; % Eq. 17.99, sum over sequences
        gamma_start = gamma_start + gamma{n}(:,1); % Eq. 17.98
      end
    end
    
    %% M step
    % from Murphy12, 17.5.2.2
    % TODO: make the M step able to do full MAP (also prior and A)
    function MStep(obj, xi_sum, gamma, gamma_start, obs, map)
      obj.updatePrior(gamma_start); % Eq. 17.103
      obj.updateTransition(xi_sum); % Eq. 17.103
      obj.updateObservation(gamma, obs, map); % Eq. 17.106-108
    end
    
    %% Xi & Gamma computation
    % from Murphy12, 17.4.3.1-2
    function [xi_sum, gamma] = computeXiGamma(obj, alpha, beta, ...
        obs_seq, p_obs)
      T = length(obs_seq);
      xi_sum = zeros(obj.N, obj.N);
      
      % Eq. 17.53
      gamma = normalize(alpha .* beta, 1);
      
      for t = T-1:-1:1
        % Eq. 17.67
        b = beta(:,t+1) .* p_obs(:,t+1);
        xit = obj.A .* (alpha(:,t) * b');
        % Eq. 17.99, sum over time
        xi_sum = xi_sum + xit./sum(xit(:));
      end
    end
    
    %% Expected Prior update
    % from Rabiner89 (Eq. 40a)
    function updatePrior(obj, gamma_start)
      obj.P = normalize(gamma_start);
    end
    
    %% Transition probability update
    % from Rabiner89 (Eq. 40b)
    function updateTransition(obj, xi_sum)
        obj.A = normalize(xi_sum,2);
    end
    
    %% Observation probability update
    % from Rabiner89 and Murphy12
    function updateObservation(obj, gamma, obs_seq, map)
      gamma_conc = []; obs_conc = [];
      % concatenate gammas and observations
      for n=1:length(gamma)
        gamma_conc = [gamma_conc, gamma{n}];
        obs_conc = [obs_conc, obs_seq{n}];
      end
      for i=1:obj.N
        % update sufficient statistic for each obs dist separately
        if map
          obj.O(i).weightedMAP(gamma_conc(i,:), obs_conc);
        else
          obj.O(i).weightedMLE(gamma_conc(i,:), obs_conc);
        end
      end
    end
    
    
    %% OBSOLETE FUNCTIONS
    
    %% Forward Procedure
    % from Rabiner89 - not used because suffers underflow
    function [prob_obs_seq, alpha] = forwardProcedure(obj, obs_seq)
      prob_obs_seq = 0;
      if(~isempty(obs_seq))
        alpha = zeros(obj.N,length(obs_seq));
        
        % initialize forward variable alpha (Eq. 19)
        alpha(:,1) = obj.P.*obj.computeB(obs_seq(:,1));
        
        % induction (Eq. 20)
        for t=2:length(obs_seq)
          alpha(:,t) = sum(repmat(alpha(:,t-1),1,obj.N).*obj.A, 1)'.*...
            obj.computeB(obs_seq(:,t));
        end
        
        % termination (Eq. 21)
        prob_obs_seq = sum(alpha(:, length(obs_seq), 1));
      end
    end
    %% Backward Procedure
    % from Rabiner89 - not used because suffers underflow
    function beta = backwardProcedure(obj, obs_seq)
      if(~isempty(obs_seq))
        beta = zeros(obj.N, length(obs_seq));
        
        % initialize beta (backward variable) (Eq. 24)
        beta(:,end) = ones(obj.N,1);
        
        % optimize version (Eq. 25)
        for t=length(obs_seq)-1:-1:1
          beta(:,t) = sum(obj.A.*...
            repmat(obj.computeB(obs_seq(:,t+1))',obj.N,1).*...
            repmat(beta(:,t+1)',obj.N,1),2);
        end
      else
        beta = 0;
      end
    end
    
    %% Compute p(data|model) (data probabily for each state)
    function b = computeB(obj, O, map)
      if nargin < 3
        map = 0;
      end
      b = zeros(obj.N,size(O,2));
      if map
        for i=1:obj.N
          b(i,:) = obj.O(i).PosteriorPredictive(O);
        end
      else
        for i=1:obj.N
          b(i,:) = obj.O(i).pdf(O);
        end
      end
    end
    
  end
end

%% REFERENCES
%   Rabiner89 - Rabiner, Lawrence R. "A tutorial on hidden Markov models 
%    and selected applications in speech recognition." Proceedings of the 
%    IEEE 77.2 (1989): 257-286.
%   Murphy12 - Murphy, Kevin P. "Machine learning: a probabilistic 
%    perspective." MIT press, 2012.
%   Ramasso14 - Ramasso, Emmanuel, and Thierry Denoeux. "Making use of 
%    partial knowledge about hidden states in HMMs: an approach based on 
%    belief functions." IEEE Transactions on Fuzzy Systems 22.2 (2014):
%    395-405.