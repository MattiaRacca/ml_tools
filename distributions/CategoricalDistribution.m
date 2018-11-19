classdef CategoricalDistribution < handle
  %CATEGORICALDISTRIBUTION Class implementation of Categorical
  %   Detailed explanation goes here
  
  properties
    %% Parameters
    D           % Number of Categories
    theta       % Caterories' probabilities (column vector)
    prior       % Dirichlet Distribution acting as prior
  end
  properties (SetAccess = private)
    posterior
  end
  
  methods
    %% Constructor
    function obj = CategoricalDistribution(category_number,...
        category_probabilities, prior_distribution)
      if nargin >= 1
        obj.D = category_number;
        obj.theta = ones(obj.D,1)/obj.D;  % if not specified, uniform distribution
      end
      if nargin >= 2
        obj.theta = category_probabilities;
      end
      if nargin >= 3
        obj.prior = prior_distribution;
      end
    end
    
    %% Setters
    function set.D(obj, category_number)
      if category_number > 0
        obj.D = category_number;
      else
        error('At least one category please')
      end
    end
    function set.theta(obj, category_probabilities)
      if(length(category_probabilities) == obj.D && ...
          length(uniquetol...
          ([sum(category_probabilities); 1], 'ByRows', true)) == 1)
        obj.theta = category_probabilities;
      else
        error('Theta must sum to 1')
      end
    end
    function set.prior(obj, prior_distribution)
      if isa(prior_distribution, 'DirichletDistribution')
        obj.prior = prior_distribution;
        % create/reset the posterior too
        obj.posterior = DirichletDistribution(obj.prior.D,...
          obj.prior.alpha);
      else
        error('Prior for Categorical must be Dirichlet')
      end
    end
    %% MLE
    function [theta] = MLE(obj, data, update_flag)
      % data is a Nxn matrix, where D is number of categories and n is
      % number of points for MLE
      [~,n] = size(data);
      theta = sum(data,2)/n;
      if nargin < 3
        obj.theta = theta;
      else
        if update_flag
          obj.theta = theta;
        end
      end
    end
    %% MAP
    % updates the categorical's theta with the posterior mode (MAP)
    function [theta] = MAP(obj)
      theta = obj.posterior.mode;
      obj.theta = theta;
    end
    %% UpdatePosterior
    % updates the posterior based on the new data
    % does not update the prior and the categorical's theta
    function [posterior] = updatePosterior(obj, data)
      counts = sum(data,2);
      obj.posterior.alpha = obj.prior.alpha + counts;
      posterior = obj.posterior;
    end
    %% UpdatePrior
    % updates the prior to the posterior
    % enables incremental update of the posterior
    function [new_prior] = updatePrior(obj)
       new_prior = obj.posterior.alpha;
       obj.prior.alpha = new_prior;
    end
    %% PosteriorPredictive
    % computes for each entry in data, the predictive probability
    % starting from the posterior
    function [predictive_prob] = posteriorPredictive(obj, data)
      % data must be Dxn
      alpha_sum = sum(obj.posterior.alpha);
      alpha_x = data'*obj.posterior.alpha;
      predictive_prob = alpha_x/alpha_sum;
    end
    %% Sampling
    function samples = sampleDistribution(obj, n)
      % generates a Dxn matrix, where N is number of categories and n is
      % number of trials
      if n > 0
        samples = mnrnd(1,obj.theta, n)';
      end
    end
  end
  
end

