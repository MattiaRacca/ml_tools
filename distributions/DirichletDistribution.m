classdef DirichletDistribution < handle
  %DIRICHLETDISTRIBUTION Class implementation of Dirichlet
  % Requires FastFit
  
  properties
    %% Parameters
    D           % Number of variables/dimensions
    alpha       % Dirichlet parameters (column vector)
  end
  properties (SetAccess = private)
    mean
    mode
    variance
    entropy
  end
  
  methods
    %% Constructor
    function obj = DirichletDistribution(dimension_number, alpha)
      if nargin == 0
        % empty constructor
      end
      if nargin >= 1
        obj.D = dimension_number;
        if nargin >= 2
          obj.alpha = alpha;
        else
          obj.alpha = ones(obj.D, 1);
        end
      end
    end
    
    %% Deep Copy
    function deep_copy = copy(this)
      % Instantiate new object of the same class.
      deep_copy = feval(class(this));
      
      % Copy all non-hidden properties.
      p = properties(this);
      for i = 1:length(p)
        deep_copy.(p{i}) = this.(p{i});
      end
    end
    
    %% Setters
    function set.D(obj, dimension_number)
      if dimension_number > 1
        obj.D = dimension_number;
      else
        error('At least two dimensions please')
      end
    end
    function set.alpha(obj, alpha)
      if length(alpha) == obj.D
        if(size(alpha,1) < size(alpha,2))
          alpha = alpha';
        end
        obj.alpha = alpha;
        obj.updateMoments();
        obj.computeEntropy();
      else
        error('Alpha must be a Dx1 vector');
      end
    end
    %% Sample
    function samples = sampleDistribution(obj, N)
      % returns a DxN matrix, samples are on colums
      samples = dirichlet_sample(obj.alpha, N);
    end
    
    %% PDF
    function prob = pdf(obj,data)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of point to evaluate
      prob = exp(dirichlet_logProb(obj.alpha, data));
    end
    %% logPDF
    function logprob = logpdf(obj,data)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of point to evaluate
      logprob = dirichlet_logProb(obj.alpha, data);
    end
    %% logLikelihood
    function loglik = loglikelihood(obj, data)
      loglik = sum(obj.logpdf(data));
    end
    
    %% MLE
    function MLE(obj, data, initial_guess)
      % data is a NxD matrix
      if size(data,2) ~= obj.D
        data = data';
      end
      if nargin < 3
         [mlealpha] = dirichlet_fit(data);
      else
        if length(initial_guess) == obj.D
          [mlealpha] = dirichlet_fit(data, initial_guess');
        else
          error('Initial guess must be Dx1');
        end
      end
      obj.alpha = mlealpha';
    end
    %% Weighted MLE
    function weightedMLE(obj, data, weights, initial_guess)
      % data should be a NxD matrix
      if size(data,2) ~= obj.D
        data = data';
      end
      % weights is a column vector
      if size(weights,2) ~= 1
        weights = weights';
      end
      
      if nargin < 4
         [mlealpha] = dirichlet_weight_fit(data, weights);
      else
        if length(initial_guess) == obj.D
          [mlealpha] = dirichlet_fit(data, weights, initial_guess');
        else
          error('Initial guess must be Dx1');
        end
      end
      obj.alpha = mlealpha';
    end
    %% Plot (D = 3)
    % Inspired by
    % https://se.mathworks.com/matlabcentral/newsreader/view_thread/139363
    function h = plotDistribution(obj, log)
      if (obj.D ~= 3)
        error('Plot supported only for 3 dimensions')
      else
        step = 301;
        x1 = linspace(0,1,step);
        x2 = linspace(0,1,step);
        [X1,X2] = ndgrid(x1,x2);
        X3 = 1 - X1 - X2;
        bad = (X1+X2 > 1);
        X1(bad) = NaN; X2(bad) = NaN; X3(bad) = NaN;
        
        if nargin < 2
          F = obj.pdf([reshape(X1,1, step^2);reshape(X2,1, step^2);...
            reshape(X3,1, step^2)]);
        else
          if log
            F = obj.logpdf([reshape(X1,1, step^2);reshape(X2,1, step^2);...
              reshape(X3,1, step^2)]);
          end
        end
        F = reshape(F, step,step);
        F = real(F);
        
        gamma = sqrt(1 - 0.5^2);
        
        h = surf(X1*gamma,X2 + repmat(linspace(0,0.5,step)',1,step),F, ...
          'EdgeColor', 'none');
        xlim([0,1]); ylim([0,1]);
        axis equal;
        view(-90,90); axis off; grid off
        % text(-0.02,-0.02,0, 'A3')
        % text(-0.02,+1.02,0, 'A2')
        % text(gamma + 0.02,+0.51,0, 'A1')
      end
    end
    
    %% update Moments (mean, variance + mode)
    function updateMoments(obj)
      alpha0 = sum(obj.alpha);
      obj.mean = obj.alpha/alpha0;
%       if obj.alpha == ones(obj.D,1)
%         warning('mode undefined for uniform case')
%       end
      obj.mode = (obj.alpha - 1)/(alpha0 - obj.D);
      var = obj.alpha.*(repmat(alpha0, obj.D,1) - obj.alpha);
      obj.variance = var/(alpha0^2*(alpha0+1));
    end
    
    %% Entropy
    function entropy = computeEntropy(obj)
      alpha0 = sum(obj.alpha);
      entropy = logBeta(obj.alpha) + (alpha0 - obj.D)*...
        digamma(alpha0) - sum((obj.alpha - 1).*digamma(obj.alpha));
      obj.entropy = entropy;
    end
    
    %% KL divergence
    % from http://bariskurt.com/kullback-leibler-divergence-between-
    % two-dirichlet-and-beta-distributions/
    function divergence = computeKLDivergence(obj, dir2)
      alph = obj.alpha;
      beta = dir2.alpha;
      
      alph0 = sum(alph);
      beta0 = sum(beta);
      
      divergence = gammaln(alph0) - gammaln(beta0)...
        - sum(gammaln(alph)) + sum(gammaln(beta)) + (alph - beta)'...
        * (digamma(alph) - digamma(alph0));
    end
    
    %% KL divergence of mean of Dirichlet (KL between Categorical)
    function divergence = computeKLCategorical(obj, dir2)
      p = obj.mean;
      q = dir2.mean;
      
      divergence = p'*log(p) - p'*log(q);
    end
  end
end

%% REFERENCES
%   Fastfit - https://github.com/tminka/fastfit
