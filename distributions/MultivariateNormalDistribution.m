classdef MultivariateNormalDistribution < handle
  %MULTIVARIATENORMALDISTRIBUTION Class implementation of MVN
  %   Detailed explanation goes here
  
  properties
    %% Parameters
    D           % Number of Dimensions
    mu          % Mean vector (column vector)
    sigma       % Covariance matrix
    
    %% Hyperparameters (priors) [naming convention from Murphy12, 4.6.3.2]
    m0          % Prior mean for Mu
    kappa0      % Weight of m0
    S0          % Prior mean for Sigma
    nu0         % Weight of S0
  end
  properties (SetAccess  = private)
    sigma_det   % Determinant of Sigma
    lambda      % Precision matrix (inverse of sigma)
  end
  
  methods
    %% Constructor
    function obj = MultivariateNormalDistribution(dimensions, mean,...
        covariance, kappa, nu)
      if nargin >= 3
        %% No prior knowledge case
        obj.D = dimensions;
        obj.mu = mean;
        obj.sigma = covariance;
        
        obj.sigma_det = det(obj.sigma);
        obj.lambda = inv(obj.sigma);
      end
      if nargin == 5
        %% prior knowledge case
        obj.m0 = mean;
        obj.S0 = covariance;
        obj.kappa0 = kappa;
        obj.nu0 = nu;
      end
    end
    
    %% Setters
    function set.D(obj, dimensions)
      if dimensions > 0
        obj.D = dimensions;
      else
        error('Dimensions should be > 0')
      end
    end
    function set.mu(obj, mean)
      if(length(mean) == obj.D)
        obj.mu = mean;
      else
        error('Mean vector should be of size D')
      end
    end
    function set.sigma(obj, covariance)
      if(prod(size(covariance) == [obj.D obj.D]))
        obj.sigma = covariance;
        % check if sigma is positive semidefinite
        [~,p] = chol(obj.sigma);
        if(p)
          warning('Covariance matrix not found positive semidefinite');
          % obj.sigma = nearestSPD(obj.sigma);
        end
        % update determinant and inverse of covariance matrix
        obj.sigma_det = det(obj.sigma);
        obj.lambda = inv(obj.sigma);
      else
        error('Size of covariance matrix should be DxD');
      end
    end
    function set.m0(obj, m)
      if(length(m) == obj.D)
        obj.m0 = m;
      else
        error('Mean prior vector should be of size D')
      end
    end
    function set.S0(obj, c)
      if(prod(size(c) == [obj.D obj.D]))
        obj.S0 = c;
      else
        error('Covariance matrix prior should be DxD');
      end
    end
    function set.kappa0(obj, k)
      if k < 0
        error('Prior weight over mean must be positive');
      else
        obj.kappa0 = k;
      end
    end
    function set.nu0(obj, n)
      if n < 0
        error('Prior weight over covariance must be positive');
      else
        obj.nu0 = n;
      end
    end
    
    %% PDF
    function prob = pdf(obj, data)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of point to evaluate
      
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and MVN do not agree')
      else
        %% Naive implementation
        %  N = size(data, 2);
        %  prob = zeros(1,N);
        %  for i=1:N
        %    prob(i) = exp(-0.5*(data(:,i) - obj.mu)'*obj.lambda*...
        %      (data(:,i) - obj.mu))/((2*pi)^(obj.D/2)*obj.sigma_det^.5);
        %  end
        %% Numerically stable implementation
        prob = exp(logpdf(obj,data));
      end
    end
    
    %% logPDF
    function logprob = logpdf(obj,data)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of point to evaluate
      
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and MVN do not agree')
      else
        N = size(data, 2);
        logprob = zeros(1,N);
        for i=1:N
          logprob(i) = -0.5*(log(obj.sigma_det) + obj.D*log(2*pi) +...
            (data(:,i) - obj.mu)'*obj.lambda*(data(:,i) - obj.mu));
        end
      end
    end
    
    %% Likelihood
    % p(D|mu,sigma) from Murphy12, 4.6.3.1
    function lik = likelihood(obj, data)
      N = size(data,2);
      data_mean = mean(data,2);
      scatter_matrix = zeros(obj.D);
      for i=1:N
        scatter_matrix = scatter_matrix + (data(:,i) - data_mean)*...
          (data(:,i) - data_mean)';
      end
      lik = (2*pi)^(-N*obj.D*.5) * obj.sigma_det^(-N*.5) *...
        exp(-N*.5*(data_mean - obj.mu)'*obj.lambda*(data_mean - obj.mu))*...
        exp(-0.5*trace(obj.lambda*scatter_matrix));
    end
    %% LogLikelihood
    function loglik = loglikelihood(obj, data)    
      loglik = sum(obj.logpdf(data));
    end
    
    %% MLE
    % from Murphy12, 4.1.3
    function [mu, sigma] = MLE(obj, data, update_flag)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of points for MLE
      [obj.D, N] = size(data);
      mu = sum(data,2)./N;
      incr = zeros(obj.D, obj.D);
      for i=1:N
        incr = incr + data(:,i)*data(:,i)';
      end
      sigma = incr/N - mu*mu';
      
      if(nargin < 3)
        obj.mu = mu;
        obj.sigma = sigma;
      else
        if(update_flag)
          obj.mu = mu;
          obj.sigma = sigma;
        end
      end
    end
    %% weightedMLE
    function [mu, sigma] = weightedMLE(obj, w, data, update_flag)
      % w is a row vector of length = length(data)
      [obj.D, N] = size(data); % N is here number of samples, not dim.
      mu = data*w'/sum(w);
      incr = zeros(obj.D, obj.D);
      for i=1:N
        incr = incr + w(i)*data(:,i)*data(:,i)';
      end
      sigma = incr/sum(w) - mu*mu';
      
      if(nargin < 4)
        obj.mu = mu;
        obj.sigma = sigma;
      else
        if(update_flag)
          obj.mu = mu;
          obj.sigma = sigma;
        end
      end
    end
    
    %% Compute weighted posterior
    % from Murphy12, 11.4.2.8
    function [m_N, kappa_N, nu_N, S_N] = computeWeightedPosterior(obj,...
        w, data, update_flag)
      [dim, N] = size(data);
      if dim == obj.D
        sw = sum(w);
        data_mean = data*w'/sw;
        
        centered_scatter_matrix = zeros(obj.D);
        for i=1:N
          centered_scatter_matrix = centered_scatter_matrix ...
            + w(i)*(data(:,i) - data_mean)*(data(:,i) - data_mean)';
        end
        
        %% Eq. 11.43-47
        kappa_N = obj.kappa0 + sw;
        m_N = (obj.kappa0*obj.m0 + sw*data_mean)./(kappa_N);
        nu_N = obj.nu0 + sw;
        S_N  = obj.S0 + centered_scatter_matrix +...
          (obj.kappa0*sw/(obj.kappa0 + sw))*...
          (data_mean - obj.m0)*(data_mean - obj.m0)';
        
        %% Update prior values
        if(nargin < 4)
          obj.m0 = m_N;
          obj.kappa0 = kappa_N;
          obj.S0 = S_N;
          obj.nu0 = nu_N;
        else
          if(update_flag)
            obj.m0 = m_N;
            obj.kappa0 = kappa_N;
            obj.S0 = S_N;
            obj.nu0 = nu_N;
          end
        end
      else
        error('Data dimensionality is different from MVN dimensionality');
      end
    end
    %% Compute posterior
    % from Murphy12, 4.6.3.3
    function [m_N, kappa_N, nu_N, S_N] = computePosterior(obj, data, ...
        update_flag)
      [dim, N] = size(data);
      if dim == obj.D
        data_mean = mean(data,2);
        uncentered_scatter_matrix = zeros(obj.D);
        for i=1:N
          uncentered_scatter_matrix = uncentered_scatter_matrix ...
            + (data(:,i))*(data(:,i))';
        end
        
        %% Eq. 4.209-214
        kappa_N = obj.kappa0 + N;
        m_N = (obj.kappa0*obj.m0 + N*data_mean)./(kappa_N);
        nu_N = obj.nu0 + N;
        S_N  = obj.S0 + uncentered_scatter_matrix +...
          obj.kappa0*(obj.m0*obj.m0') - kappa_N*(m_N*m_N');
        
        %% Update prior values
        if(nargin < 3)
          obj.m0 = m_N;
          obj.kappa0 = kappa_N;
          obj.S0 = S_N;
          obj.nu0 = nu_N;
        else
          if(update_flag)
            obj.m0 = m_N;
            obj.kappa0 = kappa_N;
            obj.S0 = S_N;
            obj.nu0 = nu_N;
          end
        end
      else
        error('Data dimensionality is different from MVN dimensionality');
      end
    end
    
    %% MAP estimate (posterior mode)
    % from Murphy12, 4.6.3.4
    function [mu, sigma] = MAP(obj, data, update_flag)
      if nargin < 3
        [m_N, ~, nu_N, S_N] = computePosterior(obj, data);
        mu = m_N;
        sigma = S_N/(nu_N + obj.D + 2);
        
        obj.mu = mu;
        obj.sigma = sigma;
      else
        [m_N, ~, nu_N, S_N] = computePosterior(obj, data, update_flag);
        mu = m_N;
        sigma = S_N/(nu_N + obj.D + 2);
        
        if(update_flag)
          obj.mu = mu;
          obj.sigma = sigma;
        end
      end
    end
    %% Weighted MAP estimate (posterior mode)
    % from Murphy12, 4.6.3.4
    function [mu, sigma] = weightedMAP(obj, w, data, update_flag)
      if nargin < 4
        [m_N, ~, nu_N, S_N] = computeWeightedPosterior(obj, w, data);
        mu = m_N;
        sigma = S_N/(nu_N + obj.D + 2);
        
        obj.mu = mu;
        obj.sigma = sigma;
      else
        [m_N, ~, nu_N, S_N] = computeWeightedPosterior(obj,...
          w, data, update_flag);
        mu = m_N;
        sigma = S_N/(nu_N + obj.D + 2);
        
        if(update_flag)
          obj.mu = mu;
          obj.sigma = sigma;
        end
      end
    end
    
    %% Log Posterior predictive p(x|D)
    % from Murphy12, 4.6.3.6
    function predicted_prob = logPosteriorPredictive(obj, data)
      % Uses m0, S0 and so on, it assumes then that the computePosterior was
      % overwriting the old prior
      dof = obj.nu0 - obj.D +1;
      covariance = (obj.kappa0 + 1)*obj.S0/...
        (obj.kappa0*(dof));

      tst = MultivariateTStudentDistribution...
        (obj.D, obj.m0, covariance, dof);
      
      predicted_prob = tst.logpdf(data);
    end
    %% Posterior predictive p(x|D)
    % from Murphy12, 4.6.3.6
    function predicted_prob = posteriorPredictive(obj, data)
      % Uses m0, S0 and so on, it assumes then that the computePosterior was
      % overwriting the old prior
      dof = obj.nu0 - obj.D +1;
      covariance = (obj.kappa0 + 1)*obj.S0/...
        (obj.kappa0*(dof));
      
      tst = MultivariateTStudentDistribution...
        (obj.D, obj.m0, covariance, dof);
      
      predicted_prob = tst.pdf(data);
    end
    
    %% Entropy
    % from Bishop06, Appendix B.41
    function entropy = computeEntropy(obj)
      entropy = .5*log(obj.sigma_det) + .5*obj.D*(1 + log(2*pi));
    end
    %% Sampling
    function samples = sampleDistribution(obj, N)
      % generate DxN matrix, where D is dimensionality and N is
      % number of samples
      if N > 0
        samples = mvnrnd(obj.mu, obj.sigma, N)';
      end
    end
    
    %% Plot 2-D
    % from https://se.mathworks.com/matlabcentral/fileexchange/
    %  16543-plot-gaussian-ellipsoid/content/plot_gaussian_ellipsoid.m
    function h = plot2d(obj, standard_deviation, color_index)
      % quite raw function, needs work before proper visualization
      if obj.D == 2
        if ~exist('color_index', 'var'), color_index = 1; end
        npts = 50;
        tt=linspace(0,2*pi,npts)';
        x = cos(tt); y=sin(tt);
        ap = [x(:) y(:)]';
        [v,d]=eig(obj.sigma);
        d = standard_deviation * sqrt(d); % convert variance to sdwidth*sd
        bp = (v*d*ap) + repmat(obj.mu, 1, size(ap,2));
        ax = gca;
        ax.ColorOrderIndex = color_index;
        h = plot(bp(1,:), bp(2,:), '-');
        ax = gca;
        ax.ColorOrderIndex = color_index;
        h = plot(obj.mu(1), obj.mu(2), 'x');
      else
        warning('plot2d requires 2D MVN (D=2)')
      end
    end
    
    %% Evaluate gradient of MVN in a set of point point
    % from Huber08, Example 1
    function gradient_value = evaluateGradient(obj, X)
      N = size(X,2);
      gradient_value = zeros(obj.D,N);
      
      for i=1:N
        gradient_value(:,i) = - obj.lambda * (X(:,i) - obj.mu)...
          * obj.pdf(X(:,i));
      end
    end
  end
end

%% REFERENCES
%   Huber08 - Huber, Marco F., et al. "On entropy approximation for 
%    Gaussian mixture random vectors." Multisensor Fusion and Integration 
%    for Intelligent Systems, 2008. MFI 2008. IEEE International Conference
%    on. IEEE, 2008.