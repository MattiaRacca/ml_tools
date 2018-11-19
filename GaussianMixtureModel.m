classdef GaussianMixtureModel < handle
  %GAUSSIAN MIXTURE MODEL Class implementation of a GMM
  %   for reference Murphy12, Chapter 11
  
  properties
    %% Parameters
    K           % Number of components
    w           % Mixing weight vector
    components  % Set of base distribution (components of the mixture)
                % (vector of MultivariateNormalDistribution objects)
  end
  properties (SetAccess = private)
    D           % Dimensionality of data
  end
  
  methods
    %% Constructor
    function obj = GaussianMixtureModel(number_components, mixing_weight...
        , gaussian_components)
      if nargin > 0
        obj.K = number_components;
        if nargin == 3
          obj.w = mixing_weight;
          obj.components = gaussian_components;
        end
      end
    end
    
    %% Setters
    function set.K(obj, number_components)
      if number_components > 0
        obj.K = number_components;
      else
        error('Number of Components must be positive')
      end
    end
    function set.w(obj, mixing_weight)
      if length(mixing_weight) ~= obj.K
        error('The number of weight must match the number of components')
      else
        if sum(mixing_weight) ~= 1
          error('Mixing weight must sum to 1')
        else
          obj.w = mixing_weight;
        end
      end
    end
    function set.components(obj, gaussians)
      if length(gaussians) ~= obj.K
        error('The number of MVN must match the number of components')
      else
        if isa(gaussians, 'MultivariateNormalDistribution')
          obj.components = gaussians;
          obj.D = obj.components(1).D;
        else
          error('GMM must contain only MVNs')
        end
      end
    end
    
    %% PDF
    function prob = pdf(obj, data)
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and GMM do not agree')
      else
        %% Naive implementation
%         N = size(data, 2);
%         prob = zeros(1,N);
%         partial_prob = zeros(1,obj.K);
%         
%         % for each data point
%         for i=1:N
%           % for each component
%           for k=1:obj.K
%             partial_prob(k) = obj.w(k) * obj.components(k).pdf(data(:,i));
%           end
%           prob(i) = sum(partial_prob);
%         end
        %% Numerically stable implementation
        prob = exp(logpdf(obj,data));
      end
    end
    
    %% logPDF
    function logprob = logpdf(obj,data)
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and GMM do not agree')
      else
        N = size(data, 2);
        logprob = zeros(1,N);
        exponents = zeros(1,obj.K);
        
        % for each data point
        for i=1:N
          % for each component
          for k=1:obj.K
            exponents(k) = log(obj.w(k)) +...
              obj.components(k).logpdf(data(:,i));
          end
          logprob(i) = logsumexp(exponents);
        end
      end
    end
    
    %% likelihood
    function lik = likelihood(obj, data)
      lik = exp(loglikelihood(obj,data));
    end
    
    %% logLikelihood
    function loglik = loglikelihood(obj,data)
      loglik = sum(obj.logpdf(data));
    end
    
    %% Sampling
    function samples = sampleMixture(obj, N)
      if N > 0
        % sample over the mixture weight
        component_effort = mnrnd(N, obj.w);
        samples = [];
        
        % sample the components
        for k=1:obj.K
          samples = [samples,...
            obj.components(k).sampleDistribution(component_effort(k))];
        end
        %         samples = samples(:,randperm(N));
      end
    end
    
    %% Evaluate gradient of GMM in a point
    % from Huber08, Example 1 (partially)
    function gradient_value = evaluateGradient(obj, X)
      gradient_value = 0;
      for k=1:obj.K
        gradient_value = gradient_value +...
          obj.w(k)*obj.components(k).evaluateGradient(X);
      end
    end
    
    %% Compute Entropy with Huber08 approximation
    % from Huber08, Appendix B
    function entropy = computeEntropyApproximation(obj)
      entropy = obj.computeH0() + obj.computeH2();
    end
    
    %% Compute Entropy with MC sampling
    % from Murphy12, 2.7 and 2.8.1
    function entropy = computeEntropyMC(obj, N)
      % sample mixture
      samples = obj.sampleMixture(N);
      entropy = - obj.loglikelihood(samples)/N;     
    end
    
    %% Compute Entropy LB
    % from Huber08, 5A
    function entropy_lb = computeEntropyLB(obj)
      entropy_lb = 0;
      
      for i=1:obj.K
        s = 0;
        for j=1:obj.K
          mvn_temp = MultivariateNormalDistribution(obj.D,...
            obj.components(j).mu, obj.components(j).sigma + ...
            obj.components(i).sigma);
          s = s + obj.w(j)*mvn_temp.pdf(obj.components(i).mu);
        end
        entropy_lb = entropy_lb - obj.w(i)*log(s);
      end
    end
    
    %% Compute Entropy Loose UB
    % from Huber08, 5B
    function entropy_ub = computeEntropyLooseUB(obj)
      entropy_ub = 0;
      for i=1:obj.K
        tmp = obj.w(i)*(-log(obj.w(i)) + .5*log((2*pi*exp(1))^obj.D...
          * det(obj.components(i).sigma)));
        entropy_ub = entropy_ub + tmp;
      end
    end
  end
  methods (Access = private)
    %% Zeroth-order Taylor-series Expansion
    % from Huber08, Appendix A
    function H0 = computeH0(obj)
      H0 = 0;
      for k=1:obj.K
        H0 = H0 - obj.w(k)*obj.logpdf(obj.components(k).mu);
      end
    end
    
    %% Second-order Taylor-series Expansion
    % from Huber08, Appendix B
    function H2 = computeH2(obj)
      H2 = 0;
      for k=1:obj.K
        s = sum(sum(obj.computeF(obj.components(k).mu)...
          .*obj.components(k).sigma));
        H2 = H2 - obj.w(k)*0.5*s; 
      end
    end
    
    %% Compute F for second-order taylor-series expansion
    function F = computeF(obj, X)
      F = 0;
      inverse_of_pdf = 1/obj.pdf(X);
      gradient_in_X  = obj.evaluateGradient(X);
      
      for j=1:obj.K
        lambda = obj.components(j).lambda;
        mean_dif = (X-obj.components(j).mu);
        F = F + obj.w(j)*lambda*(inverse_of_pdf*((mean_dif)*...
          gradient_in_X') + (mean_dif)*(lambda*(mean_dif))' - eye(obj.D))...
          *obj.components(j).pdf(X);
      end
      F = F*inverse_of_pdf;
    end
  end
end

%% REFERENCES
%   Huber08 - Huber, Marco F., et al. "On entropy approximation for 
%    Gaussian mixture random vectors." Multisensor Fusion and Integration 
%    for Intelligent Systems, 2008. MFI 2008. IEEE International Conference
%    on. IEEE, 2008.