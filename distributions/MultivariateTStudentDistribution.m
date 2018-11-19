classdef MultivariateTStudentDistribution < handle
  %MULTIVARIATETSTUDENTDISTRIBUTION
  %   Detailed explanation goes here
  
  properties
    %% Parameters
    D           % Number of Dimensions
    mu          % Mean vector (column vector)
    sigma       % Covariance matrix
    nu          % Degree of Freedom
  end
  properties (SetAccess  = private)
    sigma_det   % Determinant of Sigma
    lambda      % Precision matrix (inverse of sigma)
  end
  
  methods
    %% Constructor
    function obj = MultivariateTStudentDistribution(dimensions, mean,...
        covariance, dof)
      if nargin == 4
        %% No prior knowledge case
        obj.D = dimensions;
        obj.mu = mean;
        obj.sigma = covariance;
        obj.nu = dof;
        
        obj.sigma_det = det(obj.sigma);
        obj.lambda = inv(obj.sigma);
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
    function set.nu(obj, dof)
      if(dof > 0)
        obj.nu = dof;
      else
        error('The degree of freedom must be positive');
      end
    end
    
    %% PDF
    % from Murphy12, 2.5.3
    function prob = pdf(obj, data)
      % data is a DxN matrix, where D is dimensionality and N is
      % number of point to evaluate
      
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and MVST do not agree')
      else
        %% Naive implementation
        %  N = size(data, 2);
        %  prob = zeros(1,N);
        %  b = (obj.nu + obj.D)/2;
        %
        %  for i=1:N
        %    maha = (data(:,i) - obj.mu)'*obj.lambda*(data(:,i) - obj.mu);
        %    prob(i) = gamma(b)/gamma(obj.nu/2)*...
        %      (1+maha/obj.nu)^(-b)/...
        %      (sqrt(obj.sigma_det)*(obj.nu)^(obj.D/2)*(pi)^(obj.D/2));
        %  end
        %% Numerically stable implementation
        prob = exp(logpdf(obj,data));
      end
    end
    
    %% logPDF
    % Derived from Murphy12, 2.5.3
    function logprob = logpdf(obj,data)
      if (size(data,1) ~= obj.D)
        error('Dimensionality of data and MVST do not agree')
      else
        N = size(data, 2);
        logprob = zeros(1,N);
        
        lognu = log(obj.nu);
        logdet = log(obj.sigma_det);
        b = (obj.nu + obj.D)/2;
        
 
        for i=1:N
          mahalanobis = (data(:,i) - obj.mu)'*obj.lambda*(data(:,i) - obj.mu);
          z = log1p(mahalanobis/obj.nu);
          
          logprob(i) = gammaln(b) - gammaln(obj.nu/2) - 0.5*logdet...
            - (obj.D/2)*(lognu + log(pi)) - b * z;
        end
      end
    end
    
    %% Laplace Approximation (MVN)
    % Derived from Bishop06, 4.4
    function approximation = laplaceApproximation(obj)
      % off diagonal elements should be (a_{ij} + a_{ji}, but
      %  a_{ij} = a_{ji} because the precision matrix is symmetric
      %  so 2a_{ij} is also correct
      A = obj.lambda.*2*(obj.nu+1)/(2*obj.nu);
      approximation = MultivariateNormalDistribution(obj.D, obj.mu,...
        inv(A));
    end
  end
end

