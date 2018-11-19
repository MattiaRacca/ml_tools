function [beta] = Beta( alpha )
%BETA Beta function over alpha
%  uses logBeta to avoid numerical issues
  beta = exp(logBeta(alpha));

end

