function [ logbeta ] = logBeta( alpha )
%LOGBETA logarithm of Beta function over alpha

logbeta = sum(gammaln(alpha)) - gammaln(sum(alpha));

end

