function result = logsum( values )
%LOGSUM logsum trick implementation
%   Input: values to be summed
%   from Murphy12 3.5.3, uses logsumexp function

% compute the exponent of the values
exponents = log(values);
% logsumexp trick
result = logsumexp(exponents);

end