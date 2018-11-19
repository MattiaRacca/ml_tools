function result = logsumexp( exponents )
%LOGSUMEXP logsumexp trick implementation
%   Input: the exponents of the e base inside the sum
%   for reference Murphy12, 3.5.3

B = max(exponents);
% Eq. 3.74
result = log(sum(exp(exponents - B))) + B;

end