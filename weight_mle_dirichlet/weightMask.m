function weight = weightMask(value)
% WEIGHTMASK  Simple function for computing the weight of a point
% Just an example... it disincentivate values close to 0.8...

alpha = 10;
beta = 2;
mode = (alpha - 1)/(alpha + beta - 2);
weight = zeros(length(value),1);
for i=1:length(value)
  weight(i) = betapdf(value(i),alpha,beta)/...
    betapdf(mode, alpha, beta);
end
end