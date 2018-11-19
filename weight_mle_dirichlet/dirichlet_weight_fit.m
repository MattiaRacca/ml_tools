function [a,run] = dirichlet_weight_fit(data,w,a,bar_p)
% DIRICHLET_WEIGHT_FIT   Maximum-likelihood Dirichlet distribution.
%
% DIRICHLET_WEIGHT_FIT(data,w) returns the weighted MLE (a) for the matrix DATA.
% Each row of DATA is a probability vector.
% DIRICHLET_WEIGHT_FIT(data,w,a) provides an initial guess A to speed up the
%   search.
%
% The Dirichlet distribution is parameterized as
%   p(p) = (Gamma(sum_k a_k)/prod_k Gamma(a_k)) prod_k p_k^(a_k-1)
%
% The algorithm is an alternating optimization for m and for s, described in
% "Estimating a Dirichlet distribution" by T. Minka.

% Based on dirichlet_fit.m by Tom Minka
% https://github.com/tminka/fastfit

%[N,K] = size(data);
if nargin < 4
  bar_p = sum(log(data).*repmat(w,1,size(data, 2)),1)/sum(w);
  addflops(numel(data)*(flops_exp + 1));
end
K = length(bar_p);
if nargin < 3
  % for now I don't change this part...
  a = dirichlet_moment_match(data);
  %s = dirichlet_initial_s(a,bar_p);
  %a = s*a/sum(a);
end

s = sum(a);
if s <= 0
  % bad initial guess; fix it
  disp('fixing initial guess')
  if s == 0
    a = ones(size(a))/length(a);
  else
    a = a/s;
  end
  s = 1;
end
for iter = 1:100
  old_a = a;
  % time for fit_s is negligible compared to fit_m
  a = dirichlet_fit_s(data, a, bar_p);
  s = sum(a);
  a = dirichlet_fit_m(data, a, bar_p, 1);
  m = a/s;
  addflops(2*K-1);
  if nargout > 1
    run.e(iter) = dirichlet_logProb_fast(a, bar_p);
    run.flops(iter) = flops;
  end
  if abs(a - old_a) < 1e-4
    break
  end
end
%flops(flops + iter*(2*K-1));
