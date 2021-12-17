clc;
clear;
%  load('./21/net_big/r.mat')
  load('./21/net_small1/r.mat')
%   load('./21/fix_small/r.mat')
  
mean_error = mean(abs(r_t(3000:length(r_t)-3000)))
% max_error = max(abs(r_t(3000:length(r_t)-3000)))
min_error = min(abs(r_t(3000:length(r_t)-3000)))
rel_error = abs(r_t(3000:length(r_t)-3000));

%  load('./21/fix_big/r.mat')
% mean_error = mean(abs(r_t(3000:length(r_t))))
% min_error = min(abs(r_t(3000:length(r_t))))
