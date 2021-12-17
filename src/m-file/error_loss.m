clc;
clear;
load('./21/loss.mat')
mean_error = mean(abs(relative_error))
for i=1:length(relative_error)
    if abs(relative_error(i))>0.8
        relative_error(i)=0;
    end
end
max_error = max(abs(relative_error))
rel_error = abs(relative_error);

