
clc;
clear;
% load mat  net_big 大圆，不固定；net_small：小圆，不固定；fix_small:小圆，固定
load('./21/net_small1/network_trajectry_02_0.mat');
% load('./21/fix_small/network_trajectry_02_0.mat');
% load('./21/net_big/network_trajectry_02_0.mat');
r=0.5; %圆的半径为1
c=[r 0 4]; %圆心的坐标
delta = 10;
delta_begin2 = 3000;  % 除了13_b为30，其他都尉3000

x1 = platform_pos(3000:delta:length(platform_pos)-delta_begin2,1);
y1 = platform_pos(3000:delta:length(platform_pos)-delta_begin2,2);
z1 = platform_pos(3000:delta:length(platform_pos)-delta_begin2,3);

% length(x1)
% % 理想轨迹
theta=(0:2*pi/length(x1):2*pi)'; %theta角从0到2*pi
x_a = c(1)+r*cos(pi-theta);
y_a = c(2)+r*sin(pi-theta);
z_a= 4.0*ones(1,length(x_a));

x_error = abs(x1-x_a(1:300));
y_error = abs(y1-y_a(1:300));
z_error = abs(z1-z_a(1:300));
dist_error = ones(1,length(x_error));
for i = 1:length(x_error)
    dist_error(i) = norm([x_error(i),y_error(i),z_error(i)]);
end
dist_error;
error_mean = mean(dist_error)
error_max = max(dist_error)