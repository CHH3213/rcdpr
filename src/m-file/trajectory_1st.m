
clc;
clear;
%%% 画小圆的轨迹
% load('./21/fix_small/network_trajectry_02_0.mat')  % CDPR-small
load('./21/net_small1/network_trajectry_02_0.mat')  % RCDPR-small
r=0.5; %圆的半径为1
c=[0.5 0 4]; %圆心的坐标
delta = 30;
delta_begin1 = 3000;  % 都为3000
delta_begin2 = 3000;  % 除了fixed_small为0，其他都尉3000

x = platform_pos(3000:delta:length(platform_pos)-delta_begin2,1);
y = platform_pos(3000:delta:length(platform_pos)-delta_begin2,2);
z = platform_pos(3000:delta:length(platform_pos)-delta_begin2,3);

% draw
figure();
plot3(x,y,z,'r-','LineWidth',2)

% standard trajectory
n=[0 0 1]; %法向量n
theta=(0:2*pi/length(x):2*pi)'; %theta角从0到2*pi
a=cross(n,[1 0 0]); %n与i叉乘，求取a向量
if ~any(a) %如果a为零向量，将n与j叉乘
    a=cross(n,[0 1 0]);
end
b=cross(n,a); %求取b向量
a=a/norm(a); %单位化a向量
b=b/norm(b); %单位化b向量

c1=c(1)*ones(size(theta,1),1);
c2=c(2)*ones(size(theta,1),1);
c3=c(3)*ones(size(theta,1),1);

x_c=c1+r*a(1)*cos(theta)+r*b(1)*sin(theta);%圆上各点的x坐标
y_c=c2+r*a(2)*cos(theta)+r*b(2)*sin(theta);%圆上各点的y坐标
z_c=c3+r*a(3)*cos(theta)+r*b(3)*sin(theta);%圆上各点的z坐标
hold on;
plot3(x_c,y_c,z_c,'k--','LineWidth',2)
grid on;

%%%%%%%%%%%%%%%%坐标轴相关设置
set(gca,'GridLineStyle',':','GridColor','b','GridAlpha',1);%添加网格虚线
set(gca,'FontName','Times New Roman','FontSize',25);
set(gca,'FontSize',25);
xlabel('\it\fontname{Times New Roman}x \rm(meter)','FontSize',25,'Rotation',30)
ylabel('\it\fontname{Times New Roman}y \rm(meter)','FontSize',25,'Rotation',-40)
zlabel('\it\fontname{Times New Roman}z \rm(meter)','FontSize',25)
hl20 = legend('CDPR','Target','FontName','Times New Roman','FontSize',25);
% hl20 = legend('RCDPR','Target','FontName','Times New Roman','FontSize',25);

set(hl20,'Box','on');
axis equal;
axis([-0.5,1.5,-1.0,1.0,3.5,4.5]) %big
set(gca,'XTickLabelRotation',-45)
% title('(a)','fontsize',25,'position',[2,1.0])


