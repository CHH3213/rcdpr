
clc;
clear;
%%% 画无人机的轨迹
% load('./21/net_big/network_trajectry_02_0.mat')  % RCDPR-big
load('./21/net_small1/network_trajectry_02_0.mat')  % RCDPR-small
r=0.5; %圆的半径为1
c=[0.5 0 4]; %圆心的坐标
delta = 30;
delta_begin1 = 1;  % 都为delta_begin1
delta_begin2 = 0;  % 除了fixed_small为0，其他都尉delta_begin1

x1 = drone1_pos(delta_begin1:delta:length(drone1_pos)-delta_begin2,1);
y1 = drone1_pos(delta_begin1:delta:length(drone1_pos)-delta_begin2,2);
z1 = drone1_pos(delta_begin1:delta:length(drone1_pos)-delta_begin2,3);

x2 = drone2_pos(delta_begin1:delta:length(drone2_pos)-delta_begin2,1);
y2 = drone2_pos(delta_begin1:delta:length(drone2_pos)-delta_begin2,2);
z2 = drone2_pos(delta_begin1:delta:length(drone2_pos)-delta_begin2,3);

x3 = drone3_pos(delta_begin1:delta:length(drone3_pos)-delta_begin2,1);
y3 = drone3_pos(delta_begin1:delta:length(drone3_pos)-delta_begin2,2);
z3 = drone3_pos(delta_begin1:delta:length(drone3_pos)-delta_begin2,3);

x4 = drone4_pos(delta_begin1:delta:length(drone4_pos)-delta_begin2,1);
y4 = drone4_pos(delta_begin1:delta:length(drone4_pos)-delta_begin2,2);
z4 = drone4_pos(delta_begin1:delta:length(drone4_pos)-delta_begin2,3);

% draw
% figure();
% plot3(x1,y1,z1,'r-','LineWidth',2)
% hold on;
% plot3(x2,y2,z2,'r-','LineWidth',2)
% hold on;
% plot3(x3,y3,z3,'r-','LineWidth',2)
% hold on;
% plot3(x4,y4,z4,'r-','LineWidth',2)
% grid on;
% %%%%%%%%%%%%%%%%坐标轴相关设置
% set(gca,'GridLineStyle',':','GridColor','b','GridAlpha',1);%添加网格虚线
% set(gca,'FontName','Times New Roman','FontSize',25);
% set(gca,'FontSize',25);
% xlabel('\it\fontname{Times New Roman}x \rm(meter)','FontSize',25,'Rotation',30)
% ylabel('\it\fontname{Times New Roman}y \rm(meter)','FontSize',25,'Rotation',-40)
% zlabel('\it\fontname{Times New Roman}z \rm(meter)','FontSize',25)
% hl20 = legend('CDPR','Target','FontName','Times New Roman','FontSize',25);
% % hl20 = legend('RCDPR','Target','FontName','Times New Roman','FontSize',25);
% 
% set(hl20,'Box','on');
% axis equal;
% % axis([-0.5,1.5,-1.0,1.0,3.5,4.5]) %big
% set(gca,'XTickLabelRotation',-45)
% % title('(a)','fontsize',25,'position',[2,1.0])


figure();
plot(x1,y1,'r-','LineWidth',2)
hold on;
plot(x2,y2,'g-','LineWidth',2)
hold on;
plot(x3,y3,'b-','LineWidth',2)
hold on;
plot(x4,y4,'k-','LineWidth',2)
grid on;
%%%%%%%%%%%%%%%%坐标轴相关设置
set(gca,'GridLineStyle',':','GridColor','b','GridAlpha',1);%添加网格虚线
set(gca,'FontName','Times New Roman','FontSize',25);
set(gca,'FontSize',25);
xlabel('\it\fontname{Times New Roman}x \rm(meter)','FontSize',25,'Rotation',0)
ylabel('\it\fontname{Times New Roman}y \rm(meter)','FontSize',25,'Rotation',90)
hl20 = legend('drone1','drone2','drone3','drone4','FontName','Times New Roman','FontSize',25);
axis equal;
xticks([-5,0,5]);
yticks([-5,0,5]);
axis([-5,5,-5.0,5.0]) %big

