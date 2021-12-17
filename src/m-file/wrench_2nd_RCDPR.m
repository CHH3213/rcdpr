
clc;
clear;
% 大圆 RCDPR的r

load('./21/net_big/r.mat')
load('./21/net_big/network_trajectry_02_0.mat')



% define trajectory
x = [];
y = [];
z = [];
sz = [];
color = [];
delta = 100;
coef = 1.0;
delta_begin = 3000; 
for k=3000:delta:length(r_t)-delta_begin
      x=[x,mean(platform_pos(k:k+coef*delta-1,1))];
      y=[y,mean(platform_pos(k:k+coef*delta-1,2))];
      z=[z,mean(platform_pos(k:k+coef*delta-1,3))];
      sz=[sz,mean(-r_t(k:k+coef*delta-1))];%改变圆圈尺寸
      color=[color,mean(-r_t(k:k+coef*delta-1))];%改变圆圈颜色
end

% length(sz)

figure();
% ax1 = nexttile;
bubblechart3(x,y,z,sz,color,'MarkerFaceAlpha',0.6)
bubblesize([18 45])
% hl20 = legend('r^{AW}','FontName','Times New Roman','FontSize',25);
% set(hl20,'Box','on');
grid on;
set(gca,'GridLineStyle',':','GridColor','b','GridAlpha',1);%添加网格虚线
axis equal;
caxis([0 7])
h = colorbar('eastoutside');%右侧颜色栏
set(get(h,'label'),'string','\it\fontname{Times New Roman} \rm(N)','FontSize',40,'Rotation',0);%给右侧颜色栏命名
% set(h,'Position', [0.41 0.25 0.022 0.5]);  %设置colorbar位置
set(gca,'FontName','Times New Roman','FontSize',40);
set(gca,'FontSize',40);
xlabel('\it\fontname{Times New Roman}x \rm(meter)','FontSize',40,'Rotation',25)
ylabel('\it\fontname{Times New Roman}y \rm(meter)','FontSize',40,'Rotation',-35)
zlabel('\it\fontname{Times New Roman}z \rm(meter)','FontSize',40)

axis equal
yticks([-2,-1,0,1,2])
xticks([0,1,2,3])
axis([-0.5,3,-2,2,3.5,4.5]) %big
% title('(b) RCDPR','fontsize',25,'position',[2,1.0])
% xtickangle(45);
