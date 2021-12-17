clc;
clear;
%%%%%%%%%%%%% 画出优化和神经网络时间对比图
load('./21/net_small1/network_trajectry_02_0.mat')
load('./21/net_small1/opt_net_time_next.mat')

% for i=1:length(net_time)
%     if net_time(i)>0.04
%         net_time(i)=0;
%     end
% end
% save('network_trajectry_02_0.mat',"net_time","platform_quat","steps","platform_pos","pid_times","drone4_pos","drone3_pos","drone2_pos","drone1_pos")

mean_net_time = mean(net_time(1:10:length(net_time)))
max_net_time = max(net_time(1:10:length(net_time)))
mean_opot_time = mean(opt_times(1:10:length(net_time)))
max_opt_time = max(opt_times(1:10:length(net_time)))

% y_net = [];
% y_opt = []
% delta = 10
% for k=1:delta:length(net_time)-10
%       y_net=[y_net,mean(net_time(k:k+delta-1))];
%       y_opt=[y_opt,mean(opt_times(k:k+delta-1))];
% end
% length(y_opt)
% length(y_net)
% x =1:length(y_net);
% x = x/10;
figure();
% 每一个step的实际值,每10步取均值
% semilogy(x,y_net,color='r','LineWidth',3);
% hold on;
% semilogy(x,y_opt,color='b','LineWidth',3);
% 
% % 每10步取一次
x =1:length(net_time(1:10:length(net_time)));
x = x/10;
semilogy(x,net_time(1:10:length(net_time)),'color','r','LineWidth',3);
hold on;
semilogy(x,opt_times(1:10:length(net_time)),'color','b','LineWidth',3);

% 原来
% x =1:length(net_time);
% x = x/100;
% semilogy(x,net_time,color='r','LineWidth',3);
% hold on;
% semilogy(x,opt_times,color='b','LineWidth',3);

hold on;
% % 均值
% plot([1,length(net_time)],[mean_net_time,mean_net_time],color='#FF4500','LineWidth',3)
% hold on;
% plot([1,length(opt_times)],[mean_opot_time,mean_opot_time],color='#1E90FF','LineWidth',3)
% hold on;
% % max
% plot([1,length(net_time)],[max_net_time,max_net_time],color='#32CD32','LineWidth',3)
% hold on;
% plot([1,length(opt_times)],[max_opt_time,max_opt_time],color='#DAA520','LineWidth',3)



grid on;
% axis equal;

set(gca,'GridLineStyle',':','GridColor','b','GridAlpha',1);%添加网格虚线

set(gca,'FontName','Times New Roman','FontSize',40);
set(gca,'FontSize',40);
% xlabel('time','FontName','Times New Roman','FontSize',40);
% ylabel('time','FontName','Times New Roman','FontSize',40,'Rotation',90);
xlabel('\fontname{Times New Roman}time \rm(second)','FontSize',40)
ylabel('\fontname{Times New Roman}computation time \rm(second)','FontSize',40)
hl20 = legend('Proposed reconfiguration planning approach','SLSQP','Mean-net-time','Mean-opt-time','Max-net-time','Max-opt-time','FontName','Times New Roman','FontSize',40);
set(hl20,'Box','on');
axis([0 90 0.0001 1.0])


% load('17\fix_big1\network_trajectry_02_0.mat')
% max_pid = max(pid_times)
% mean_pid = mean(pid_times)


