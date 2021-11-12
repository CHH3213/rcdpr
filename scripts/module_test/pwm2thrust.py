# -*-coding:utf-8-*-
'''
PWM映射简单测试
'''
import matplotlib.pyplot as plt
import numpy as np



# 定义x、y散点坐标
y = [(i+18.88) for i in range(14)] # 每个pwm对应的推力
y = np.array(y)
data = [1562,1577,1592,1605,1618,1632,1643,1658,1670,1683,1698,1714,1720,1733]
x = np.array(data)
# x = x*4
# 用3次多项式拟合
f1 = np.polyfit(x, y, 1)
print('f1 is :\n', f1)

p1 = np.poly1d(f1)
print('p1 is :\n', p1)

# 也可使用yvals=np.polyval(f1, x)
yvals = p1(x)  # 拟合y值
print('yvals is :\n', yvals)
# 绘图
plot1 = plt.plot(x, y, 's', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)  # 指定legend的位置右下角
plt.title('polyfitting')
plt.show()
