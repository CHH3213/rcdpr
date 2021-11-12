# -*-coding:utf-8-*-
'''
画图
'''
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self):
        pass
    def plot_info(self, title, legend):
        '''
        画图信息输入
        :param title:图名,字符串
        :param legend: 图力，字符串组成的列表
        '''
        plt.title(title)
        plt.legend(labels=legend, loc='best')
    def plot(self, args_data):
        plt.figure()
        for data in args_data:
            plt.plot(data)
    def plot_save(self, save_dir):
        '''
        :param save_dir: 保存路径  字符串
        '''
        plt.savefig(save_dir)


if __name__=='__main__':
    plot = Plot()
    list = np.array([np.array([1,2,3,4,5,6]), np.array([1,5,3,4,5,6])])
    plot.plot(list)
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axhline(y=2, color='r', linestyle='-')
    plot.plot_info('asd',['1','2'])
    plot.plot_save('figure/ll.png')

