
# -*- coding:utf-8 -*-

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.integrate as sci
import numpy
import seaborn as sns


class ServiceRegion(object):
    """
    定义基站服务区域并确定基站的具体位置
    """
    BS_X = np.array([], dtype=float)
    BS_Y = np.array([], dtype=float)
    BS_P = np.array([], dtype=float)
    BS_position = np.array([], dtype=float)
    num_BS = 0

    def __init__(self,
                 x_left=0.0,
                 x_right=100.0,
                 y_left=0.0,
                 y_right=100.0,
                 P_sum=1.0,
                 density=0.01,
                 loss_factor=4.0,
                 small_fade='no_small_fade'):
        """
        定义基站的服务区域
        定义总功率
        定义密度参数
        定义损耗因子
        定义是否有小尺度衰落
        """
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.P_sum = P_sum
        self.density = density
        self.loss_factor = loss_factor
        self.small_fade = small_fade

    def set_region(self, x_left, x_right, y_left, y_right):
        """
        定义基站的服务区域
        """
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right

    def set_P_sum(self, P_sum):
        """
        重置总功率
        """
        self.P_sum = P_sum

    def set_small_fade(self, small_fade):
        """
        重置小尺度衰落的形式
        """
        self.small_fade = small_fade

    def set_density(self, density):
        """
        重置基站的密度
        """
        self.density = density

    def set_loss_factor(self, loss_factor):
        """
        重置损耗因子
        """
        self.loss_factor = loss_factor

    def set_BS(self, distribution='Poisson'):
        """
        构造具体的基站位置，基站服从泊松点过程， 用区域面积 * 密度这么多个随机分布的基站去模拟是合理的
        """
        # 基站数量为密度乘以面积
        self.num_BS =int(self.density *
                         (self.x_right - self.x_left) *
                         (self.y_right - self.y_left))
        # 只考虑基站服从的是泊松分布
        if distribution == 'Poisson':
            # 基站的横纵轴坐标是在区域内服从随机分布
            self.BS_X = np.random.uniform(self.x_left, self.x_right, (self.num_BS, 1))
            self.BS_Y = np.random.uniform(self.y_left, self.y_right, (self.num_BS, 1))
            self.BS_position = np.concatenate([self.BS_X, self.BS_Y], axis=1)
            # 暂时不考虑功率分配的情况，如需功率分配，可以继承该类，增加功率分配方法
            P = self.P_sum / self.num_BS
            self.BS_P = P * np.ones((1, self.num_BS))

    def SIR_received(self, X, Y):
        """
        批量求SIR
        :param X: UE的X坐标，数据类型为np.array
        :param Y: UE的Y坐标，数据类型为np.array
        :return: 给定存储在np.array中的坐标下UE的接收SIR
        """
        # 首先求出所有用户对所有基站的距离，这里用到了python广播，需要仔细看才能看懂
        # 着重考虑为什么要用np.reshape
        distance = np.sqrt(np.square(X - np.reshape(self.BS_X, (-1, 1))) +
                           np.square(Y - np.reshape(self.BS_Y, (-1, 1))))
        # 断言为什么要这样下？ 保证广播的正确进行
        assert (np.size(X, axis=0) == 1)
        assert (np.size(X, axis=0) == np.size(Y, axis=0))
        assert (np.size(X, axis=1) == np.size(Y, axis=1))

        # 根据离用户最近原则进行基站选择
        selected_BS = np.argmin(distance, axis=0)

        # 只考虑大尺度衰落，即距离对用户的接收信噪比的影响
        P_received_no_small = np.multiply(np.reshape(self.BS_P, (-1, 1)),
                                          np.power(distance, -self.loss_factor))

        # 这里面分有小尺度衰落和没有小尺度衰落进行划分
        if self.small_fade == 'no_small_fade':
            P_received = P_received_no_small
        elif self.small_fade == 'Rayleigh':
            # 如果有小尺度衰落，如果为瑞利衰落，功率前面要乘一个指数分布的随机变量
            P_received = (np.random.exponential(1.0, np.shape(P_received_no_small))
                          * P_received_no_small)
        # 根据索引找到接收的有用功率的大小
        S_received = P_received[selected_BS,
                                np.arange(0, np.size(X, axis=1))]
        # 剩下的都是干扰功率，将他们加起来作为干扰
        I_received = np.sum(P_received, axis=0) - S_received
        # SIR = S / R
        received_SIR = np.divide(S_received, I_received)
        index_inf = np.argwhere(np.isinf(received_SIR))
        for i in index_inf:
            received_SIR[i] = np.finfo(np.float32).max
        index_nan = np.argwhere(np.isnan(received_SIR))
        for i in index_nan:
            received_SIR[i] = np.finfo(np.float32).resolution
        return received_SIR

    def UE_ergodic_capacity(self, X, Y, iter_num=1000):
        """
        用于求取遍历容量
        :param X: 坐标
        :param Y: 坐标
        :param iter_num: 小尺度下的每个位置的仿真的点数
        :return: 每个位置的遍历容量
        """
        # 断言保证形状一致
        assert(np.size(X, axis=0) == 1)
        assert(np.size(X, axis=0) == np.size(Y, axis=0))
        assert (np.size(X, axis=1) == np.size(Y, axis=1))
        capacity = 0
        # 没有小尺度下不用迭代每个位置即为遍历容量
        if self.small_fade == 'no_small_fade':
            SIR_received = self.SIR_received(X, Y)
            capacity = 1.0 / 2.0 * np.log2(1.0 + SIR_received)
        # 有小尺度每个位置多次求取得到平均值
        elif self.small_fade == 'Rayleigh':
            for i in range(0, iter_num):
                SIR_received = self.SIR_received(X, Y)
                capacity += 1.0 / 2.0 * np.log2(1.0 + SIR_received)
            capacity /= iter_num
        return capacity

    def set_X_Y(self, quan=0.5):
        """
        以固定的粒度将区域离散化为二维的点阵
        :param quan: 粒度
        :return: XY的点阵坐标
        """
        x = np.arange(self.x_left, self.x_right + quan, quan)
        y = np.arange(self.y_left, self.y_right + quan, quan)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def plot_BS_scatter(self, figure=None):
        """
        画出基站的散点图，在该类中未被使用
        """
        figure.scatter(self.BS_X, self.BS_Y)
        plt.xlim(self.x_left, self.x_right)
        plt.ylim(self.y_left, self.y_right)

    def plot_BS_voronoi(self, figure=None, show_points_or_not=True):
        """
        画出关基站分布位置的Voronoi图
        """
        vor = Voronoi(self.BS_position)
        voronoi_plot_2d(vor, show_vertices=False, ax=figure, show_points=show_points_or_not)
        plt.xlim(self.x_left, self.x_right)
        plt.ylim(self.y_left, self.y_right)

    def plot_ergodic_capacity_contour(self, figtype='contour'):
        """
        画出容量的等高线图
        """
        x = np.arange(self.x_left, self.x_right + 0.5, 0.5)
        y = np.arange(self.y_left, self.y_right + 0.5, 0.5)
        X, Y = np.meshgrid(x, y)

        capacity = np.reshape(self.UE_ergodic_capacity(
                                    np.reshape(X, (1, -1)),
                                    np.reshape(Y, (1, -1))), np.shape(X))
        if figtype == 'contour':
            img = plt.contourf(X,
                               Y,
                               capacity,
                               100,
                               alpha=0.5,
                               cmap='jet',
                               vmin=0,
                               vmax=10)
            # v = np.linspace(-.1, 2.0, 15, endpoint=True)
            plt.colorbar(img)
            # 绘制等高线
            # fig = plt.contour(X, Y, capacity, 10, colors='black', linewidth=0.5)
            # 显示各等高线的数据标签
            # plt.clabel(fig, inline=True, fontsize=10)
        elif figtype == 'heatmap':
            plt.matshow(capacity)
        # plt.xlim(self.x_left, self.x_right)
        # plt.ylim(self.y_left, self.y_right)
        # plt.colorbar()

    def plot_hist(self):
        """
        画出频率分布直方图，遍历图上的所有的点，每个点的SIR均进行记录
        如果有小尺度衰落，则每个位置点多次记录
        """
        x = np.arange(self.x_left, self.x_right + 0.5, 0.5)
        y = np.arange(self.y_left, self.y_right + 0.5, 0.5)
        X, Y = np.meshgrid(x, y)
        SIR_received = self.SIR_received(np.reshape(X, (1, -1)),
                                         np.reshape(Y, (1, -1)))
        SIR_received_log = 10.0 * np.log10(SIR_received)

        SIR_received_log = SIR_received_log
        f = sns.kdeplot(SIR_received_log)
        return f

    def plot_cdf(self):
        """
        画出累积分布图，遍历图上的所有的点，每个点的SIR均进行记录
        如果有小尺度衰落，则么个位置点多次记录
        """
        x = np.arange(self.x_left, self.x_right + 0.5, 0.5)
        y = np.arange(self.y_left, self.y_right + 0.5, 0.5)
        X, Y = np.meshgrid(x, y)
        SIR_received = self.SIR_received(np.reshape(X, (1, -1)),
                                         np.reshape(Y, (1, -1)))
        hist, segment = np.histogram(10 * np.log10(SIR_received),
                                     bins=100,
                                     normed=True)
        dx = segment[1] - segment[0]
        cdf = np.cumsum(hist) * dx
        f = plt.scatter(segment[1:], 1 - cdf, marker='o', s=10)
        return f

    def plot_SIR_ccdf_theory(self, x_min_db, x_max_db):

        x_log = np.arange(x_min_db, x_max_db + 0.5, 0.5)
        x = np.power(10.0, x_log / 10.0)
        if self.loss_factor == 4.0:
            x_sqrt = np.sqrt(x)
            pi = np.pi
            y = 1.0 / (1.0 + x_sqrt * (pi / 2.0 - np.arctan(1.0 / x_sqrt)))
            # f = plt.scatter(x_log, y, s=25, marker='o')
            f = plt.plot(x_log, y)
        if self.loss_factor == 2.0:
            def f(u):
                return 1 / (u + 1)
            rou = np.empty(x.shape, dtype=np.float64)
            y = np.empty(x.shape, dtype=np.float64)
            for index, value in enumerate(x):
                rou[index] = value * sci.quad(f, 1 / value, 260)[0]
                y[index] = 1 / (1 + rou[index])
            # f = plt.scatter(x_log, y, s=25, marker='o')
            f = plt.plot(x_log, y)
        return f


if __name__ == '__main__':

    C = ServiceRegion(0.0, 200.0, 0.0, 200.0, 10.0, 1.0 / 400, 4.0)

    C.set_region(0.0, 200.0, 0.0, 200.0)
    C.set_BS()
    C.set_P_sum(10.0)
    C.set_density(1.0/400)
    C.set_loss_factor(4.0)
    '''
    C.set_small_fade(small_fade='no_small_fade')
    C.plot_hist()
    C.set_small_fade('Rayleigh')
    C.plot_hist()
    plt.show()
    '''
    '''
    C.set_loss_factor(4.0)
    fig = plt.figure()
    C.set_small_fade('Rayleigh')
    C.plot_cdf()
    C.plot_SIR_ccdf_theory(-10, 20)
    plt.xlim(-10, 20)
    C.set_loss_factor(2.0)
    C.set_small_fade('Rayleigh')
    C.plot_cdf()
    C.plot_SIR_ccdf_theory(-10, 20)
    plt.xlim(-10, 20)
    plt.show()
    '''

    C.set_small_fade(small_fade='no_small_fade')
    C.set_loss_factor(4.0)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    C.plot_ergodic_capacity_contour()
    C.plot_BS_voronoi(ax, show_points_or_not=False)
    ax = fig.add_subplot(1, 2, 2)
    C.set_loss_factor(2.0)
    C.plot_ergodic_capacity_contour()
    C.plot_BS_voronoi(ax, show_points_or_not=False)
    # fig.savefig('small_ergodic_capacity.tiff', bbox_inches='tight')
    plt.show()
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    C.plot_BS_voronoi(ax)
    fig.savefig('Voronoi.png', bbox_inches='tight')
    plt.show()

    '''
    C.set_loss_factor(4.0)
    C.plot_hist()
    C.set_loss_factor(2.0)
    C.plot_hist()
    plt.show()
    '''
