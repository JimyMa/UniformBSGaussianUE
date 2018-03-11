# -*- coding:utf-8 -*-
import numpy as np
from service_region import ServiceRegion
import matplotlib.pyplot as plt
import scipy.integrate as sci
import seaborn as sns


class HotServiceRegion(ServiceRegion):
    """
    热点区域继承自服务区域， 用户的分布不再是随机分布，而是按聚集在热点附近
    """
    # 设置热点区域，用户围绕着热点区域进行分布
    # 热点X轴，列向量
    hot_point_X = np.array([], dtype=float)
    # 热点Y轴，列向量
    hot_point_Y = np.array([], dtype=float)
    # 热点坐标数组
    hot_point_position = np.array([], dtype=float)
    # 热点的个数
    hot_point_num = 0
    # 热点的密度（个数 = 区域面积 x 密度）
    hot_density = 0.0
    # 热点的分布，默认为均匀分布
    hot_point_distribution = 'Uniform'
    # 存放每个热点所包含的用户的一个字典
    # 用户X轴
    UE_X = {}
    # 用户Y轴
    UE_Y = {}
    # 用户的坐标字典
    UE_position = {}
    # 存放所有用户的数组
    # 所有用户X轴，行向量
    all_UE_X = np.array([], dtype=float)
    # 所有用户Y轴，行向量
    all_UE_Y = np.array([], dtype=float)
    # 所有用户的坐标
    all_UE_position = np.array([], dtype=float)
    # 用户的总数
    all_UE_num = 0
    # 每个热点的用户数，该参数决定了迭代的次数，也变向的决定了精度
    UE_num = np.array([], dtype=int)
    # 每个热点的密度，默认情况下
    # 每个热点的密度和用户数默认是相等的
    # 同时也留有为每个热点分别决定用户数和密度的接口
    UE_density = np.array([], dtype=float)
    # 用户围绕热点的分布
    UE_distribution = 'Gaussian'
    # 当用户围绕热点为高斯分布的时候的标准差，默认情况下为1.0
    UE_sigma = np.array([], dtype=float)

    def set_hot_density(self, hot_density):
        """
        用于设置热点的密度
        :param hot_density: 预设值
        :return: 改变类对象的对应值
        """
        self.hot_density = hot_density

    def set_hot_point_distribution(self, distribution):
        """
        用于预设分布参数
        :param distribution: 预设值
        :return: 改变类对象的对应值
        """
        self.hot_point_distribution = distribution

    def set_hot_point(self):
        """
        根据密度和分布给热点区域撒点
        :return: 改变对应的对象只self.hot_point_X，self.hot_point_Y，self.hot_point_position
        与基站有关的，与热点有关的向量都是列向量
        与用户有关的向量都是行向量
        """
        # 产生随机数的种子，不然每次撒点的结果是相同的
        np.random.seed(np.random.randint(0, np.iinfo(np.uint32).max))
        # 根据热点密度求热点的个数
        self.hot_point_num = int(self.hot_density *
                                 (self.x_right - self.x_left) *
                                 (self.y_right - self.y_left))
        # 如果热点以均匀分布分布
        if self.hot_point_distribution == 'Uniform':
            # 撒X坐标
            self.hot_point_X = np.random.uniform(self.x_left,
                                                 self.x_right,
                                                 (self.hot_point_num, 1))
            # 撒Y坐标
            self.hot_point_Y = np.random.uniform(self.y_left,
                                                 self.y_right,
                                                 (self.hot_point_num, 1))
            # 点的坐标值
            self.hot_point_position = np.concatenate([self.hot_point_X,
                                                      self.hot_point_Y],
                                                     axis=1)
            # 此处加断言，保证容器是一个列向量
            assert(np.size(self.hot_point_X, axis=0) == np.size(self.hot_point_Y, axis=0))
            assert(np.size(self.hot_point_position, axis=1) == 2)
        else:
            # TODO: 加入其他的分布
            pass

    def set_UE_distribution(self, distribution, UE_sigma, same=True):
        """
        用户的分布
        :param distribution: 分布预设值
        :param UE_sigma:  标准差预设值
        :return: 对应的对象值
        """
        self.UE_distribution = distribution
        if distribution == 'Gaussian':
            if same:
                self.UE_sigma = UE_sigma * np.ones((self.hot_point_num, 1))
            else:
                self.UE_sigma = UE_sigma

    def set_UE_sigma(self, sigma, same=True):
        """
        改变标准差
        :param sigma: 预设值
        :return: 类对象的对应值
        """
        if self.UE_distribution == 'Gaussian':
            if same is True:
                self.UE_sigma = sigma * np.ones(np.shape(self.hot_point_num, 1))
            else:
                self.UE_sigma = sigma
        else:
            raise Exception("UE distribution should be Gaussian")

    def set_UE_density(self, density, same=True):
        """
        设置用户的围绕热点的密度值
        :param density: 预设值
        :param same: 是否每个热点的用户的密度是相同的
        :return: 对应的对象值
        """
        if same is True:
            self.UE_density = np.ones((self.hot_point_num, 1)) * density

        elif same is False:
            # 断言保证了对应的热点均能得到密度参数，不多也不少
            assert(np.size(self.UE_density) ==
                   np.size(self.hot_point_X) ==
                   np.size(self.hot_point_Y))
            self.UE_density = density

    def set_UE(self):
        """
        根据热点坐标，用户密度，用户分布给用户撒点
        :return: 撒点装入容器中
        """
        if self.UE_distribution == 'Gaussian':

            self.UE_num = self.UE_density * (3.0 * self.UE_sigma) ** 2.0 * np.pi
            self.UE_num = np.array(self.UE_num, dtype=np.int32)
            for i in xrange(0, self.hot_point_num):

                self.UE_X[i] = (self.UE_sigma[i] * np.random.randn(1, self.UE_num[i, 0])
                                + self.hot_point_X[i])

                self.UE_Y[i] = (self.UE_sigma[i] * np.random.randn(1, self.UE_num[i, 0])
                                + self.hot_point_Y[i])

                self.UE_position[i] = np.concatenate([self.UE_X[i],
                                                      self.UE_Y[i]],
                                                     axis=0)
        else:
            # TODO加入其他的用户分布类型
            raise Exception("Unimplementable")

    def set_all_UE(self):
        """
        字典容器转为数组容器
        :return: 所有用户装到行向量容器中
        """
        for i in range(0, self.hot_point_num):
            if i == 0:
                self.all_UE_X = self.UE_X[i]
                self.all_UE_Y = self.UE_Y[i]
                self.all_UE_position = self.UE_position[i]
            else:
                self.all_UE_X = np.concatenate([self.all_UE_X,
                                               self.UE_X[i]],
                                               axis=1)
                self.all_UE_Y = np.concatenate([self.all_UE_Y,
                                               self.UE_Y[i]],
                                               axis=1)
                self.all_UE_position = np.concatenate([self.all_UE_position,
                                                      self.UE_position[i]],
                                                      axis=1)
        assert(np.size(self.all_UE_X) == np.size(self.all_UE_Y))
        assert(np.size(self.all_UE_X, axis=0) == np.size(self.all_UE_Y, axis=0) == 1)
        self.all_UE_num = np.size(self.all_UE_X, axis=1)

    def set_hot_point_as_BS(self):
        """
        假设刚好热点处放置了基站
        :return: 类对象热点的X轴，Y轴，坐标赋值给类对象基站的X轴，Y轴，坐标
        """
        self.BS_X = self.hot_point_X
        self.BS_Y = self.hot_point_Y
        self.BS_position = self.hot_point_position
        P = self.P_sum / self.hot_point_num
        self.BS_P = P * np.ones((1, self.hot_point_num))

    def plot_UE_scatter(self):
        """
        画出热点和热点区域的用户的散点图，不同组的用户用不同的颜色区分尽量全部区分
        TODO 让所有的不同的热点的用户全部不相邻（四色着图）
        :return: 热点用户散点图以及以热点为依据的voronoi图
        """
        color = ["c", "g", "m", "y"]
        for i in xrange(0, self.hot_point_num):
            plt.scatter(self.UE_X[i],
                        self.UE_Y[i],
                        c=color[i % len(color)],
                        s=1)
        '''
        plt.scatter(self.hot_point_X,
                    self.hot_point_Y,
                    marker="o",
                    c="k",
                    s=30,
                    alpha=0.3)
        plt.xlim(self.x_left, self.x_right)
        plt.ylim(self.y_left, self.y_right)
        '''

    def plot_hot_UE_SIR_ccdf(self):
        SIR_received = self.SIR_received(self.all_UE_X,
                                         self.all_UE_Y)
        hist, segment = np.histogram(10 * np.log10(SIR_received),
                                     bins=100,
                                     normed=True)
        dx = segment[1] - segment[0]
        cdf = np.cumsum(hist) * dx
        f = plt.scatter(segment[1:], 1 - cdf, marker='o', s=10)
        return f

    def plot_hot_UE_SIR_hist(self):
        """
        画出频率分布直方图，遍历图上的所有的点，每个点的SIR均进行记录
        如果有小尺度衰落，则每个位置点多次记录
        """
        SIR_received = self.SIR_received(self.all_UE_X,
                                         self.all_UE_Y)
        SIR_received_log = 10.0 * np.log10(SIR_received)

        SIR_received_log = SIR_received_log
        f = sns.kdeplot(SIR_received_log)
        return f

    def plot_hot_SIR_ccdf_theory(self, x_min_db, x_max_db):
        x_log = np.arange(x_min_db, x_max_db + 0.5, 0.5)
        x = np.power(10.0, x_log / 10.0)
        sigma = self.UE_sigma[0]
        d2s2 = self.hot_density * 2 * sigma * sigma
        if self.loss_factor == 4.0:
            x_sqrt = np.sqrt(x)
            pi = np.pi
            rou = x_sqrt * (pi / 2.0 - np.arctan(1.0 / x_sqrt))
            y = 1.0 / (1.0 + rou) + rou / (1 + rou) / ((1.0 + rou) * d2s2 * pi + 1)
            # f = plt.scatter(x_log, y, s=25, marker='o')
            f = plt.plot(x_log, y)
        if self.loss_factor == 2.0:
            def f(u):
                return 1 / (u + 1)

            rou = np.empty(x.shape, dtype=np.float64)
            y = np.empty(x.shape, dtype=np.float64)
            pi = np.pi
            for index, value in enumerate(x):
                rou[index] = value * sci.quad(f, 1 / value, 400)[0]
                y[index] = rou[index]/(1 + rou[index])/((1.0 + rou[index]) * d2s2 * pi + 1)+1.0 / (1.0 + rou[index])
            # f = plt.scatter(x_log, y, s=25, marker='o')
            f = plt.plot(x_log, y)
        return f


if __name__ == '__main__':
    hot_region = HotServiceRegion()
    hot_region.set_region(0.0, 200.0, 0.0, 200.0)
    hot_region.set_hot_density(1.0 / 400)
    hot_region.set_hot_point_distribution('Uniform')
    hot_region.set_small_fade('Rayleigh')
    hot_region.set_hot_point()
    hot_region.set_UE_density(0.1)
    hot_region.set_UE_distribution('Gaussian', 10.0)
    hot_region.set_UE()
    hot_region.set_hot_point_as_BS()
    hot_region.set_all_UE()
    hot_region.plot_BS_voronoi()
    hot_region.plot_UE_scatter()
    plt.show()
    hot_region.set_loss_factor(4.0)
    hot_region.set_P_sum(5.0)
    hot_region.plot_hot_UE_SIR_ccdf()
    hot_region.plot_hot_SIR_ccdf_theory(-10, 20)
    hot_region.set_loss_factor(2.0)
    hot_region.plot_hot_UE_SIR_ccdf()
    hot_region.plot_hot_SIR_ccdf_theory(-10, 20)
    plt.xlim(-10, 20)
    plt.show()
    hot_region.set_loss_factor(4.0)
    hot_region.plot_hot_UE_SIR_hist()
    hot_region.set_loss_factor(2.0)
    hot_region.plot_hot_UE_SIR_hist()
    plt.xlim(-10, 20)
    plt.show()
