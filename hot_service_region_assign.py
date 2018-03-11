# -*- coding:utf-8 -*-

import minpy.numpy as np
from hot_service_region import HotServiceRegion
import matplotlib.pyplot as plt
from soft_k_means import SoftKMeans2d


class HotServiceRegionAssign(HotServiceRegion):
    assigned_BS_X = np.array([], dtype=float)
    assigned_BS_Y = np.array([], dtype=float)
    assigned_BS_position = np.array([], dtype=float)
    assigned_BS_num = 0
    assigned_BS_density = 0.0

    def set_assigned_BS_num(self, number, equal_hot_point_num=False):
        """
        给定基站数目，默认等于热点数目
        :param number: 预设值
        :param equal_hot_point_num: 是否默认等于热点的个数
        :return: 传递给类对象相应的值
        """
        if equal_hot_point_num is False:
            self.assigned_BS_num = number
        else:
            self.assigned_BS_num = self.hot_point_num

    def set_assigned_BS_given_all_UE(self, equal_hot_point_num=True):
        """
        目前的实现方法是：给定基站的数目等于热点的数目
        根据用户围绕着热点进行高斯分布进行基站的部署
        算法参考自:
        Information Theory, Inference, and Learning Algorithm
            - David Mackay
        :param equal_hot_point_num:是否保证热点数和部署的基站数相等
        :return:将聚类得到的基站向量存入容器中
                self.assigned_BS_X, self.assigned_BS_Y..
        """
        if equal_hot_point_num is True:
            self.assigned_BS_num = self.hot_point_num
            self.assigned_BS_density = (float(self.hot_point_num)
                                        / (self.x_right - self.x_left)
                                        / (self.y_right - self.y_left))
        self.soft_k_means_all_UE()

    def soft_k_means_all_UE(self, distribution='Gaussian', iter_num=100):
        assert(np.size(self.all_UE_X) != 0)
        assert(np.size(self.all_UE_Y) != 0)
        assert(np.size(self.all_UE_position) != 0)
        # TODO 默认用户依附热点服从高斯分布
        if distribution == 'Gaussian':
            # 随机生成self.assigned_BS_num个点为聚类做准备
            self.assigned_BS_X = np.random.uniform(self.x_left, self.x_right,
                                                   (self.assigned_BS_num, 1))
            self.assigned_BS_Y = np.random.uniform(self.y_left, self.y_right,
                                                   (self.assigned_BS_num, 1))
            self.assigned_BS_position = np.concatenate([self.assigned_BS_X,
                                                       self.assigned_BS_Y],
                                                       axis=1)
            gmm = SoftKMeans2d(self.assigned_BS_num,
                               self.UE_sigma,
                               iter_num,
                               self.x_left,
                               self.x_right,
                               self.y_left,
                               self.y_right)
            gmm.fit(self.all_UE_position.T,
                    self.x_left,
                    self.x_right,
                    self.y_left,
                    self.y_right)
            self.assigned_BS_position = gmm.means_
            self.assigned_BS_X = self.assigned_BS_position[:, 0]
            self.assigned_BS_X = np.stack(self.assigned_BS_X, axis=1)
            self.assigned_BS_Y = self.assigned_BS_position[:, 1]
            self.assigned_BS_Y = np.stack(self.assigned_BS_Y, axis=1)
            '''
            # TODO: Soft-k-means(hard subscribe soft by now)
            # 解决办法1:用sklearn库去实现EM algorithm
            gmm = BayesianGaussianMixture(n_components=self.assigned_BS_num,
                                          covariance_type='spherical',
                                          init_params='kmeans',
                                          warm_start=True)
            gmm.fit(np.transpose(self.all_UE_position).asnumpy())
            self.assigned_BS_position = np.array(gmm.means_)
            self.assigned_BS_X = np.reshape(self.assigned_BS_position[:, 0], (-1, 1))
            self.assigned_BS_Y = np.reshape(self.assigned_BS_position[:, 1], (-1, 1))
            '''
            '''
            # 首先hard-k-means迭代35次
            for i in xrange(0, 35):
                distance_X_hard = self.assigned_BS_X - self.all_UE_X
                distance_Y_hard = self.assigned_BS_Y - self.all_UE_Y
                distance_hard = np.sqrt(np.square(distance_X_hard) + np.square(distance_Y_hard))
                k_means_m_index = np.argmin(distance_hard, axis=0)
                k_means_delta = np.zeros((self.assigned_BS_num, self.all_UE_num))
                k_means_delta[k_means_m_index, np.arange(0, self.all_UE_num)] = 1
                self.assigned_BS_X = (np.sum(k_means_delta * self.all_UE_X, axis=1)
                                      / np.sum(k_means_delta, axis=1))
                self.assigned_BS_X = np.reshape(self.assigned_BS_X, (-1, 1))
                self.assigned_BS_Y = (np.sum(k_means_delta * self.all_UE_Y, axis=1)
                                      / np.sum(k_means_delta, axis=1))
                self.assigned_BS_Y = np.reshape(self.assigned_BS_Y, (-1, 1))
                self.assigned_BS_position = np.concatenate([self.assigned_BS_X,
                                                           self.assigned_BS_Y],
                                                           axis=1)
            # TODO soft-k-means: How to solve the nan data

            # Step 1：每个点n在聚类k中的影响力
            assert(np.size(self.assigned_BS_X, axis=1)
                   == np.size(self.assigned_BS_Y, axis=1)
                   == 1)

            assert(np.size(self.all_UE_X, axis=0)
                   == np.size(self.all_UE_Y, axis=0)
                   == 1)
            k_means_pi = np.ones(np.shape(self.assigned_BS_X)) / 100.0
            k_means_sigma = self.UE_sigma
            for i in xrange(0, iter_num):
                distance_X = self.assigned_BS_X - self.all_UE_X
                distance_Y = self.assigned_BS_Y - self.all_UE_Y
                distance = 0.5 * (np.square(distance_X) + np.square(distance_Y))
                # 此处的代码仿照sklearn
                log_responsibility = (- np.log(2 * np.pi * np.square(k_means_sigma))
                                      - distance / 2 * np.square(k_means_sigma))
                log_weights = logsumexp(log_responsibility.asnumpy(), axis=1)
                log_weights = numpy.reshape(log_weights, (-1, 1))
                log_weights = log_weights - logsumexp(log_weights)
                log_responsibility = log_responsibility.asnumpy() + log_weights
                log_prob_norm = logsumexp(log_responsibility, axis=0)
                log_prob_norm = numpy.reshape(log_prob_norm, (1, -1))
                log_resp = log_responsibility - log_prob_norm
                log_resp = np.array(log_resp)
                resp = np.exp(log_resp)
                self.assigned_BS_position = np.dot(resp, self.assigned_BS_position.T)
                self.assigned_BS_X = np.reshape(self.assigned_BS_position[:, 0], (-1, 1))
                self.assigned_BS_Y = np.reshape(self.assigned_BS_position[:, 1], (-1, 1))
                k_means_r = (k_means_pi
                             / np.square(k_means_sigma) / 2 / np.pi
                             * np.exp(-1 / np.square(k_means_sigma) * distance))
                k_means_r_sum_k = np.sum(k_means_r, axis=0)
                k_means_r_sum_k = np.reshape(k_means_r_sum_k, (1, -1))
                k_means_r_sum_n = np.sum(k_means_r, axis=1)
                k_means_r_sum_n = np.reshape(k_means_r_sum_n, (-1, 1))
                assert(np.size(k_means_r_sum_k, axis=0) == 1)
                assert(np.size(k_means_r_sum_k, axis=1) == self.all_UE_num)
                assert(np.size(k_means_r_sum_n, axis=1) == 1)
                assert(np.size(k_means_r_sum_n, axis=0) == self.assigned_BS_num)
                k_means_r = k_means_r / k_means_r_sum_k
                print np.sum(k_means_r, axis=0)
                self.assigned_BS_X = np.sum(k_means_r * self.all_UE_X, axis=1)
                self.assigned_BS_X = np.reshape(self.assigned_BS_X, (-1, 1))
                self.assigned_BS_Y = np.sum(k_means_r * self.all_UE_Y, axis=1)
                self.assigned_BS_Y = np.reshape(self.assigned_BS_Y, (-1, 1))
                self.assigned_BS_position = np.concatenate([self.assigned_BS_X,
                                                           self.assigned_BS_Y],
                                                           axis=1)
                k_means_sigma = np.sqrt(np.reshape(np.sum(k_means_r * distance, axis=1), (-1, 1))
                                        / 2 / k_means_r_sum_n)
                k_means_sigma = np.reshape(k_means_sigma, (-1, 1))
                k_means_pi = k_means_r_sum_n / np.sum(k_means_r_sum_n, axis=0)
                k_means_pi = np.reshape(k_means_pi, (-1, 1))
            '''
        else:
            raise Exception("Unimplementable")

    def set_assinged_BS_as_BS(self):
        self.BS_X = self.assigned_BS_X
        self.BS_Y = self.assigned_BS_Y
        self.BS_position = self.assigned_BS_position

    def plot_assign_BS_scatter(self):
        plt.scatter(self.assigned_BS_X.asnumpy(), self.assigned_BS_Y.asnumpy(), s=10, c="r")
        # plt.scatter(self.hot_point_X.asnumpy(), self.assigned_BS_Y.asnumpy())


if __name__ == '__main__':
    hot_region = HotServiceRegionAssign()
    hot_region.set_region(0.0, 200.0, 0.0, 200.0)
    hot_region.set_hot_density(1.0 / 400)
    hot_region.set_hot_point_distribution('Uniform')
    hot_region.set_hot_point()
    hot_region.set_UE_density(0.1)
    hot_region.set_UE_distribution('Gaussian', 3)
    hot_region.set_UE()
    hot_region.set_hot_point_as_BS()
    hot_region.plot_BS_voronoi()
    hot_region.set_all_UE()
    hot_region.set_assigned_BS_given_all_UE()
    hot_region.plot_assign_BS_scatter()
    hot_region.set_assinged_BS_as_BS()
    hot_region.plot_BS_voronoi()
    # hot_region.plot_UE_scatter()
    plt.show()
    hot_region.plot_UE_scatter()
    hot_region.plot_assign_BS_scatter()
    plt.show()

