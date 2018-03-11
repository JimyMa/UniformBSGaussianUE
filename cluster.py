# -*- coding:utf-8 -*-

import numpy as np
from hot_service_region import HotServiceRegion
from service_region import ServiceRegion
import matplotlib.pyplot as plt
if __name__ == '__main__':
    region = HotServiceRegion()
    region.set_region(0, 200, 0, 200)
    region.set_hot_density(1.0/400)
    region.set_P_sum(10.0)
    region.set_small_fade('Rayleigh')
    region.set_hot_point_distribution('Uniform')
    region.set_hot_point()
    region.set_UE_density(0.1)
    region.set_UE_distribution('Gaussian', 5.0)
    region.set_UE()
    region.set_all_UE()
    region.set_hot_point_as_BS()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    region.plot_UE_scatter()
    plt.show()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(region.all_UE_X, region.all_UE_Y, s=1)
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=region.hot_point_num,
                          covariance_type='spherical',
                          init_params='kmeans',
                          verbose=1,
                          warm_start=1,
                          max_iter=100)
    gmm.fit(region.all_UE_position.T)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(region.all_UE_X, region.all_UE_Y, s=1)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1])
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    region.plot_BS_voronoi(ax)
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    region.BS_position = gmm.means_
    region.plot_BS_voronoi(ax)
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.show()
    region.set_hot_point_as_BS()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    f1 = region.plot_hot_UE_SIR_ccdf()
    region.BS_position = gmm.means_
    region.BS_X = region.BS_position[:, 0]
    region.BS_Y = region.BS_position[:, 1]
    f2 = region.plot_hot_UE_SIR_ccdf()
    region.set_BS()
    f3 = region.plot_hot_UE_SIR_ccdf()
    plt.xlim(-10, 20)
    plt.ylim(0, 1)
    plt.legend([f1, f2, f3], [u'热点中心作为基站仿真',
                              u'采用EM算法部署的基站仿真',
                              u'随机部署的基站仿真'])
    plt.xlabel(u'信干比(dB)')
    plt.ylabel(u'覆盖率')
    plt.grid(True)
    plt.savefig('deploy.png', bbox_inches='tight')
    plt.show()
