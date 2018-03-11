# -*- coding:utf-8 -*-


import numpy as np
from hot_service_region import HotServiceRegion
from service_region import ServiceRegion
from theory import capacity_theory, capacity_theory2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    density = np.arange(0.0, 0.0051, 0.00001)
    c1 = np.empty(np.shape(density))
    c3 = np.empty(np.shape(density))
    c5 = np.empty(np.shape(density))
    c7 = np.empty(np.shape(density))
    for i, value in enumerate(density):
        c3[i] = capacity_theory(value, 4.0, 3.0)
        c5[i] = capacity_theory(value, 4.0, 5.0)
        c7[i] = capacity_theory(value, 4.0, 7.0)
        c1[i] = capacity_theory2(value, 4.0, 3.0)
    f1, = plt.plot(density, c3)
    f2, = plt.plot(density, c5)
    f3, = plt.plot(density, c7)
    f4, = plt.plot(density, c1)
    plt.xlim(0, 0.005)
    plt.ylim(0, 0.0025)
    plt.xlabel(ur'基站密度(m$^{-2}$)')
    plt.ylabel(ur'单位面积谱效率(bps/Hz/m${^2}$)')
    plt.legend([f1, f2, f3, f4],
               [ur'$\sigma=3.0$理论',
                ur'$\sigma=5.0$理论',
                ur'$\sigma=7.0$理论',
                ur'均匀分布理论'])
    plt.grid(True)
    plt.show()
