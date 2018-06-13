import numpy as np
import matplotlib.pyplot as plt
from .dim2_distance import dim2_distance


class DFSDictByDistance(object):
    def __init__(self, position, distance_thold, ifplot=False):

        # self.distance_logic_: 是否存在连接两个基站的边。
        # self.near_distance_dict: 存放着索引字典由于分簇。
        # self.distance_thold_: 用于判定是否为协作基站的连线。
        # self.base_station_num_: 基站的数量
        # self.dfs_fig： 存储示意图的图像
        self.distance_logic_ = np.array([])
        self.near_distance_dict_ = {}
        self.distance_thold_ = distance_thold
        self.base_station_num_ = np.shape(position)[0]

        # 函数：判断是否存在连接两个基站的边
        self.distance_to_logic(position)

        # 函数：找到用于存放索引的字典
        self.near_distance_means_dict(position, ifplot)

    # dfs_fig = 10

    def distance_to_logic(self, position):
        distance = dim2_distance(position, position.T)
        self.distance_logic_ = distance < self.distance_thold_

    def dfs(self, node, position, ifplot=False):
        if node not in self._visited_node:
            self._visited_node = np.append(self._visited_node, node)
            if ifplot:
                plt.scatter(position[node, 0],
                            position[node, 1],
                            c='lightskyblue')
        for i in range(self.base_station_num_):
            if i not in self._visited_node \
                    and i != node \
                    and self.distance_logic_[node, i] == 1:
                self._visited_node = np.append(self._visited_node, i)
                if ifplot:
                    plt.plot(position[[i, node], 0],
                             position[[i, node], 1],
                             'hotpink')
                    plt.scatter(position[[i, node], 0],
                                position[[i, node], 1],
                                c='lightpink')
                self.dfs(i, position, ifplot)
            else:
                continue

    _visited_node = np.array([], dtype=np.int)

    def near_distance_means_dict(self, position, ifplot=False):
        clusters = 0
        self.near_distance_dict_ = {}
        for i in range(self.base_station_num_):
            if i not in self._visited_all:
                self.dfs(i, position, ifplot)
                self.near_distance_dict_[clusters] = self._visited_node
                self._visited_all = np.append(self._visited_all,
                                              self._visited_node)
                self._visited_node = np.array([], dtype=np.int)
                clusters += 1
            else:
                continue
        self._visited_all = np.array([], dtype=np.int)

    _visited_all = np.array([], dtype=np.int)

    def dfs_plot(self, position):
        self.near_distance_means_dict(position, ifplot=True)
