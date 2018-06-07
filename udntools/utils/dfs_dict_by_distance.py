import numpy as np
import matplotlib.pyplot as plt
from . import dim2_distance


class DFSDictByDistance(object):
    def __init__(self, position, distance_thold):

        # self.distance_logic_: 是否存在连接两个基站的边。
        # self.near_distance_dict: 存放着索引字典由于分簇。
        # self.distance_thold_: 用于判定是否为协作基站的连线。
        # self.base_station_num_: 基站的数量
        self.distance_logic_ = np.array([])
        self.near_distance_dict_ = {}
        self.distance_thold_ = distance_thold
        self.base_station_num_ = np.shape(position)[0]

        # 函数：判断是否存在连接两个基站的边
        self.distance_to_logic(position)

        # 函数：找到用于存放索引的字典
        self.near_distance_means_dict()

    def distance_to_logic(self, position):
        distance = dim2_distance(position, position.T)
        self.distance_logic_ = distance > self.distance_thold_

    def dfs(self, node):
        if node not in self._visited_node:
            self._visited_node = np.append(self._visited_node, node)
        for i in range(self.base_station_num_):
            if i not in self._visited_node \
                    and i != node \
                    and self.distance_logic_[node, i] == 1:
                np.append(self._visited_node, i)
                # plt.plot(position[[i, node], 0], position[[i, node], 1])
                self.dfs(i)
            else:
                continue

    _visited_node = np.array([])

    def near_distance_means_dict(self):
        clusters = 0
        for i in range(self.base_station_num_):
            if i not in self._visited_all:
                self.dfs(i)
                self.near_distance_dict_[clusters] = self._visited_node
                self._visited_all = np.append(self._visited_all,
                                              self._visited_node)
                self._visited_node = np.array([])
                clusters += 1
            else:
                continue
        self._visited_all = np.array([])

    _visited_all = np.array([])



