from .comp_service_region import CompServiceRegion
import numpy as np
from ..utils.dim2_distance import dim2_distance
from ..channel.small_fade_channel import SmallFadeChannel
from ..channel.large_fade_channel import LargeFadeChannel
from ..utils.pc_theory import get_radius_given_threshold


class UsrCenterCompServiceRegion(CompServiceRegion,
                                 SmallFadeChannel,
                                 LargeFadeChannel):

    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 layer=1, power=1.0, bs_distribution="uniform",
                 if_fix_bs=True,
                 ue_distribution="gaussian", ue_sigma=5.0,
                 path_loss_factor=4.0,
                 small_fade='Rayleigh',
                 p_c_hold=0.6,
                 sir_db_hold=3):

        CompServiceRegion.__init__(self,
                                   x_min, x_max,
                                   y_min, y_max,
                                   bs_number, ue_number,
                                   layer, power,
                                   bs_distribution,
                                   if_fix_bs,
                                   ue_distribution,
                                   ue_sigma,
                                   path_loss_factor,
                                   small_fade)
        self.ue_radius_ = np.array([])
        self.p_c_hold_ = p_c_hold
        self.sir_db_hold_ = sir_db_hold
        self.ue_position_now_ = np.array([])
        self.bs_ue_now_dict_ = {}
        self.cluster_by_usr()

    # 此种用户选择算法被废弃
    # 采用基站数远远小于用户数的情况
    def usr_generator(self):
        self.kill_ue()
        self.ue_position_ = np.zeros([2, self.bs_number_])
        self.ue_number_ = self.bs_number_
        for i in range(self.bs_number_):
            self.ue_position_[:, i] = self.bs_position_[i, :] \
                                      + np.random.randn(2) * self.ue_sigma

    def get_usr_circle_radius(self, position):
        number = np.shape(position)[1]
        self.ue_radius_ = np.zeros(number)
        distance = dim2_distance(self.bs_position_, position)
        bs_p_ue_distance = np.min(distance, axis=0)
        for i in range(number):
            radius = get_radius_given_threshold(bs_p_ue_distance[i],
                                                self.p_c_hold_,
                                                self.sir_db_hold_,
                                                self.path_loss_factor)
            self.ue_radius_[i] = radius

    def cluster_by_usr(self):
        position = np.array([])
        bs_index = np.array([], dtype=np.int)
        self.bs_ue_now_dict_ = {}
        for key, values in self.bs_ue_dict_.items():
            self.bs_ue_now_dict_[key] = np.array([], dtype=np.int)
            if np.size(values) > 0:
                bs_index = np.append(bs_index, key)
                ue_now = np.reshape(self.ue_position_[:, values[0]], (2, 1))
                position = ue_now if np.size(position) == 0 \
                    else np.concatenate([position, ue_now], axis=1)
                self.ue_number_ -= 1
                # self.ue_position_ = np.delete(self.ue_position_, values[0], axis=1)
                self.bs_ue_dict_[key] = np.delete(values, 0)
                self.bs_ue_now_dict_[key] = np.append(self.bs_ue_now_dict_[key], values[0])
        self.get_usr_circle_radius(position)
        self.ue_position_now_ = position

        number = np.size(bs_index)
        distance = dim2_distance(self.bs_position_, position)
        sort_index = np.argsort(-self.ue_radius_)
        radius_sort = self.ue_radius_[sort_index]
        bs_index_sort = bs_index[sort_index]
        position_sort = position[:, sort_index]
        distance_sort = distance[:, sort_index]

        ue_center_set = {}
        ue_map = {}
        for i in range(number):
            ue_map[self.bs_ue_now_dict_[bs_index_sort[i]][0]] = i
            distance_this_ue = distance_sort[:, i]
            distance_this_ue_sort = np.sort(distance_this_ue)
            distance_this_ue_sort_index = np.argsort(distance_this_ue)
            bs_this_ue_index = distance_this_ue_sort_index[distance_this_ue_sort
                                                           < radius_sort[i] + 0.01]
            ue_center_set[i] = bs_this_ue_index

        print("-------------")
        print(ue_center_set)
        print("*************")
        print(ue_map)

        # 下面开始分簇

        # for index, value in enumerate(position):

