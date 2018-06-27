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
                 sir_db_hold=3.0,
                 max_bs_number=3):

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
        self.max_bs_number_ = max_bs_number
        self.ue_number_now_ = 0
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
                # self.ue_position_ = np.delete(self.ue_position_, values[0], axis=1)
                self.bs_ue_dict_[key] = np.delete(values, 0)
                self.bs_ue_now_dict_[key] = np.append(self.bs_ue_now_dict_[key], values[0])
        if np.size(position) != 0:
            self.get_usr_circle_radius(position)
            self.ue_position_now_ = position

            number = np.size(bs_index)
            self.ue_number_now_ = number
            distance = dim2_distance(self.bs_position_, position)
            sort_index = np.argsort(-self.ue_radius_)
            radius_sort = self.ue_radius_[sort_index]
            bs_index_sort = bs_index[sort_index]
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

            '''
            print("-------------")
            print(ue_center_set)
            print("*************")
            print(ue_map)
            print("@@@@@@@@@@@")
            print(self.bs_ue_now_dict_)
            '''

            # 下面开始分簇
            # 表示用户已经被分簇

            user_clustered_array = np.array([], dtype=np.int)
            bs_clustered_array = np.array([], dtype=np.int)
            self.cluster_set_ = {}
            self.cluster_ue_set_ = {}
            self.cluster_bs_position_ = {}
            self.cluster_ue_position_ = {}
            count = 0
            for key, values in ue_center_set.items():
                # print(values)
                if np.size(values) > self.max_bs_number_:
                    self.cluster_set_[count] = values[0:self.max_bs_number_]
                    self.cluster_bs_position_[count] = \
                        self.bs_position_[values[0:self.max_bs_number_]]

                elif self.max_bs_number_ >= np.size(values) > 0:
                    self.cluster_set_[count] = values
                    self.cluster_bs_position_[count] = \
                        self.bs_position_[values]

                else:
                    continue

                self.cluster_ue_set_[count] = np.array([], dtype=np.int)
                self.cluster_ue_position_[count] = np.array([])
                # 删除对应基站的用户
                for value in self.cluster_set_[count]:
                    # print(ue_map[self.bs_ue_now_dict_[value][0]])
                    if np.size(self.bs_ue_now_dict_[value]) > 0:
                        ue_center_set[ue_map[self.bs_ue_now_dict_[value][0]]] = \
                            np.array([], dtype=np.int)
                        self.cluster_ue_set_[count] = np.append(self.cluster_ue_set_[count],
                                                                self.bs_ue_now_dict_[value])
                    '''
                    if value in bs_clustered_array:
                        print("error")
                        print(value)
                        print(count)
                    '''

                self.cluster_ue_position_[count] = \
                    self.ue_position_[:, self.cluster_ue_set_[count]]

                bs_clustered_array = np.append(bs_clustered_array,
                                               self.cluster_set_[count])

                user_clustered_array = np.append(user_clustered_array,
                                                 self.cluster_ue_set_[count])

                # 删除相应用户的基站
                for value in self.cluster_set_[count]:
                    for key_2, values_2 in ue_center_set.items():
                        ue_center_set[key_2] = \
                            np.delete(values_2, np.argwhere(values_2 == value))

                # print(ue_center_set)
                count += 1

            '''
            print("$$$$$$$$$$$$")
            print(np.sort(bs_clustered_array))
            print("&&&&&&&&&&&&&&")
            print(np.size(user_clustered_array))
    
            print(self.cluster_set_)
            print(ue_center_set)
            '''
        else:
            self.ue_number_now_ = 0

        # print(self.ue_number_now_)

    def zfbf_equal_allocation(self):
        self.generate_h_matrix()
        distance_factor = dim2_distance(self.bs_position_, self.ue_position_)
        large_loss_factor = distance_factor ** (-self.path_loss_factor)
        small_loss_factor = large_loss_factor * self.h_square_matrix_
        sig_gain_factor = self.h_matrix_ * distance_factor ** (-2.0)
        sig_gain_factor = sig_gain_factor.T
        # print(self.cluster_set_)

        self.cluster_by_usr()
        while self.ue_number_now_ > 0:
            # print(self.cluster_ue_set_)
            for key, values in self.cluster_set_.items():
                # print(self.cluster_ue_set_[key])
                # print(self.cluster_set_[key])
                sig_g_inner \
                    = sig_gain_factor[:, self.cluster_set_[key]][self.cluster_ue_set_[key], :]
                ue_number_in_serve = np.shape(self.cluster_ue_set_[key])[0]
                bs_number = np.shape(self.cluster_set_[key])[0]
                ue_index_now = self.cluster_ue_set_[key]

                sig_g_inner = np.reshape(sig_g_inner, (ue_number_in_serve, bs_number))
                w_inner = np.array(np.mat(sig_g_inner).I)
                w_square_inner = w_inner ** 2
                p_received = 1 / np.max(np.sum(w_square_inner, axis=1))
                i_received = \
                    np.sum(np.delete(small_loss_factor, values, axis=0),
                           axis=0)[ue_index_now]
                sir = p_received / i_received
                sir_db = 10 * np.log10(sir)
                # print(np.size(self.cluster_ue_set_[key]))
                # print('......')
                self.sir_array = np.append(self.sir_array, sir)
                self.sir_db_array = np.append(self.sir_db_array, sir_db)
            self.cluster_by_usr()
