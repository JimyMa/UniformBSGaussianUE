from .service_region import ServiceRegion
import numpy as np
from ..utils.dim2_distance import dim2_distance
from ..utils.dfs_dict_by_distance import DFSDictByDistance
from ..channel.small_fade_channel import SmallFadeChannel
from ..channel.large_fade_channel import LargeFadeChannel


class CompServiceRegion(ServiceRegion, SmallFadeChannel, LargeFadeChannel):

    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 layer=1, power=1.0, bs_distribution="uniform",
                 ue_distribution="gaussian", ue_sigma=0,
                 path_loss_factor=4.0,
                 small_fade='Rayleigh'):
        ServiceRegion.__init__(self, x_min, x_max, y_min, y_max,
                               bs_number, ue_number,
                               layer, power,
                               bs_distribution, ue_distribution, ue_sigma)
        LargeFadeChannel.__init__(self, path_loss_factor)
        SmallFadeChannel.__init__(self,
                                  self.bs_number_,
                                  self.ue_number_,
                                  small_fade)
        self.cluster_set_ = {}
        self.cluster_bs_position_ = {}
        self.cluster_ue_position_ = {}
        self.cluster_ue_set_ = {}
        self.sir_array = np.array([])
        self.sir_array_db = np.array([])

    def cluster_by_dfs(self, distance_thold):
        self.cluster_set_ = {}
        self.cluster_bs_position_ = {}
        self.cluster_set_ = DFSDictByDistance(self.bs_position_,
                                              distance_thold).near_distance_dict_
        for key, values in self.cluster_set_.items():
            self.cluster_bs_position_[key] = \
                np.reshape(self.bs_position_[values, :], (-1, 2))

    def get_cluster_ue_position(self):
        self.cluster_ue_position_ = {}
        self.cluster_ue_set_ = {}
        for key, values in self.cluster_set_.items():
            self.cluster_bs_position_[key] = self.bs_position_[values, :]
            self.cluster_ue_set_[key] = np.array([], dtype=np.int)
            for bs_index in values:
                if np.size(self.bs_ue_dict_[bs_index]) != 0:
                    ue_position_this_bs = \
                        np.reshape(self.ue_position_[:, self.bs_ue_dict_[bs_index]], (2, -1))

                    self.cluster_ue_position_[key] = \
                        np.concatenate([self.cluster_ue_position_[key],
                                        ue_position_this_bs], axis=1)  \
                        if key in self.cluster_ue_position_ \
                        else ue_position_this_bs
                    self.cluster_ue_set_[key] = np.append(self.cluster_ue_set_[key],
                                                          self.bs_ue_dict_[bs_index])

    def zfbf_equal_allocation(self):
        self.get_cluster_ue_position()
        self.generate_h_matrix()
        distance_factor = dim2_distance(self.bs_position_, self.ue_position_)
        large_loss_factor = distance_factor ** (-self.path_loss_factor)
        small_loss_factor = large_loss_factor * self.h_square_matrix_
        power_gain_factor = small_loss_factor.T
        sig_gain_factor = self.h_matrix_ * distance_factor ** (-2.0)
        sig_gain_factor = sig_gain_factor.T
        for key, values in self.cluster_set_.items():
            # print(key)
            # print('*********')
            while np.size(self.cluster_ue_set_[key]) != 0:
                ue_num_now = np.size(self.cluster_ue_set_[key])
                # print(ue_num_now)
                # print('--------')
                bs_num = np.size(self.cluster_set_[key])
                ue_index = np.arange(0, ue_num_now, 1)
                if ue_num_now >= bs_num:
                    ue_chosen = np.sort(np.random.choice(ue_index, bs_num, replace=False))
                else:
                    ue_chosen = ue_index
                # print(ue_chosen)
                ue_index_now = self.cluster_ue_set_[key][ue_chosen]
                ue_num_in_serve = np.size(ue_num_now)
                self.cluster_ue_position_[key] = \
                    np.delete(self.cluster_ue_position_[key], ue_chosen, axis=1)
                self.cluster_ue_set_[key] = \
                    np.delete(self.cluster_ue_set_[key], ue_chosen)
                # print(values.dtype)
                # print(ue_index_now.dtype)
                # print(values)
                # print(h_matrix.shape)
                sig_g_inner = \
                    np.reshape(sig_gain_factor[ue_index_now, :],
                               (ue_num_in_serve, -1))[:, values]
                sig_g_inner = np.reshape(sig_g_inner, (ue_num_in_serve, bs_num))
                w_inner = np.array(np.mat(sig_g_inner).I)
                w_square_inner = w_inner ** 2
                p_received = 1 / np.max(np.sum(w_square_inner, axis=1))
                i_received = \
                    np.sum(np.delete(small_loss_factor[:, ue_index_now], values, axis=0),
                           axis=0)
                sir = p_received / i_received
                # print(np.size(self.cluster_ue_set_[key]))
                # print('......')
                self.sir_array = np.append(self.sir_array, sir)

    def sir_array_sim(self, iteration=10):
        self.sir_array = np.array([])
        for i in range(iteration):
            self.set_ue_to_region()
            self.select_ue()
            self.zfbf_equal_allocation()
    # 这是一种用户选择方式
    # 区域联合起来了
    # 假设一簇内有 l 个基站
    # 则随机选择区域内的 l 个用户进行服务
    # 该方法被废弃
    # 因为与不采用联合传输优化时的场景不相对应
    # 修改为
    # 区域内的基站满载
    # 即区域内每个基站对应的服务区域内随机选择一个用户
    # 一共 n 个基站，n 个用户
    # 基站根据上面 cluster 方法得到的字典进行分簇

    '''
    def full_loaded(self):
        for key, values in self.cluster_bs_position_.items():
            if key in self.cluster_ue_position_all_:
                if np.size(self.cluster_ue_position_all_[key]) \
                 < np.size(self.cluster_bs_position_[key]):
                    return False
            else:
                return False

        return True

    def user_set_when_many(self):
        self.kill_ue()
        bs_index = 0
        while self.full_loaded() is not True:
            ue_locate = self.bs_position_[bs_index, :] + np.random.randn(2) * self.ue_sigma
            ue_locate = np.reshape(ue_locate, (2, 1))
            bs_index = (bs_index + 1) % self.bs_number_
            distance = np.reshape(dim2_distance(self.bs_position_, ue_locate),
                                  (self.bs_number_, -1))
            selected_bs_by_ue = np.argmin(distance, axis=0)
            for key, comp_set in self.cluster_set_.items():
                for bs_index_in_set in comp_set:
                    if bs_index_in_set == selected_bs_by_ue:
                        if key in self.cluster_ue_position_all_:
                            self.cluster_ue_position_all_[key] = \
                                np.concatenate([self.cluster_ue_position_all_[key],
                                                ue_locate], axis=1)
                        else:
                            self.cluster_ue_position_all_[key] = ue_locate
        for key, values in self.cluster_ue_position_all_.items():
            bs_num_in_set = np.size(self.cluster_set_[key])
            ue_index = np.arange(0, bs_num_in_set)
            self.cluster_ue_position_service_[key] = \
                self.cluster_ue_position_all_[key][:, ue_index]
            self.cluster_ue_position_service_[key] = \
                np.reshape(self.cluster_ue_position_service_[key],
                           (2, -1))
    '''

    # 这是一种用户选择方式
    # 区域联合起来了
    # 区域内的基站满载
    # 即区域内每个基站对应的服务区域内随机选择一个用户
    # 一共 n 个基站，n 个用户
    # 基站根据上面 cluster 方法得到的字典进行分簇
    # 这种方法被废弃
    # 因为要考虑到所有的生成的用户都要被服务，这是公平性的原则
    # 现在改为，所有生成的用户均被服务但是采用联合的方式
    # 用户选择为在簇内随机的选择
    '''
    def user_generate(self):
        self.kill_ue()
        flag_v = np.zeros(self.bs_number_)
        # 标记是否基站被使用
        self.ue_position_ = np.zeros((2, self.bs_number_))
        bs_index = 0
        count = 0

        while np.sum(flag_v) != self.bs_number_:
            count += 1
            ue_locate = self.bs_position_[bs_index, :] + \
                        np.random.randn(2) * self.ue_sigma
            ue_locate = np.reshape(ue_locate, (2, 1))
            bs_index = (bs_index + 1) % self.bs_number_
            distance = np.reshape(dim2_distance(self.bs_position_, ue_locate),
                                  (self.bs_number_, -1))
            selected_bs_by_ue = np.argmin(distance, axis=0)
            if flag_v[selected_bs_by_ue] == 0:
                self.ue_position_[:, selected_bs_by_ue] = ue_locate
                flag_v[selected_bs_by_ue] = 1
            self.ue_number_ = self.bs_number_
        for key, values in self.cluster_set_.items():
            self.cluster_ue_position_[key] = \
                np.reshape(self.ue_position_[:, values], (2, -1))
        print(count)
    
    # 在联合传输中使用 ZFBF 去进行干扰管理算法
    def zfbf_equal_allocation(self):
        self.generate_h_matrix()
        p_factor_array = np.zeros(self.bs_number_)
        p_receive_dict = {}
        for key, values in self.cluster_set_.items():
            w_matrix_in_cluster = np.array(np.mat(self.h_matrix_[values, :][:, values]).I)
            w_matrix_in_cluster_square = w_matrix_in_cluster ** 2
            w_matrix_in_cluster_square_sum = np.sum(w_matrix_in_cluster_square, axis=1)
            p_receive = 1 / np.max(w_matrix_in_cluster_square_sum)
            p_factor_array[values] = p_receive * w_matrix_in_cluster_square_sum
            p_receive_dict[key] = p_receive
        bs_ue_distance = dim2_distance(self.bs_position_, self.ue_position_)
        p_factor_array = np.reshape(p_factor_array, (-1, 1))
        large_loss_array = p_factor_array * (bs_ue_distance ** (- self.path_loss_factor))
        small_large_loss_array = self.h_square_matrix_ * large_loss_array
        if (small_large_loss_array < 0).any():
            print('error')

        for key, values in self.cluster_set_.items():
            received_power = p_receive_dict[key]
            #print(np.sum(small_large_loss_array[:, values], axis=0))
            #print(np.sum(small_large_loss_array[values, values], axis=0))
            inference_power = np.sum(small_large_loss_array[:, values], axis=0) - \
                              np.sum(small_large_loss_array[:, values][values, :], axis=0)
            sir = received_power / inference_power
            if (sir < 0).any():
                print(sir)
            self.sir_array = np.append(self.sir_array, sir)
            sir_db = 10 * np.log10(sir / 10)
            self.sir_array_db = np.append(self.sir_array_db, sir_db)

    def sir_array_sim(self, iter=10):
        self.sir_array = np.array([])
        self.sir_array_db = np.array([])
        for i in range(iter):
            self.user_generate()
            self.zfbf_equal_allocation()
    '''


