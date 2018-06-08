from . import ServiceRegion
import numpy as np
from ..utils.dim2_distance import dim2_distance
from ..utils.dfs_dict_by_distance import DFSDictByDistance
from ..channel.small_fade_channel import SmallFadeChannel
from ..channel.large_fade_channel import LargeFadeChannel


class CompServiceRegion(ServiceRegion, SmallFadeChannel, LargeFadeChannel):

    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 layer=1, power=1.0, bs_distribution="uniform",
                 ue_distribution="uniform", ue_sigma=0,
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
        self.cluster_ue_position_service_ = {}
        self.cluster_ue_position_all_ = {}

    def cluster_by_dfs(self, distance_thold):
        self.cluster_set_ = {}
        self.cluster_bs_position_ = {}
        self.cluster_set_ = DFSDictByDistance(self.bs_position_,
                                              distance_thold).near_distance_dict_
        for key, value in self.cluster_set_.items():
            self.cluster_bs_position_[key] = np.reshape(self.bs_position_[value, :], (-1, 2))

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

    def precoding(self):
        pass
