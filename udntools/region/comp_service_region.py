from . import ServiceRegion
import numpy as np
from ..utils.dfs_dict_by_distance import DFSDictByDistance
from ..channel.base_channel import BaseChannel


class CompServiceRegion(ServiceRegion):

    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 layer=1, power=1.0, bs_distribution="uniform",
                 ue_distribution="uniform", ue_sigma=0,
                 channel=BaseChannel(4.0, 'Rayleigh', 'Gaussian')):
        ServiceRegion.__init__(self, x_min, x_max, y_min, y_max,
                               bs_number, ue_number,
                               layer, power,
                               bs_distribution, ue_distribution, ue_sigma)
        self.cluster_set_ = {}
        self.csi_state = channel

    def cluster_by_dfs(self, distance_thold):
        self.cluster_set_ = {}
        self.cluster_set_ = DFSDictByDistance(self.bs_position_,
                                              distance_thold).near_distance_dict_

    def precoding(self):
        pass
