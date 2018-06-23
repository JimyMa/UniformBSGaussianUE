from . import BaseRegion
from ..channel import BaseChannel
from ..bs import BaseBS
from ..ue import BaseUE
import numpy as np
from ..utils.dim2_distance import dim2_distance


class ServiceRegion(BaseRegion, BaseBS, BaseUE):
    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 layer=1, power=1.0, bs_distribution="single_circle",
                 ue_distribution="gaussian", ue_sigma=0,
                 if_fix_bs=True,
                 bs_radius_1=50):
        """
        x_min(need to be assigned)
        x_max(need to be assigned)
        y_min(need to be assigned)
        y_max(need to be assigned)
        path_loss_factor(4.0)
        small_fade("Rayleigh")
        noise("Gaussian")
        big_fade("no_big_fade")
        bs_number(need to be assigned)
        layer(1)
        power(1.0)
        distribution("uniform")
        ue_number(need to be assigned)
        distribution("uniform")
        """
        BaseRegion.__init__(self, x_min, x_max, y_min, y_max)
        BaseBS.__init__(self, bs_number, layer, power, bs_distribution, if_fix_bs)
        BaseUE.__init__(self, ue_number, ue_distribution, ue_sigma)
        self.bs_radius_1_ = bs_radius_1
        if not if_fix_bs:
            self.set_bs_to_region()
        self.set_ue_to_region()
        self.bs_ue_dict_ = {}
        # a dict that show which ue belong to which bs
        # key: 0, 1, ..., num_bs
        # value: 0, 1, ..., num_ue belong to the key
        self.select_ue()

    def set_bs_number(self, number, fresh_ue=False):
        self.bs_number_ = number
        self.set_bs_to_region()
        if fresh_ue:
            self.kill_ue()
            self.set_ue_to_region()

    def set_bs_radius_1(self, radius, fresh_ue=False):
        self.bs_radius_1_ = radius
        self.set_bs_to_region()
        if fresh_ue:
            self.kill_ue()
            self.set_ue_to_region()

    def set_ue_sigma(self, sigma):
        self.ue_sigma = sigma
        self.kill_ue()
        self.set_ue_to_region()
        self.bs_ue_dict_ = {}
        self.select_ue()

    def set_ue_distribution(self, distribution):
        if distribution != self.ue_distribution_:
            self.kill_ue()
            self.ue_distribution_ = distribution
            self.set_ue_to_region()
            self.select_ue()

    def kill_ue(self):
        self.ue_position_ = np.array([])

    def set_bs_to_region(self):
        return self._set_bs_to_region_func_dict.get(self.bs_distribution_)(self)

    def set_uniform_bs_to_region(self):
        bs_position_x = np.random.uniform(self.x_min,
                                          self.x_max,
                                          (self.bs_number_, 1))
        bs_position_y = np.random.uniform(self.y_min,
                                          self.y_max,
                                          (self.bs_number_, 1))
        self.bs_position_ = np.concatenate([bs_position_x, bs_position_y], axis=1)

    def set_single_circle_bs_to_region(self):
        self.bs_position_ = np.zeros((self.bs_number_, 2))
        x_ave = (self.x_min + self.x_max) / 2
        y_ave = (self.y_min + self.y_max) / 2
        for i in range(self.bs_number_):
            self.bs_position_[i, 0] = x_ave + \
                self.bs_radius_1_ * np.cos((2 * np.pi) / self.bs_number_ * i)
            self.bs_position_[i, 1] = y_ave + \
                self.bs_radius_1_ * np.sin((2 * np.pi) / self.bs_number_ * i)

    _set_bs_to_region_func_dict = {"uniform": set_uniform_bs_to_region,
                                   "single_circle": set_single_circle_bs_to_region}

    def select_ue(self):
        self.bs_ue_dict_ = {}
        distance = dim2_distance(self.bs_position_, self.ue_position_)
        selected_bs_index = np.argmin(distance, axis=0)
        for i in range(self.bs_number_):
            self.bs_ue_dict_[i] = np.array([], dtype=np.int)
        for ue_index, bs_index in enumerate(selected_bs_index):
            self.bs_ue_dict_[bs_index] = np.append(self.bs_ue_dict_[bs_index], ue_index)

    def set_ue_to_region(self):
        return self._set_ue_to_region_func_dict.get(self.ue_distribution_)(self)

    def set_uniform_ue_to_region(self):
        ue_position_x = np.random.uniform(self.x_min,
                                          self.x_max,
                                          (1, self.ue_number_))
        ue_position_y = np.random.uniform(self.y_min,
                                          self.y_max,
                                          (1, self.ue_number_))
        self.ue_position_ = np.concatenate([ue_position_x, ue_position_y], axis=0)

    def set_gaussian_ue_to_region(self):
        self.ue_position_ = np.zeros([2, self.ue_number_])
        for i in range(0, self.ue_number_):
            self.ue_position_[:, i] = self.bs_position_[i % self.bs_number_, :] + np.random.randn(2) * self.ue_sigma
            # self.ue_position_[1, i] = self.bs_position_[i % self.bs_number_, 1] + np.random.rand(1) * self.ue_sigma

    _set_ue_to_region_func_dict = {"uniform": set_uniform_ue_to_region,
                                   "gaussian": set_gaussian_ue_to_region}
