from . import BaseRegion
from ..channel import BaseChannel
from ..bs import BaseBS
from ..ue import BaseUE
import numpy as np
from ..utils.dim2_distance import dim2_distance


class ServiceRegion(BaseRegion, BaseChannel, BaseBS, BaseUE):
    def __init__(self, x_min, x_max, y_min, y_max, bs_number, ue_number,
                 loss_factor=4.0, small_fade="Rayleigh", noise="Gaussian", big_fade="no_big_fade",
                 layer=1, power=1.0, bs_distribution="uniform",
                 ue_distribution="uniform"):
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
        BaseChannel.__init__(self, loss_factor, small_fade, noise, big_fade)
        BaseBS.__init__(self, bs_number, layer, power, bs_distribution)
        BaseUE.__init__(self, ue_number, ue_distribution)
        self.set_bs_to_region()
        self.set_ue_to_region()
        self.bs_ue_dict = {}
        # a dict that show which ue belong to which bs
        # key: 0, 1, ..., num_bs
        # value: 0, 1, ..., num_ue belong to the key
        self.select_ue()

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

    _set_bs_to_region_func_dict = {"uniform": set_uniform_bs_to_region}

    def select_ue(self):
        self.bs_ue_dict = {}
        distance = dim2_distance(self.bs_position_, self.ue_position_)
        selected_bs_index = np.argmin(distance, axis=0)
        for i in range(self.bs_number_):
            self.bs_ue_dict[i] = []
        for ue_index, bs_index in enumerate(selected_bs_index):
            self.bs_ue_dict[bs_index].append(ue_index)

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

    _set_ue_to_region_func_dict = {"uniform": set_uniform_ue_to_region}

