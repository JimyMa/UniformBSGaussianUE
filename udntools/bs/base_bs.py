from abc import abstractmethod


class BaseBS(object):

    def __init__(self,
                 bs_number,
                 layer=1,
                 power=1.0,
                 bs_distribution="uniform"):
        self.bs_number_ = int(bs_number)
        self.bs_layer_ = layer
        self.bs_power_ = power
        self.bs_distribution_ = bs_distribution
        self.bs_position_ = None
        # bs_position: bs_nums * 2-dim matrix

    @abstractmethod
    def set_bs_to_region(self):
        pass

    @abstractmethod
    def set_uniform_bs_to_region(self):
        pass

    @abstractmethod
    def select_ue(self):
        pass
