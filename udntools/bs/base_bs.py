from abc import abstractmethod
import numpy as np

class BaseBS(object):

    def __init__(self,
                 bs_number,
                 layer=1,
                 power=1.0,
                 bs_distribution="uniform",
                 if_fix_bs=True):
        self.bs_number_ = int(bs_number)
        self.bs_layer_ = layer
        self.bs_power_ = power
        self.bs_distribution_ = bs_distribution
        self.bs_position_ = None
        if if_fix_bs:
            self.bs_position_ = np.loadtxt("/home/zoo2/Documents/UDNs/program/udntools/bs/bs_position_.txt")
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
