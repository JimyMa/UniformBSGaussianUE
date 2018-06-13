import numpy as np


class SmallFadeChannel(object):
    def __init__(self,
                 bs_num,
                 ue_num,
                 small_fade='Rayleigh'):
        self.small_fade = small_fade
        self.h_matrix = np.array([])
        self.bs_number_ = bs_num
        self.ue_number_ = ue_num
        self.generate_h_matrix()

    def generate_h_matrix(self):
        self.h_matrix = np.random.exponential(1, (self.bs_number_,
                                                  self.ue_number_))