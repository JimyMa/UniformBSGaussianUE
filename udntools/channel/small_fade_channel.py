import numpy as np


class SmallFadeChannel(object):
    def __init__(self,
                 bs_num,
                 ue_num,
                 small_fade='Rayleigh'):
        self.small_fade = small_fade
        self.h_matrix_ = np.array([])
        self.h_square_matrix_ = np.array([])
        self.bs_number_ = bs_num
        self.ue_number_ = ue_num
        self.generate_h_matrix()

    def generate_h_matrix(self):
        h_matrix_i = np.random.randn(self.bs_number_, self.ue_number_)
        h_matrix_q = np.random.randn(self.bs_number_, self.ue_number_)
        self.h_square_matrix_ = h_matrix_i ** 2 + h_matrix_q ** 2
        self.h_matrix_ = np.sqrt(self.h_square_matrix_)
