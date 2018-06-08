import numpy as np
from ..utils.dim2_distance import dim2_distance


class BaseChannel(object):
    def __init__(self,
                 path_loss_factor=4.0,
                 small_fade='Rayleigh',
                 noise='Gaussian'):
        self.path_loss_factor = path_loss_factor
        self.small_fade = small_fade
        self.noise = noise
        self.h_matrix = np.array([])

    # You can not know the influence of the large scale fade if you don't know distance
    # but distance is not a attr of the channel, so it may be a static function

    def set_path_loss_factor(self, loss_factor):
        self.path_loss_factor = loss_factor

    @staticmethod
    def distance_matrix(bs_position, ue_position):
        """
        bs_position: num_bs * 2-dim matrix
        user_position: 2-dim * num_user matrix
        """
        # distance_matrix: num_bs * num_user matrix
        return dim2_distance(bs_position, ue_position)

    def large_fade_power_matrix(self, bs_position, user_position, p_send):
        """
        p_send: a scale or a num_bs * 1 vector
        bs_position: num_bs * 2-dim matrix
        user_position: 2-dim * num_user matrix
        """
        distance = self.distance_matrix(bs_position, user_position)
        # distance: num_bs * num_user matrix
        return p_send * distance ** (-self.path_loss_factor)

    def small_fade_power_matrix(self, large_fade_power_matrix):
        """
        :param large_fade_power_matrix: num_bs * num_user matrix
        :return: num_bs * num_user matrix
        """
        if self.small_fade == 'no_small_fade':
            return large_fade_power_matrix
        elif self.small_fade == 'Rayleigh':
            self.h_matrix = np.random.exponential(1, np.shape(large_fade_power_matrix))
            return self.h_matrix * large_fade_power_matrix

    def power_vector(self, bs_position, user_position, p_send):
        """
        :return: 1 * num_user vector
        """
        large_fade_power_matrix = self.large_fade_power_matrix(bs_position, user_position, p_send)
        small_fade_power_matrix = self.small_fade_power_matrix(large_fade_power_matrix)
        power_vector = np.max(small_fade_power_matrix, axis=0)
        return np.reshape(power_vector, (1, -1))

    def interference_vector(self, bs_position, user_position, p_send):
        """
        :return: 1 * num_user vector
        """
        large_fade_power_matrix = self.large_fade_power_matrix(bs_position, user_position, p_send)
        small_fade_power_matrix = self.small_fade_power_matrix(large_fade_power_matrix)
        power_vector = np.min(small_fade_power_matrix, axis=0)
        sum_power_vector = np.sum(small_fade_power_matrix, axis=0)
        return np.reshape(sum_power_vector-power_vector, (1, -1))

    def sir_vector(self, bs_position, user_position, p_send=1.0):
        large_fade_power_matrix = self.large_fade_power_matrix(bs_position, user_position, p_send)
        small_fade_power_matrix = self.small_fade_power_matrix(large_fade_power_matrix)
        power_vector = small_fade_power_matrix[np.argmax(large_fade_power_matrix, axis=0),
                                               np.arange(0, np.shape(large_fade_power_matrix)[1])]
        sum_power_vector = np.sum(small_fade_power_matrix, axis=0)
        interference_vector = sum_power_vector-power_vector
        return np.reshape(power_vector / interference_vector, (1, -1))



