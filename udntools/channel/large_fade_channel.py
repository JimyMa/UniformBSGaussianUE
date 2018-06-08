from ..utils.dim2_distance import dim2_distance


class LargeFadeChannel(object):
    def __init__(self, path_loss_factor):
        self.path_loss_factor = path_loss_factor

    def large_fade_factor_matrix(self, bs_position, user_position):
        """
        p_send: a scale or a num_bs * 1 vector
        bs_position: num_bs * 2-dim matrix
        user_position: 2-dim * num_user matrix
        """
        distance = dim2_distance(bs_position, user_position)
        # distance: num_bs * num_user matrix
        return distance ** (-self.path_loss_factor)
