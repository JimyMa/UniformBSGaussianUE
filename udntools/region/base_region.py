import numpy as np


class BaseRegion(object):
    def __init__(self,
                 x_min,
                 x_max,
                 y_min,
                 y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        '''
        ground_position x_range/atom * y_range/atom * 2 tensor
        '''
        self.ground_position = self.get_ground()

    def get_ground(self, atom=0.5):
        x = np.arange(self.x_min, self.x_max + atom, atom)
        y = np.arange(self.y_min, self.y_max + atom, atom)
        ground_x, ground_y = np.meshgrid(x, y)
        ground_x = ground_x[:, :, np.newaxis]
        ground_y = ground_y[:, :, np.newaxis]
        ground_position = np.concatenate([ground_x, ground_y])
        return ground_position
