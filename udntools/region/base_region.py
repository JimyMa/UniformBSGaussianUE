import numpy as np


class BaseRegion(object):

    _atom = 0.5

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
        self.ground_position_ = self.get_ground()

    def get_ground(self):
        x = np.arange(self.x_min, self.x_max + self._atom, self._atom)
        y = np.arange(self.y_min, self.y_max + self._atom, self._atom)
        ground_x, ground_y = np.meshgrid(x, y)
        ground_x = ground_x[:, :, np.newaxis]
        ground_y = ground_y[:, :, np.newaxis]
        ground_position = np.concatenate([ground_x, ground_y], axis=2)
        return ground_position

    def set_atom(self, atom):
        self._atom = atom

    def get_atom(self):
        return self._atom
