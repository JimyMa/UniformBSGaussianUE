
import numpy as np


def get_circle(x_ave, y_ave, radius, atom=100):
    position = np.zeros((atom+1, 2))
    for i in range(atom + 1):
        position[i, 0] = x_ave + \
            radius * np.cos((2 * np.pi) * i / atom)

        position[i, 1] = y_ave + \
            radius * np.sin((2 * np.pi) * i / atom)

    return position
