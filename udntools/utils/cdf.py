import numpy as np


def cdf_y_axis(x_axis, data):
    data = np.reshape(data, (-1))
    x_axis = np.reshape(x_axis, (-1))
    data_sort = np.sort(data)
    cdf = []
    x_axis_size = np.size(x_axis)
    index_now = 0
    total = np.size(data_sort)
    for index, value in enumerate(data_sort):
        if value > x_axis[index_now]:
            cdf.append((index + 1) / total)
            if index_now < x_axis_size - 1:
                index_now += 1
            else:
                break
    if index_now == 0:
        return np.zeros(np.shape(x_axis))
    while index_now < x_axis_size:
        cdf.append(cdf[-1])
        index_now += 1
    return np.array(cdf)
