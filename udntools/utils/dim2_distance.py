import numpy as np


def dim2_distance(matrix_a, matrix_b):
    """
    distance of two 2dim points
    :param matrix_a: num_samplesA * 2 matrix
    :param matrix_b: 2 * num_samplesA matrix
    :return: num_samplesA * num_samplesB matrix
    """
    distance_vector = matrix_a[:, :, np.newaxis] - matrix_b[np.newaxis, :, :]
    distance = np.sqrt(distance_vector[:, 0, :] ** 2.0
                       + distance_vector[:, 1, :] ** 2.0)
    # distance: num_samplesA * num_samplesB matrix
    return distance
