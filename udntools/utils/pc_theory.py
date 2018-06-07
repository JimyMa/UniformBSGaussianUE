import numpy as np
import scipy.integrate as sci


def rho(t, alpha):
    def f(u):
        return 1 / (u ** (alpha / 2) + 1)
    return np.power(t, 2 / alpha) * sci.quad(f, t **(-2 / alpha), 400)[0]


def pc_gaussian_one_point(t, lambda_s, alpha, sigma):
    return 1 / (1 + rho(t, alpha)) \
            + rho(t, alpha) / (1 + rho(t, alpha)) \
            * (1 / (2 * np.pi * sigma ** 2 * lambda_s * (1 + rho(t, alpha)) + 1))


def pc_gaussian_ue(t, lambda_s, alpha, sigma):
    t = 10 ** (t / 10)
    t = np.reshape(t, (-1))
    pc_array = np.zeros(t.shape)
    for index, value in enumerate(t):
        pc_array[index] = pc_gaussian_one_point(value, lambda_s, alpha, sigma)
    return pc_array


def pc_gaussian_ue_exp(t, lambda_s, alpha, sigma):
    t_exp = 2 ** t - 1
    return pc_gaussian_one_point(t_exp, lambda_s, alpha, sigma)


def pc_uniform_one_point(t, alpha):
    return 1 / (1 + rho(t, alpha))


def pc_uniform_ue(t, alpha):
    t = 10 ** (t / 10)
    t = np.reshape(t, (-1))
    pc_array = np.zeros(t.shape)
    for index, value in enumerate(t):
        pc_array[index] = pc_uniform_one_point(value, alpha)
    return pc_array


def pc_uniform_ue_exp(t, alpha):
    t_exp = 2 ** t - 1
    return pc_uniform_one_point(t_exp, alpha)
