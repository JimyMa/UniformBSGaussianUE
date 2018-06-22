import numpy as np
import scipy.integrate as sci


def rho(t, alpha):
    def f(u):
        return 1.0 / (u ** (alpha / 2.0) + 1)
    t_size = np.size(t)
    t_shape = np.shape(t)

    t_reshape = np.reshape(t, (-1))

    rho_value = np.zeros(t_size)
    for i in range(t_size):
        rho_value[i] = np.power(t_reshape[i], 2.0 / alpha) \
                       * sci.quad(f, t_reshape[i] **(-2.0 / alpha), 10000.0)[0]
    return np.reshape(rho_value, t_shape)


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


def p_coverage(r_p, r_i, sir_db, alpha):
    sir = 10 ** (sir_db / 10)
    sir_real = (r_p / r_i) ** alpha * sir
    rho_value = rho(sir_real, alpha)
    return np.exp(-np.pi * r_i ** 2 * 0.01 * rho_value)


def get_radius_given_threshold(r_p, p_c, sir_db, alpha):
    r_i = r_p
    while p_coverage(r_p, r_i, sir_db, alpha) < p_c:
        r_i += 0.1

    return r_i
