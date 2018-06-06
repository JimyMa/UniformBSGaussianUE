from . import pc_gaussian_ue_exp
from . import pc_uniform_ue_exp
import numpy as np
import scipy.integrate as sci


def ase_theory_gaussian_one(lambda_s, alpha, sigma):
    def pc_exp(t):
        return lambda_s * pc_gaussian_ue_exp(t, lambda_s, alpha, sigma)

    return sci.quad(pc_exp, 0.0, 1000)[0]


def ase_theory_uniform_one(lambda_s, alpha):
    def pc_exp(t):
        return lambda_s * pc_uniform_ue_exp(t, alpha)

    return sci.quad(pc_exp, 0.0, 1000)[0]


def ase_theory_gaussian(lambda_s_array, alpha, sigma):
    ase_array = np.zeros(np.shape(lambda_s_array))
    for index, value in enumerate(lambda_s_array):
        ase_array[index] = ase_theory_gaussian_one(value, alpha, sigma)
    return ase_array


def ase_theory_uniform(lambda_s_array, alpha):
    ase_array = np.zeros(np.shape(lambda_s_array))
    for index, value in enumerate(lambda_s_array):
        ase_array[index] = ase_theory_uniform_one(value, alpha)
    return ase_array
