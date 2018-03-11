import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def capacity_theory(density, loss_factor, sigma):
    def f(u):
        return 1.0 / (u + 1.0)

    def pc(t, density, loss_factor, sigma):
        d2s2 = density * 2.0 * sigma * sigma
        pi = np.pi
        rou = t ** (2.0/loss_factor) * sci.quad(f, t**(-2.0/loss_factor), np.inf)[0]
        pcv = rou / (1.0 + rou) / ((1.0 + rou) * d2s2 * pi + 1.0) + 1.0 / (1.0 + rou)
        return pcv

    def pc_exp(t):
        a = density * pc(2.0 ** t - 1.0, density, loss_factor, sigma)
        return a

    return sci.quad(pc_exp, 0.0, 1000)[0]


def capacity_theory2(density, loss_factor, sigma):
    def f(u):
        return 1.0 / (u + 1.0)

    def pc(t, density, loss_factor, sigma):
        rou = t ** (2.0/loss_factor) * sci.quad(f, t**(-2.0/loss_factor), np.inf)[0]
        pcv = 1.0 / (1.0 + rou)
        return pcv

    def pc_exp(t):
        a = density * pc(2.0 ** t - 1.0, density, loss_factor, sigma)
        return a

    return sci.quad(pc_exp, 0.0, 1000)[0]


if __name__ == '__main__':
    density = np.arange(0.0, 0.0051, 0.0001)
    c1 = np.empty(np.shape(density))
    c2 = np.empty(np.shape(density))
    for i, value in enumerate(density):
        c1[i] = capacity_theory(value, 4.0, 10.0)
        c2[i] = capacity_theory2(value, 4.0, 3.0)
    plt.plot(density, c1)
    plt.plot(density, c2)
    plt.xlim(0, 0.005)
    plt.ylim(0, 0.005)
    plt.show()
