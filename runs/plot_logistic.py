


import numpy as np
from matplotlib import pyplot as plt


class Logistic:

    def __init__(self, alpha, beta, omega):
        self._alpha = alpha
        self._beta = beta
        self._omega = omega

    def __call__(self, x: np.ndarray):
        return self._omega / (1 + np.exp(-self._alpha * (x - self._beta)))


class DoubleLogistic:

    def __init__(self,
                 alpha_1,
                 alpha_2,
                 beta_1,
                 beta_2,
                 omega_1,
                 omega_2,
                 ):
        self._l1 = Logistic(alpha_1, beta_1, omega_1)
        self._l2 = Logistic(alpha_2, beta_2, omega_2)

    def __call__(self, x: np.ndarray):
        return self._l1(x) + self._l2(x)


if __name__ == '__main__':

    _alpha_1 = 5
    _alpha_2 = 1
    _beta_1 = 2
    _beta_2 = 14
    _omega_1 = 1
    _omega_2 = -2

    # _alpha_1 = 100
    # _alpha_2 = 3e-1
    # _alpha_3 = 1e-1
    # _beta_1 = 100
    # _beta_2 = 100
    # _beta_3 = 100
    # _omega_1 = 1
    # _omega_2 = 1
    # _omega_3 = 1

    # _alpha_1 = 50
    # _alpha_2 = 10
    # _alpha_3 = 1
    # _beta_1 = 1
    # _beta_2 = 1
    # _beta_3 = 1
    # _omega_1 = 1
    # _omega_2 = 1
    # _omega_3 = 1

    f = DoubleLogistic(
        _alpha_1,
        _alpha_2,
        _beta_1,
        _beta_2,
        _omega_1,
        _omega_2,
    )

    f_1 = Logistic(
        _alpha_1,
        _beta_1,
        _omega_1,
    )

    f_2 = Logistic(
        _alpha_2,
        _beta_2,
        _omega_2,
    )

    # ALPHA1 = 3
    # ALPHA2 = 3
    # BETA1 = 0
    # BETA2 = 7.2
    # OMEGA1 = 1
    # OMEGA2 = -1
    f_log_chill_hours = DoubleLogistic(
        3,
        3,
        0,
        7.2,
        1,
        -1,
    )

    # f_3 = Logistic(
    #     _alpha_3,
    #     _beta_3,
    #     _omega_3,
    # )

    x = np.arange(-10, 30, 0.1)
    # x = np.arange(0, 2, 0.01)
    # x = np.arange(0, 200, 1)

    from phenology.chill.chill_hours import f_chill_hours
    from phenology.chill.utah_model import utah_chill

    fig, ax = plt.subplots(3, figsize=(8, 15))

    # ax[0].plot(x, f(x), '--', c='black')
    ax[0].plot(x, f_log_chill_hours(x), c='black')

    # ax[0].plot(x, f_chill_hours(x.reshape(-1, 1)), label='Chill Hours')
    # ax[0].plot(x, utah_chill(x.reshape(-1, 1)), label='Utah Chill', c='black')

    # ax[0].plot(x, f_1(x), label='s1', c='black')
    # ax[0].plot(x, f_2(x), '--', label='s1', c='black')
    # ax[0].plot(x, f_3(x), ':', label='s1', c='black')
    # ax[1].plot(x, f_2(x), label='s2')
    # ax[2].plot(x, f(x), label='s')

    # ax[0].set_title(f'f({_alpha_1}, {_beta_1}, {_omega_1})')
    # ax[1].set_title(f'f({_alpha_2}, {_beta_2}, {_omega_2})')
    # ax[2].set_title(f'g')
    #
    # ax[0].set_xlabel('unit sum')
    # ax[0].set_ylabel('stage transition')

    ax[0].set_xlabel('Temperature (°C)')
    # ax[0].set_ylabel('Thermal time')

    # ax[0].legend()

    # plt.show()
    plt.savefig('plot_chill_operators_temp.png')



