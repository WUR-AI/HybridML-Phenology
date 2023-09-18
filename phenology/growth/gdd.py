import numpy as np


def f_gdd(ts: np.ndarray,
          t_base: float,
          ) -> np.ndarray:
    """
    Function to compute a Growing Degree Day (GDD) estimate for temperature data based on N daily measurements
    :param ts: a numpy ndarray of shape (*X, N), where N is the number of measurements that were obtained in one day
    :param t_base: GDD hyperparameter that dictates at which temperature gdd starts accumulating
    :return: a numpy ndarray of shape (*X) containing GDD estimates for all days
    """
    gdd = np.maximum(0, ts - t_base).sum(axis=-1) / ts.shape[-1]
    return gdd


def f_gdd_minmax(ts: np.ndarray,
                 t_base: float,
                 ) -> np.ndarray:
    """
    Function to compute a Growing Degree Day (GDD) estimate for temperature data based on daily minimum and maximum
    temperature.
    :param ts: a numpy ndarray of shape (*X, N), where N is the number of measurements that were obtained in one day
    :param t_base: GDD hyperparameter that dictates at which temperature gdd starts accumulating
    :return: a numpy ndarray of shape (*X) containing GDD estimates for all days
    """
    t_max = ts.max(axis=-1)
    t_min = ts.min(axis=-1)
    t_mean = (t_max + t_min) / 2
    gdd = t_mean - t_base
    return np.maximum(0, gdd)


def f_gdd_sum(tss: np.ndarray,
              t_base: float,
              ) -> np.ndarray:
    """
    Convenience function for computing the cumulative GDD over a series of days. Assumes the last dimension specifies
    the number of measurements in a day and the second to last dimension the number of days in the series.
    :param tss:
    :param t_base:
    :return:
    """
    return f_gdd(tss, t_base).sum(axis=-1)


def f_gdd_sum_at(tss: np.ndarray,
                 t_base: float,
                 threshold: float,
                 ) -> np.ndarray:
    """

    :param tss:
    :param t_base:
    :param threshold:
    :return:
    """
    assert threshold >= 0
    assert len(tss.shape) == 2
    if len(tss) == 0:
        return np.array(-1)

    gdd = f_gdd(tss, t_base)

    cs = gdd.cumsum(axis=-1)

    ixs, = np.where(cs >= threshold)
    if len(ixs) == 0:
        return np.array(-1)

    return ixs.min(axis=-1)


if __name__ == '__main__':

    _ts = (np.random.rand(24) * 30) - 10
    _tss = (np.random.rand(100, 24) * 30) - 10
    _tsss = (np.random.rand(5, 7, 24) * 30) - 10

    _t_base = 5
    _threshold = 100

    print(_ts)
    print(f_gdd(_ts, _t_base))

    print(f_gdd(_tss, _t_base))

    print(f_gdd_sum(_tss, _t_base))

    print(f_gdd_sum_at(_tss, _t_base, _threshold))
    # print(f_gdd_sum_at(_tsss, _t_base, _threshold))
