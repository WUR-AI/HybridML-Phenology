import numpy as np


def f_chill_hours(ts: np.ndarray,
                  t_base: float = 7.2,
                  ):
    return ((0 <= ts) & (ts <= t_base)).sum(axis=-1)


if __name__ == '__main__':

    _ts = np.random.rand(365, 24) * 20 - 5

    _ch = f_chill_hours(_ts)

    print(_ts[0])
    print(_ch[0])
    print(_ch)