import numpy as np


def utah_chill(ts: np.ndarray):

    bin_0 = (ts <= 1.4).sum(axis=-1).astype(ts.dtype)
    bin_1 = ((1.4 < ts) & (ts <= 2.4)).sum(axis=-1).astype(ts.dtype)
    bin_2 = ((2.4 < ts) & (ts <= 9.1)).sum(axis=-1).astype(ts.dtype)
    bin_3 = ((9.1 < ts) & (ts <= 12.4)).sum(axis=-1).astype(ts.dtype)
    bin_4 = ((11.4 < ts) & (ts <= 15.9)).sum(axis=-1).astype(ts.dtype)
    bin_5 = ((15.9 < ts) & (ts <= 18)).sum(axis=-1).astype(ts.dtype)
    bin_6 = (18 < ts).sum(axis=-1).astype(ts.dtype)

    bin_0 *= 0.
    bin_1 *= 0.5
    bin_2 *= 1.
    bin_3 *= 0.5
    bin_4 *= 0.
    bin_5 *= -0.5
    bin_6 *= -1

    return bin_0 + bin_1 + bin_2 + bin_3 + bin_4 + bin_5 + bin_6


if __name__ == '__main__':

    _ts = np.random.rand(365, 24) * 20 - 5

    _ch = utah_chill(_ts)

    print(_ts[0])
    print(_ch[0])
    print(_ch)
