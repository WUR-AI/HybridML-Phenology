
import numpy as np


def chill_days(ts: np.ndarray,
               t_C: float,
               ):
    """

    Chilling and forcing model to predict bud-burst of crop and forest species

    https://www.sciencedirect.com/science/article/pii/S0168192304000632


    :param ts:
    :param t_C:
    :return:
    """

    # Set variables following the notation used in the paper
    t_x = ts.max(axis=-1)  # Daily max temperature
    t_n = ts.min(axis=-1)  # Daily min temperature
    t_M = ts.mean(axis=-1)  # Daily mean temperature

    result = np.zeros(shape=ts.shape[:-1], dtype=ts.dtype)

    result += ((0 <= t_n) & (t_n <= t_C) & (t_C < t_x)) * -((t_M - t_n) - (((t_x - t_C) ** 2) / (2 * (t_x - t_n))))
    result += ((0 <= t_n) & (t_n <= t_x) & (t_x <= t_C)) * -(t_M - t_n)
    result += ((t_n < 0) & (0 < t_x) & (t_x <= t_C)) * -((t_x ** 2) / (2 * (t_x - t_n)))
    result += ((t_n < 0) & (0 < t_C) & (t_C < t_x)) * -(((t_x ** 2) / (2 * (t_x - t_n))) + (((t_x - t_C) ** 2) / (2 * (t_x - t_n))))

    return result


def anti_chill_days(ts: np.ndarray,
                    t_C: float,
                    ):

    # Set variables following the notation used in the paper
    t_x = ts.max(axis=-1)
    t_n = ts.min(axis=-1)
    t_M = ts.mean(axis=-1)

    result = np.zeros(shape=ts.shape[:-1], dtype=ts.dtype)

    result += ((0 <= t_C) & (t_C <= t_n) & (t_n <= t_x)) * (t_M - t_C)
    result += ((0 <= t_n) & (t_n <= t_C) & (t_C < t_x)) * (((t_x - t_C) ** 2) / (2 * (t_x - t_n)))
    result += ((t_n < 0) & (0 < t_C) & (t_C < t_x)) * (((t_x - t_C) ** 2) / (2 * (t_x - t_n)))

    return result


def chill_days_model(ts: np.ndarray,
                     t_C: float,
                     C_R: float,
                     C_aR: float = 0,
                     ) -> dict:
    """

    :param ts:
    :param t_C: Base temperature
    :param C_R: Chill requirement threshold
    :param C_aR: Anti-chill requirement threshold (0 in original paper)
    :return:
    """
    assert len(ts.shape) == 2
    if len(ts) == 0:
        return {
            'bloom': False,
        }

    # Temperatures tensor ts is assumed to have shape (t, n), where
    #   t is the number of days
    #   n is the number of temperature measurements per day

    # Compute the chill days that are acquired for each day
    cds = chill_days(ts, t_C)

    # Compute the cumulative chill days over time
    ccds = cds.cumsum(axis=-1)

    # Obtain the first index where the cumulative chill days exceeds some threshold C_R
    ixs, = np.where(ccds <= C_R)
    if len(ixs) == 0:  # If the threshold is never reached, return -1
        return {
            'chill_days': cds,
            'bloom': False,
        }

    # Get the first index where the c_R threshold is exceeded
    # This is where, according to the model, the chill requirement has been met
    ix_cr = ixs.min(axis=-1)

    # Select the remaining temperature time series over which the anti-chill days should be computed
    # That is, all days after meeting the chill requirement
    ts = ts[ix_cr:]
    # Chill days are accumulated until the requirement has been met
    # Omit the other dates
    cds = cds[:ix_cr]

    # Compute the anti-chill days that are acquired for each day
    ads = anti_chill_days(ts, t_C)

    # Compute the cumulative anti-chill days over time
    cads = ads.cumsum(axis=-1)

    # Get the first index where the cumulative anti-chill days, summed with the cumulative chill days when the chill
    # requirement was met, exceeds some threshold C_aR
    ixs, = np.where(cads >= C_aR)
    if len(ixs) == 0:  # If the threshold is never reached, return -1
        return {
            'ix_cr': ix_cr,
            'chill_days': cds,
            'anti-chill_days': ads,
            'bloom': False,
        }

    # Get the first index at which the threshold was reached
    ix_acr = ixs.min(axis=-1)

    # As additional info, add the index where the chill days and anti-chill days sum to 0
    # Originally, the model is fit to let this point indicate bud burst
    ix_bud = np.where((C_R + cads) >= 0)[0].min(axis=-1, initial=270)  # TODO -- fix

    # The bloom index is when both chill and anti-chill requirements have been met
    ix_bloom = ix_acr + ix_cr

    return {
        'ix_bloom': ix_bloom,
        'ix_cr': ix_cr,
        'ix_bud': ix_bud,
        'chill_days': cds,
        'anti-chill_days': ads,
        'bloom': True,
    }


if __name__ == '__main__':

    # _ts = np.ones(shape=(24,))
    # _ts = np.random.rand(365, 24) * 10 - 5
    _ts = np.arange(-10, 15, dtype=float).reshape((-1, 1))
    _ts = np.concatenate([_ts, _ts + 4], axis=-1)
    _end_doy = 150
    _t_C = 9
    _C_R = -100
    _C_aR = 100

    # print(_ts[0])
    # # print(f_chill_units(_ts, _t_C))
    # print(anti_chill_days(_ts[3], _t_C))
    #
    # print(chill_days(_ts, _t_C))
    # print(anti_chill_days(_ts, _t_C))
    #

    # print(_ts.mean())

    _r = chill_days_model(_ts, _t_C, _C_R, _C_aR)

    print(_r['bloom'])
    print(_r['ix_bloom'] if _r['bloom'] else -1)
    # print(_r['ix_cr'])

    for _l in list(zip(_ts.min(axis=-1), _ts.mean(axis=-1), _ts.max(axis=-1), _r['chill_days'])):
        print(_l)
