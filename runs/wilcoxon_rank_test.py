
from scipy.stats import wilcoxon

if __name__ == '__main__':

    # # Japan
    # maes_pb = [  # Utah
    #     3.26097,
    #     3.04195,
    #     3.29263,
    #     2.99658,
    #     2.74707,
    #     3.18721,
    #     2.95108,
    #     2.88319,
    #     3.22158,
    #     3.10872,
    # ]
    # maes_nn = [
    #     2.25534,
    #     2.06689,
    #     2.28105,
    #     2.10137,
    #     2.3829,
    #     2.22326,
    #     2.32423,
    #     2.33406,
    #     2.23077,
    #     2.37599,
    # ]

    # # Switzerland
    # maes_pb = [  # Utah
    #     7.64778,
    #     5.61154,
    #     7.31046,
    #     8.58314,
    #     7.29826,
    #     6.36003,
    #     6.93425,
    #     8.45171,
    #     7.40325,
    #     7.58801,
    # ]
    # maes_nn = [
    #     5.82006,
    #     4.73308,
    #     5.125,
    #     5.00937,
    #     4.90689,
    #     4.6633,
    #     5.48318,
    #     5.71635,
    #     5.113,
    #     5.30728,
    # ]

    # South Korea
    maes_pb = [  # Utah
        6.2948,
        5.13615,
        6.4083,
        5.6087,
        5.0122,
        5.38824,
        7.29577,
        7.47549,
        5.86719,
        6.76245,
    ]
    maes_nn = [
        4.46243,
        4.61502,
        5.11419,
        4.50725,
        4.64024,
        4.93529,
        5.12676,
        4.7402,
        4.63281,
        4.88506,
    ]

    print(maes_pb)
    print(maes_nn)

    print(sum(maes_pb))
    print(sum(maes_nn))

    # print('Two-sided:')
    # result = wilcoxon(ses_pb, ses_nn)
    # print(result)
    # print(result.statistic)
    # print(result.pvalue)

    print('One-sided:')
    result = wilcoxon(maes_pb, maes_nn, alternative='greater')
    # result = wilcoxon(ses_pb, ses_nn, alternative='less')

    print(result)
    print(result.statistic)
    print(result.pvalue)
