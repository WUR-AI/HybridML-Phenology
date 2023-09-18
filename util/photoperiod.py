

"""
    Code used to compute photoperiod for a given location/date

    Obtained from:

    https://github.com/soilwater/pynotes-agriscience
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt


# Define function
def photoperiod(phi, doy, verbose=False):

    """
        Paper referenced:
        Keisling, T.C., 1982. Calculation of the Length of Day 1. Agronomy Journal, 74(4), pp.758-759.
    """
    phi = np.radians(phi)  # Convert to radians
    light_intensity = 2.206 * 10 ** -3

    C = np.sin(np.radians(23.44))  # sin of the obliquity of 23.44 degrees.
    B = -4.76 - 1.03 * np.log(
        light_intensity)  # Eq. [5]. Angle of the sun below the horizon. Civil twilight is -4.76 degrees.

    # Calculations
    alpha = np.radians(90 + B)  # Eq. [6]. Value at sunrise and sunset.
    M = 0.9856 * doy - 3.251  # Eq. [4].
    lmd = M + 1.916 * np.sin(np.radians(M)) + 0.020 * np.sin(np.radians(2 * M)) + 282.565  # Eq. [3]. Lambda
    delta = np.arcsin(C * np.sin(np.radians(lmd)))  # Eq. [2].

    # Defining sec(x) = 1/cos(x)
    P = 2 / 15 * np.degrees(
        np.arccos(np.cos(alpha) * (1 / np.cos(phi)) * (1 / np.cos(delta)) - np.tan(phi) * np.tan(delta)))  # Eq. [1].

    # Print results in order for each computation to match example in paper
    if verbose:
        print('Input latitude =', np.degrees(phi))
        print('[Eq 5] B =', B)
        print('[Eq 6] alpha =', np.degrees(alpha))
        print('[Eq 4] M =', M[0])
        print('[Eq 3] Lambda =', lmd[0])
        print('[Eq 2] delta=', np.degrees(delta[0]))
        print('[Eq 1] Daylength =', P[0])

    return P
