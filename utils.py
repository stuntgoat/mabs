import math

import numpy as np
from statsmodels.stats.proportion import std_prop


def standard_error(r, o):
    """
    Returns the standard error of the ratio of `r / o`.

    """
    if o == 0:
        return 0, 0

    o = float(o)
    phat = r / o
    return phat, np.sqrt((phat * (1 - phat)) / o)


def prop_diff_confint(r1, offer1, r2, offer2):
    """
    Args:
    - r1: redemptions of trial 1
    - offer1: total number of trials for trial 1
    - r2: redemptions of trial 2
    - offer2: total number of trials for trial 2

    Returns the (<difference of proportions>, <lower>, <upper>) of the difference
    between the two proportions with 95% confidence.

    Reference: http://www.stat.wmich.edu/s216/book/node85.html
    >>> prop_diff_conint(74*2, 200, 66*2, 200)
    (0.07999999999999996, -0.01130169768410666, 0.17130169768410658)

    """
    p1 = r1 / float(offer1)
    p2 = r2 / float(offer2)

    se1 = std_prop(p1, offer1)
    se2 = std_prop(p2, offer2)

    # Confidence interval
    diff_se = np.sqrt(se1 ** 2 + se2 ** 2)
    diff = abs(p1 - p2)

    # For 95% confidence interval multiply the standard error of the difference
    # by 2.
    return (diff, (diff - 2 * diff_se), (diff + 2 * diff_se))


def confint95(r, o):
    """
    Returns the 95% uppper and lower confidence interval for the ratio
    `r / 0`.

    """
    conv, se = standard_error(r, o)
    # Return the hi/low interval
    return conv + se * 2, max(0, conv - se * 2)


def soft_max_temp(count, numerator=1):
    """
    Like softmax temperature but instead of using the total number of offers as a value
    in the equation, we use the least offered Asset count.

    """
    t = count + 1
    return numerator / math.log(t + .1e-7)
