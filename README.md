`mabs`
====

Multi-Armed Bandit Simulations


Example simulations are found in `tests.py`, for instance:

    In [1]: from tests import runtests, get_set, get_set_2, get_set3

    In [2]: df = runtests(get_set, get_set_2, get_set3, cycles=500, pulls=2000)
    total: 2000
    total: 2000
    total: 2000
    total: 2000
    total: 2000
    total: 2000

    In [3]: df.loc[df.name == 'Softmax'].avg_conv.describe()
    Out[3]:
    count    1500.000000
    mean        0.174410
    std         0.095378
    min         0.046025
    25%         0.072817
    50%         0.162162
    75%         0.290500
    max         0.329000
    Name: avg_conv, dtype: float64

    In [4]: df.loc[df.name == 'Softmax 2'].avg_conv.describe()
    Out[4]:
    count    1500.000000
    mean        0.157598
    std         0.077826
    min         0.047594
    25%         0.068118
    50%         0.158361
    75%         0.242726
    max         0.287513
    Name: avg_conv, dtype: float64
