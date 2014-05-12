"""
Examples for running simulations.

"""

import pandas as pd

from assets import (
    Asset,
    AssetSet,
    Selector
)


## Example functions for returning a repeatable AssetSet instance for tests.
def get_set():
    """
    Returns an AssetSet that introduces a set some assets
    that perform better than those initially available.

    """

    original = [Asset('A', .03), Asset('B', .045), Asset('C', .067)]

    return AssetSet({
        0: original,
        500: [Asset('D', .15)],
        (600, 800): [Asset('E', .02)],
        700: [Asset('F', .24)]
    })


def get_set_2():
    """
    Returns an AssetSet that introduces a set some assets
    that do not perform as well as ones introduced earlier.

    """
    original = [Asset('A', .3), Asset('B', .045), Asset('C', .067)]

    return AssetSet({
        0: original,
        500: [Asset('D', .15)],
        (600, 800): [Asset('E', .02)],
        700: [Asset('F', .0024)]
    })


def get_set3():
    """
    Returns an AssetSet that continues to introduce assets that perform
    better than the previous.

    """
    original = [Asset('A', .02), Asset('B', .03), Asset('C', .04)]

    return AssetSet({
        0: original,
        500: [Asset('D', .08)],
        (600, 800): [Asset('E', .09)],
        700: [Asset('F', .11)]
    })


## Example functions for simulating the Selector algorithms.
def softmax(aset_func, cycles, pulls, minimum):
    """
    Runs a series softmax experiments.

    """
    for i in xrange(cycles):
        aset = aset_func()
        s = Selector(Selector.USE_SOFT_MAX, minimum)
        for _ in xrange(pulls):
            s.make_choice(aset)
        yield s.as_row(aset.assets)

    aset.finish()


def softmax2(aset_func, cycles, pulls, minimum):
    """
    Runs a series softmax2 experiments.

    """
    for i in xrange(cycles):
        aset = aset_func()
        s = Selector(Selector.USE_SOFT_MAX_2, minimum)
        for _ in xrange(pulls):
            s.make_choice(aset)

        yield s.as_row(aset.assets)
    aset.finish()


def rho_vals(aset_func, _, pulls, minimum):
    """
    Runs a series experiments in which the rho value is varied.

    """
    for i in xrange(100):
        # Set the probability at which we choose a random
        # value or choose to minimize the variance
        s = Selector(i * .01, minimum)
        aset = aset_func()
        for c_idx in xrange(pulls):
            s.make_choice(aset)

        yield s.as_row(aset.assets)


def rho_func(aset_func, cycles, pulls, minimum):
    def rf(choices):
        """
        Reduces the probability at which the asset with the largest width
        is chosen given the width.
        """
        width = max([_.width for _ in choices])
        if width > .05:
            return .6
        if width > .035:
            return .20
        if width > .02:
            return .06
        return 0

    for i in xrange(cycles):
        s = Selector(Selector.USE_RHO_FUNC, minimum)
        s.rho_func = rf
        aset = aset_func
        for _ in xrange(pulls):
            s.make_choice(aset)

        yield s.as_row(aset.assets)


def only_best(aset_func, cycles, pulls, minimum):
    for i in xrange(cycles):
        s = Selector(Selector.ONLY_BEST, minimum)
        aset = aset_func()
        for _ in xrange(pulls):
            s.make_choice(aset)

        yield s.as_row(aset.assets)


## Example function for running a series of simulations with
## a series of AssetSet instances, returning a Pandas DataFrame
## of results.
def runtests(*aset_funcs, **kwargs):
    """
    Example for running a series of simulations.

    Runs a series of tests for the AssetSet instances returned
    from `aset_funcs`.

    Args:
    - aset_funcs a list of callables that will

    Returns a Pandas DataFrame object with the results.

    """
    cycles = kwargs.pop('cycles', 100)
    pulls = kwargs.pop('pulls', 1500)
    minimum = kwargs.pop('minimum', 0)

    RESULTS = {}

    key = 0
    for aset_func in aset_funcs:
        for result in softmax(aset_func, cycles, pulls, minimum):
            key += 1
            RESULTS[key] = result

        for result in softmax2(aset_func, cycles, pulls, minimum):
            key += 1
            RESULTS[key] = result

    df = pd.DataFrame(list(RESULTS.values()), columns=Selector.COL_NAMES)
    return df.sort('avg_conv', ascending=False)
