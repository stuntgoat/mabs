import math
from random import random, choice


from utils import confint95


def avg_conv(assets):
    """
    Accepts a collection of assets and calculates the average conversion rate.

    """
    return sum(map(lambda x: x.redemptions, assets)) / float(sum(map(lambda x: x.offers, assets)))


class Asset(object):
    """
    Asset(name, likelihood)
    'name': the name, str
    'likelihood': probability of being selected, float

    """

    def __init__(self, name, likelihood):
        self.name = name

        self._conversion = 0

        # Probability of being selected per offer.
        self.likelihood = likelihood

        self.offers = 0.0
        self.redemptions = 0.0

    @property
    def conversion(self):
        if self.offers == 0:
            return 0
        return self._conversion

    @property
    def confint95(self):
        return confint95(self.redemptions, self.offers)

    def offer(self):
        if random() <= self.likelihood:
            self.redemptions += 1
        self.offers += 1
        self._conversion = self.redemptions / float(self.offers)

    @property
    def width(self):
        hi, low = self.confint95
        return hi - low

    def __repr__(self):
        return "Asset:%s" % self.name


class Selector(object):
    ## Constants used to determine the logic of choosing
    ## an asset in the `make_choice` method of a Selector instance.
    # A modified Softmax algorithm that makes use of
    # the least chosen asset count for generating the annealing temperature.
    USE_SOFT_MAX_2 = -40

    # Standard Softmax from:
    # "Bandit Algorithms for Website Optimization"
    # http://shop.oreilly.com/product/0636920027393.do
    USE_SOFT_MAX = -30

    # Use a function that generates the probability when to choose the
    # asset such that we minimize the variance.
    # NOTE: self.rhof must be set to a function that accepts an AssetSet
    USE_RHO_FUNC = -20

    # Only select the best performing asset.
    ONLY_BEST = -10

    # Select an asset at random.
    RANDOM = -1

    def __init__(self, rho, start):
        """
        Args:
        - rho: By default, rho is either a value between 0 and 1, and it
               is used to determine the probability of the rate
               at which the Selector instance chooses the best performing Asset
               or chooses to minimize the widest confidence interval. If it is a
               value less than zero, it is a class constant which determines
               an alternate method of selecting from an AssetSet instance.
        - start: a minimum value that each asset must reach before
                 attempting to use a method for optimizing for the
                 best asset.

        """
        self.rho = rho

        # Number of offers to start rho selection.
        self.start = start

        self.rho_selected = 0
        self.best_selected = 0

        # Used to hold an optional function that determines rho.
        # This function expects an AssetSet instance as an only argument.
        self.rhof = lambda x: self.rho

    @property
    def name(self):
        "Return a readable name of the selection method."
        if self.rho == self.USE_RHO_FUNC:
            return "Rho Function"
        if self.rho == self.RANDOM:
            return "Random"
        if self.rho == self.ONLY_BEST:
            return "Best Converting"
        if self.rho == self.USE_SOFT_MAX:
            return "Softmax"
        if self.rho == self.USE_SOFT_MAX_2:
            return "Softmax 2"
        if 0 <= self.rho <= 1:
            return "Rho %s" % self.rho

        return "Rho %s" % self.rho

    def rho_func(self, assets):
        return self.rhof(assets)

    def select_best(self, choices):
        self.best_selected += 1
        choices = sorted(choices, key=lambda x: x.conversion)
        choices[-1].offer()

    def make_choice(self, asset_set):
        ## TODO: - simplifiy this logic

        # Choose until we have hit the start threshold for each choice.
        choices = None
        assets_incremented = False
        if self.start:
            choices = sorted(asset_set.assets_with_increment, key=lambda x: x.offers)
            assets_incremented = True
            if choices[0].offers < self.start:
                # Choose choice with least offers.
                choices[0].offer()
                return

        # If we incremented when we got choices from having a start value, we do not
        # want to increment again when we call the softmax choices, otherwise, we
        # increment the assets by passing in `not assets_incremented`, which will be True.
        if self.rho == self.USE_SOFT_MAX:
            asset_set.soft_max_choice(not assets_incremented).offer()
            return
        elif self.rho == self.USE_SOFT_MAX_2:
            asset_set.soft_max_2_choice(not assets_incremented).offer()
            return

        # Potentially assign rho dynamically.
        elif self.rho == self.USE_RHO_FUNC:
            rho = self.rho_func(asset_set)
        else:
            rho = self.rho

        # Increment the assets since we have not acquired them or incremented the AssetSet
        # instance.
        if not choices:
            choices = sorted(asset_set.assets_with_increment, key=lambda x: x.offers)

        # Check for constants.
        if self.rho == self.RANDOM:
            choice(choices).offer()
            return
        elif self.rho == self.ONLY_BEST:
            self.select_best(choices)
            return

        # Choose value to decrease worst confidence interval.
        if random() <= rho:
            self.rho_selected += 1
            # Sort by 'width', which is the difference of upper and lower
            # confidence interval, and select the value with the largest width.
            choices = sorted(choices, key=lambda x: x.width, reverse=True)
            choices[0].offer()
            return

        self.select_best(choices)

    def as_row(self, assets):
        """
        Returns a tuple of column names and the column values.
        """
        vals = [self.name, avg_conv(assets), self.rho_selected, self.best_selected]
        return vals
    COL_NAMES = ['name', 'avg_conv', 'minimized_confint', 'called_select_best']


class AssetSet(object):
    """
    Introduces a set of assets at the corresponding counts.

    Motivation: the ability to add and remove Asset instances that perform
    at various conversion rates.

    Contains logic for calculating composite values from the assets.

    """
    def __init__(self, amap):
        """
        `amap` is a dict of intervals to lists of assets; at each
        key(total count) the assets are introduced. If given a tuple,
        index 0 and 1 signifies the insert and removal count, respectively.

        """
        # TODO: - cache the assets returned based on the counts property. This will
        #         prevent calculating which assets are needed on consecutive calls to
        #         the 'assets' property.

        self.asset_map = amap
        # The total number of asset offersin the current set at a given state
        # of the AssetSet's instance.
        self._counts = 0

        # The number of iterations for the AssetSet instance. This is different
        # than the 'counts' property in that it is the total number of all offers
        # that the AssetSet instance has seen.
        self.total = 0

    @property
    def assets(self):
        asset_set = set()
        for insert_val, asset_list in self.asset_map.iteritems():
            if isinstance(insert_val, (tuple, list)):
                mn, mx = insert_val
                if mn <= self.total and self.total < mx:
                    for a in asset_list:
                        asset_set.add(a)
                continue

            if self.total >= insert_val:
                for a in asset_list:
                    asset_set.add(a)
        self._counts = sum([_.offers for _ in asset_set])
        return asset_set

    def assets_with_increment(self):
        "Returns the set of current assets and increments the self.total offers."
        a = self.assets
        self.total += 1
        return a

    @property
    def counts(self):
        "The sum of all offers for current Assets"
        return self._counts

    @property
    def soft_max_temp(self):
        "Standard Softmax temperature"
        t = self.counts + 1
        return .25 / math.log(t + .1e-7)

    @property
    def soft_max_temp_min(self):
        """
        Like softmax temperature but instead of using the total number of offers as a value
        in the equation, we use the least offered Asset count.

        """
        t = min([_.offers for _ in self.assets]) + 1
        return .25 / math.log(t + .1e-7)

    def _assets_values(self, increment=False):
        cur_assets = tuple(_ for _ in self.assets)
        if increment:
            self.total += 1
        vals = tuple(_.conversion for _ in cur_assets)
        return cur_assets, vals

    def _probs_from_conversions_temps(self, conversions, temp):
        denominator = sum(math.exp(_ / temp) for _ in conversions)
        return [math.exp(_ / temp) / denominator for _ in conversions]

    def _choose_assets_with_temp(self, cur_assets, vals, temp):
        probabilities = self._probs_from_conversions_temps(vals, temp)
        rand = random()
        cum_prob = 0.0
        for i in xrange(len(probabilities)):
            cum_prob += probabilities[i]
            if cum_prob > rand:
                return cur_assets[i]
        return cur_assets[-1]

    def soft_max_2_choice(self, increment=False):
        """
        Uses a modified version of Softmax to select the Asset.
        Selects an asset but does not call the method to pull the arm.

        NOTE: increments self.total for this instance.
        """
        cur_assets, vals = self._assets_values(increment)
        sm_temp = self.soft_max_temp_min
        return self._choose_assets_with_temp(cur_assets, vals, sm_temp)

    def soft_max_choice(self, increment=False):
        """
        Uses standard Softmax to select the Asset.
        Selects an asset but does not call the method to pull the arm.

        NOTE: increments self.total for this instance.

        """
        cur_assets, vals = self._assets_values(increment)
        sm_temp = self.soft_max_temp
        return self._choose_assets_with_temp(cur_assets, vals, sm_temp)

    def finish(self):
        print 'total: %s' % self.total
