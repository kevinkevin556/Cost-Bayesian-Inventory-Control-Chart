import collections.abc
from functools import partial
from pprint import pformat

import numpy as np
from scipy.stats import norm, t


class BayesianControlChart:
    def __init__(self, mu, p, c, r, llambda=None, alpha=None, beta=None, gamma=0.99, sigma=10, delta=2, n=3):
        """
        Initialize the BayesianControlChart with given parameters.

        Parameters:
            mu (float): The initial mean of the demand distribution.
            p (float): The selling price per unit.
            c (float): The cost per unit.
            r (float): The salvage value per unit.
            llambda (float, optional): The prior precision (1/variance) of the demand distribution.
            alpha (float, optional): The shape parameter of the Gamma distribution (prior on precision).
            beta (float, optional): The rate parameter of the Gamma distribution (prior on precision).
            gamma (float, optional): The confidence level for the control limits. Default is 0.99.
            sigma (float, optional): The standard deviation of the demand distribution for benchmarking. Default is 10.
        """
        # Distribution Parameters
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.llambda = llambda
        self.gamma = gamma

        # Costs
        self.p = p  # Selling price per unit
        self.c = c  # Cost per unit
        self.r = r  # Salvage value per unit
        self.sigma = sigma

        # Note: The sigma can be determined by estimating the posterior predictive distibution of demand.
        # We specify the value independent from estimation for the sake of benchmarking.

        # Control Chart APIs
        self.data = []
        self.history_demand = []
        self.history_purchased = []

        # Initialize
        self.rule = t.ppf(q=1 / 2 + self.gamma / 2, df=2 * self.alpha)
        self.delta = delta
        self.n = n

    def test(self, x, time=None):
        """
        When a market demand occurs, check whether it is out of control

        Parameters:
            x (float): The observed market demand at a given time point.
            time (str, optional): A placeholder to show the period of the out-of-control event
                when the time information is given. Default is None.
        """
        v_scale = (self.llambda * self.alpha / self.beta) ** 0.5
        lcl = self.mu - self.rule / v_scale
        ucl = self.mu + self.rule / v_scale

        self.history_demand.append(x)
        self.data.append(x)

        d = np.array(self.data)
        mad = np.mean(np.abs(d - self.mu))
        ts = np.abs(np.sum(d - self.mu) / mad)
        # ts = self.delta + 1

        if lcl <= x <= ucl or ts <= self.delta:
            # In control
            pass
        else:
            self.update_p(self.data)
            self.restart()
            if time is not None:
                info = f"New (mu, sigma, mad, ts) = {self.mu:.3f}, {self.sigma}, {mad:.3f}, {ts:.3f}"
                cl_info = f"(lcl, ucl) = ({lcl:.3f}, {ucl:.3f})"
                print(time, "Out of control", x, info, lcl, ucl)
            return time

    def purchase(self):
        """Purchase based on estimated distribution"""
        critical_fractile = (self.p - self.c) / (self.p - self.r)
        purchased = norm.ppf(critical_fractile, loc=self.mu, scale=self.sigma)
        self.history_purchased.append(purchased)
        return purchased

    def update_p(self, data):
        """Compute posterior based on Bayes theorem."""
        n = self.n if self.n else len(data)
        x = np.array(data)[-n:]

        llambda_ = self.llambda + n
        mu_ = (self.llambda * self.mu + n * x.mean()) / llambda_
        alpha_ = self.alpha + n / 2
        beta_ = (
            self.beta
            + (1 / 2) * np.sum((x - x.mean()) ** 2)
            + n * self.llambda * (x.mean() - self.mu) ** 2 / (2 * llambda_)
        )
        self.mu, self.llambda, self.alpha, self.beta = mu_, llambda_, alpha_, beta_

    def restart(self):
        """Process out of control. Regenerate control limits."""
        self.rule = t.ppf(q=1 / 2 + self.gamma / 2, df=2 * self.alpha)
        self.data = []

    def __repr__(self):
        properties = {}
        for attr_name in dir(self):
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            attr_value = getattr(self, attr_name)
            if not callable(attr_value) and not isinstance(
                attr_value, (collections.abc.Sequence, collections.abc.Mapping)
            ):
                properties[attr_name] = attr_value
        return pformat(properties)
