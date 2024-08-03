import numpy as np
import pandas as pd
from scipy.stats import norm


class ShewhartControlChart:
    def __init__(self, mu: float, sigma: float, p: float, c: float, r: float):
        """
        Initialize the ShewhartControlChart with given parameters.

        Parameters:
            mu (float): The initial mean of the demand distribution.
            sigma (float): The initial standard deviation of the demand distribution.
            p (float): The selling price per unit.
            c (float): The cost per unit.
            r (float): The salvage value per unit.
        """
        # Distribution Parameters
        self.mu = mu
        self.sigma = sigma

        # Costs
        self.p = p  # Selling price per unit
        self.c = c  # Cost per unit
        self.r = r  # Salvage value per unit

        # Control Chart APIs
        self.data = []  # Data for the control chart
        self.history_demand = []  # History of market demand
        self.history_purchased = []  # History of purchased quantities

    def test(self, x, time=None):
        """
        When a market demand occurs, check whether it is out of control

        Parameters:
            x (float): The observed market demand at a given time point.
            time (str, optional): A placeholder to show the period of the out-of-control event
                when the time information is given. Default is "*".
        """
        lcl = self.mu - 3 * self.sigma
        ucl = self.mu + 3 * self.sigma

        self.history_demand.append(x)
        self.data.append(x)
        if lcl <= x <= ucl:
            # If in control, do nothing
            pass
        else:
            # If out of control, update mu
            self.mu = np.mean(self.data)
            self.data = []  # Clear the data as control limits are reset
            if time is not None:
                print(time, "Out of control", x, f"New (mu, sigma) = {self.mu, self.sigma}")
            return time

    def purchase(self):
        """Purchase based on estimated distribution"""
        critical_fractile = (self.p - self.c) / (self.p - self.r)
        purchased = norm.ppf(critical_fractile, loc=self.mu, scale=self.sigma)
        self.history_purchased.append(purchased)
        return purchased
