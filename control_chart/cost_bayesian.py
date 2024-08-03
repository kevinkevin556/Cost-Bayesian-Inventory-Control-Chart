import collections.abc
from functools import partial
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # flake8: noqa
import scipy.integrate as integrate
import scipy.optimize as opt
import tikzplotlib
from scipy.stats import norm


def loss(usc, uhc, inv_level, f):
    shortage_cost = integrate.quad(lambda x: (x - inv_level) * f(x), inv_level, np.inf)[0] * usc
    holding_cost = integrate.quad(lambda x: (inv_level - x) * f(x), 0, inv_level)[0] * uhc
    return shortage_cost + holding_cost


class CostBayesianControlChart:
    def __init__(self, mu0=100, sigma=10, mu1=None, p=6, c=5, r=3.8, T=26, N=12, rule=3, p_=None):
        # Distribution Parameters
        self.mu0 = mu0
        self.rule = rule
        self.mu1 = mu1 if mu1 is not None else mu0 + self.rule * sigma
        self.sigma = sigma
        self.setup_dist()

        # Costs
        self.p, self.c, self.r = p, c, r
        self.N = N
        self.T = T
        self.calculate_costs()

        # Control Chart APIs
        self.p_ = self.L1 / (self.L0 + self.L1) if p_ is None else p_
        self.data = []
        self.history_demand = []
        self.history_purchased = []
        self.history_p = []
        self.history_pa = []
        self.history_pb = []
        self.history_x = []
        self.history_xa = []
        self.history_xb = []

        # Initialize
        self.alpha_beta = [None] * (self.N + 1)
        self.pa, self.pb = [None] * (self.N + 1), [None] * (self.N + 1)
        self.V, self.CEV = [None] * (self.N + 1), [None] * (self.N + 1)
        self.func_points = [None] * (self.N + 1)
        self.restart()

    def calculate_costs(self):
        self.critical_fractile = (self.p - self.c) / (self.p - self.r)
        self.opt_level0 = norm.ppf(self.critical_fractile, loc=self.mu0, scale=self.sigma)
        self.opt_level1 = norm.ppf(self.critical_fractile, loc=self.mu1, scale=self.sigma)
        self.opt_cost0 = loss(self.p - self.c, self.c - self.r, self.opt_level0, self.f0)
        self.nonopt_cost0 = loss(self.p - self.c, self.c - self.r, self.opt_level1, self.f0)
        self.opt_cost1 = loss(self.p - self.c, self.c - self.r, self.opt_level1, self.f1)
        self.nonopt_cost1 = loss(self.p - self.c, self.c - self.r, self.opt_level0, self.f1)
        self.L0 = self.T * (self.nonopt_cost1 - self.opt_cost1)
        self.L1 = self.T * (self.nonopt_cost0 - self.opt_cost0)
        self.C = max(self.nonopt_cost0 - self.opt_cost0, self.nonopt_cost1 - self.opt_cost1)

    def setup_dist(self):
        self.f0 = partial(norm.pdf, loc=self.mu0, scale=self.sigma)
        self.F0 = partial(norm.cdf, loc=self.mu0, scale=self.sigma)
        self.f1 = partial(norm.pdf, loc=self.mu1, scale=self.sigma)
        self.F1 = partial(norm.cdf, loc=self.mu1, scale=self.sigma)

    def purchase(self):
        self.opt_level0 = norm.ppf(self.critical_fractile, loc=self.mu0, scale=self.sigma)
        purchased = self.opt_level0
        self.history_purchased.append(purchased)
        return purchased

    def test(self, x, time="*"):
        """
        When a market demand occurs, check whether it is out of control

        Parameters:
            x (float): The observed market demand at a given time point.
            time (str, optional): A placeholder to show the period of the out-of-control event
                when the time information is given. Default is "*".
        """
        self.history_p.append(self.p_)
        pa = self.pa[0]
        pb = self.pb[0]
        self.pa.pop(0)
        self.pb.pop(0)

        # print(time, self.p_, pa, pb, self.pa, self.pb)
        self.history_demand.append(x)
        self.data.append(x)

        if pa is None or pb is None:
            pa = self.L1 / (self.L0 + self.L1)
            pb = self.L1 / (self.L0 + self.L1)

        self.history_pa.append(pa)
        self.history_pb.append(pb)
        self.history_x.append(x)
        self.history_xa.append(self.x(self.p_, pa))
        self.history_xb.append(self.x(self.p_, pb))

        # N terminates
        if pa is None and pb is None:
            if self.p_ >= self.L1 / (self.L0 + self.L1):
                self.restart()
            else:
                self.mu0 = np.mean(self.data)
                self.mu1 = self.mu0 + self.rule * self.sigma
                print(time, "Out of control", x, f"New (mu, sigma) = {self.mu0, self.sigma}")
                self.data = []
                self.restart()
        # p_ out of control
        elif self.p_ >= pb:
            self.restart()
        elif self.p_ < pa:
            self.mu0 = np.mean(self.data)
            self.mu1 = self.mu0 + self.rule * self.sigma
            print(time, "Out of control", x, f"New (mu, sigma) = {self.mu0, self.sigma}")
            self.data = []
            self.restart()
        else:
            pass
        # Update posterior
        self.p_ = self.update_p(self.p_, x)

    def restart(self):
        """Process out of control. Regenerate control limits."""
        self.setup_dist()
        self.calculate_costs()
        self.backward()

    def update_p(self, p, x):
        """Compute posterior based on Bayes theorem."""
        p_ = p * self.f0(x) / (p * self.f0(x) + (1 - p) * self.f1(x))
        if p_ > 0.95:
            p_ = 0.95
        if p < 0.05:
            p_ = 0.05
        return p_

    def evn(self, p):
        """Exact E[V_N(p)]"""
        mu0, mu1, sigma = self.mu0, self.mu1, self.sigma
        L0, L1, C = self.L0, self.L1, self.C
        F0, F1 = self.F0, self.F1
        xp = 0.5 * (mu0 + mu1) + sigma**2 / (mu1 - mu0) * np.log(p * L0 / ((1 - p) * L1))
        riskf = (1 - p) * L0 * F1(xp) + p * L1 * (1 - F0(xp))
        return riskf

    def find_crit(self, cev):
        """Solve two critical point of E[V(p)] when k=N-1"""
        eps = 1e-11
        L0, L1 = self.L0, self.L1
        # Solve p_a
        erfa = lambda p: cev(p) - p * self.L1
        sol = opt.root_scalar(erfa, bracket=[eps, 1 - eps], method="brentq")
        pa = sol.root
        # Solve p_b
        erfb = lambda p: cev(p) - (1 - p) * L0
        sol = opt.root_scalar(erfb, bracket=[eps, 1 - eps], method="brentq")
        pb = sol.root
        # print(f"(pa, pb) = ({pa}, {pb})")
        return pa, pb

    def x(self, p, p_):
        mu0, mu1, sigma = self.mu0, self.mu1, self.sigma
        if np.allclose(p, 1) or np.allclose(p_, 0):
            return np.inf
        if np.allclose(p, 0) or np.allclose(p_, 1):
            return -np.inf
        return (mu0 + mu1) / 2 + sigma**2 / (mu1 - mu0) * np.log(p * (1 - p_) / (p_ * (1 - p)))

    def beta(self, p, pa_, pb_, alpha_, beta_):
        L0, L1, C = self.L0, self.L1, self.C
        F0, F1 = self.F0, self.F1
        xa = self.x(p, pa_)
        xb = self.x(p, pb_)
        return (
            -L0 * F1(xb)
            + (C + alpha_ + beta_) * (F0(xa) - F0(xb))
            - (C + alpha_) * (F1(xa) - F1(xb))
            + L1 * (1 - F0(xa))
        )

    def alpha(self, p, pa_, pb_, alpha_, beta_):
        L0, L1, C = self.L0, self.L1, self.C
        F0, F1 = self.F0, self.F1
        xa = self.x(p, pa_)
        xb = self.x(p, pb_)
        return L0 * F1(xb) + (C + alpha_) * (F1(xa) - F1(xb))

    def backward(self):
        # Results
        self.alpha_beta = [None] * (self.N + 1)
        self.pa, self.pb = [None] * (self.N + 1), [None] * (self.N + 1)
        self.V, self.CEV = [None] * (self.N + 1), [None] * (self.N + 1)
        self.func_points = [None] * (self.N + 1)
        # Parameters
        L0, L1, C = self.L0, self.L1, self.C
        N = self.N

        self.V[N] = lambda p: min(self.L0 * (1 - p), self.L1 * p)
        self.func_points[N] = samples(self.V[N], L0, L1)  # for plotting functions

        for n in range(N, 0, -1):
            k = n - 1
            # Solve or approximate V
            if k == N - 1:
                CEV_ = lambda p: C + self.evn(p)
                V = lambda p: min((1 - p) * L0, p * L1, CEV_(p))
                self.V[k], self.CEV[k + 1] = V, CEV_
            else:
                alpha_ = self.alpha_beta[k + 1][0]
                beta_ = self.alpha_beta[k + 1][1]
                pa_ = self.pa[k + 1]  # (C + alpha_) / (L1 - beta_)
                pb_ = self.pb[k + 1]  # (C + alpha_ - L0) / (-L0 - beta_)
                CEV_ = lambda p: C + self.alpha(p, pa_, pb_, alpha_, beta_) + p * self.beta(p, pa_, pb_, alpha_, beta_)
                V = lambda p: min((1 - p) * L0, p * L1, CEV_(p))
                self.V[k], self.CEV[k + 1] = V, CEV_

            # for plotting functions
            self.func_points[k] = samples(CEV_, L0, L1)

            # Linearize  EV
            pa, pb = self.find_crit(CEV_)
            assert pa <= pb, f"Invalid limits at time={k}"
            v_pa, v_pb = CEV_(pa), CEV_(pb)
            beta = (v_pb - v_pa) / (pb - pa)
            alpha = -beta * pa + v_pa - C
            self.alpha_beta[k] = (alpha, beta)
            self.pa[k], self.pb[k] = pa, pb

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


# utils


def plot(functions=None, samples=None, labels=None, pa=None, pb=None, tikz=None):
    x = np.linspace(start=0, stop=1, num=100)
    ls = ["-", "--", "-.", ":"]
    # Initialize a plot
    # plt.style.use("science")
    plt.style.use(["science", "no-latex"])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Plot functions
    if functions is not None:
        if callable(functions):
            y = [functions(i) for i in x]
            ax.plot(x, y, label="f(x)")
        else:
            for i, f in enumerate(functions):
                if callable(f):
                    y = [f(j) for j in x]
                    label = labels[i] if labels is not None else f"f{i+1}(x)"
                    # ax.plot(x, y, label=label, linestyle=ls[i % 4])
                    ax.plot(x, y, label=label)
    if samples is not None:
        for i, sample in enumerate(samples):
            if sample is not None:
                label = labels[i] if labels is not None else f"f{i+1}(x)"
                ax.plot(sample[0], sample[1], label=label)
                # ax.plot(sample[0], sample[1], label=label, linestyle=ls[i % 4])
    # Draw points
    if pa is not None:
        for i, x_pt in enumerate(pa):
            if x_pt is not None:
                ax.vlines(x_pt, ymin=0, ymax=-0.5, color="red")  # Short vertical line
                ax.text(x_pt, -0.8, f"{i}", fontsize=9, ha="center")  # Label with index
    if pb is not None:
        for i, x_pt in enumerate(pb):
            if x_pt is not None:
                ax.vlines(x_pt, ymin=0, ymax=-0.5, color="red")  # Short vertical line
                ax.text(x_pt, -0.8, f"{i}", fontsize=9, ha="center")  # Label with index
    # Output
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.title("Plot of the function f(x)")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    if tikz:
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(tikz)
    else:
        plt.show()


def samples(f, L0, L1):
    x1 = np.linspace(start=0, stop=L1 / (L0 + L1), num=51)
    x2 = np.linspace(start=L1 / (L0 + L1), stop=1, num=50)
    x = np.append(x1[0:-1], x2)
    y = [f(i) for i in x]
    return x, y


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
