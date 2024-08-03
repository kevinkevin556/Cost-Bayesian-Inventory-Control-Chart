import numpy as np
from scipy.stats import norm


def stable(mu: float, sigma: float, n: int, random_state: int = None):
    demands = norm.rvs(loc=mu, scale=sigma, size=n, random_state=random_state)
    return demands


def stepwise_increasing(mu: dict, sigma: float, n: int, random_state: int = None):
    mu = [(t, m) for t, m in mu.items()]
    mu = [mu[j] for j in np.argsort([i[0] for i in mu])]
    # print(mu)
    assert mu[0][0] == 0
    times = [i[0] for i in mu]
    means = [i[1] for i in mu]

    demands = []
    for i, (time, mean) in enumerate(zip(times, means)):
        if i != len(mu) - 1:
            demands = np.append(
                demands, norm.rvs(loc=mean, scale=sigma, size=times[i + 1] - time, random_state=random_state)
            )
        else:
            demands = np.append(demands, norm.rvs(loc=mean, scale=sigma, size=n - time, random_state=random_state))

    return demands
