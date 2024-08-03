import numpy as np


class Meter:
    def __init__(self, p, c, r):
        assert r <= c <= p
        self.p = p  # product price
        self.c = c  # purchased cost
        self.r = r  # residual value
        self.placeholder = {
            "n_stockout": [],
            "n_holding": [],
            "stockout_cost": [],
            "holding_cost": [],
            "total_cost": [],
            "service_level": [],
        }

    def n_stockout(self, demand, purchased):
        demand = np.array(demand)
        purchased = np.array(purchased)
        diff = np.abs(demand - purchased)
        return diff[demand > purchased].sum()

    def n_holding(self, demand, purchased):
        demand = np.array(demand)
        purchased = np.array(purchased)
        diff = np.abs(demand - purchased)
        return diff[demand < purchased].sum()

    def stockout_cost(self, demand, purchased):
        return self.n_stockout(demand, purchased) * (self.p - self.c)

    def holding_cost(self, demand, purchased):
        return self.n_holding(demand, purchased) * (self.c - self.r)

    def total_cost(self, demand, purchased):
        return self.stockout_cost(demand, purchased) + self.holding_cost(demand, purchased)

    def service_level(self, demand, purchased):
        demand = np.array(demand)
        purchased = np.array(purchased)
        stockout_time = np.sum((demand - purchased) > 0)
        return 1 - stockout_time / len(demand)

    def __call__(self, demand, purchased, verbose=False):
        n_stockout_val = self.n_stockout(demand, purchased)
        n_holding_val = self.n_holding(demand, purchased)
        stockout_cost_val = self.stockout_cost(demand, purchased)
        holding_cost_val = self.holding_cost(demand, purchased)
        total_cost_val = self.total_cost(demand, purchased)
        service_level_val = self.service_level(demand, purchased)

        self.placeholder["n_stockout"].append(n_stockout_val)
        self.placeholder["n_holding"].append(n_holding_val)
        self.placeholder["stockout_cost"].append(stockout_cost_val)
        self.placeholder["holding_cost"].append(holding_cost_val)
        self.placeholder["total_cost"].append(total_cost_val)
        self.placeholder["service_level"].append(service_level_val)

        results = {
            "n_stockout": n_stockout_val,
            "n_holding": n_holding_val,
            "stockout_cost": stockout_cost_val,
            "holding_cost": holding_cost_val,
            "total_cost": total_cost_val,
            "service_level": service_level_val,
        }

        # Determine field width based on the longest key name
        field_width = max(len(key) for key in results.keys()) + 2  # Adding extra space for values
        # Print results in aligned columns
        result_row = " | ".join([f"{key}: {value:>{field_width}.4f}" for key, value in results.items()])
        if verbose:
            print(result_row)
        return results

    def calculate_statistics(self):
        statistics = {}
        for key, values in self.placeholder.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            statistics[key] = {"mean": mean_val, "std": std_val}

        # Determine field width based on the longest key name
        field_width = max(len(key) for key in statistics.keys()) + 2  # Adding extra space for values

        # Print statistics in aligned columns
        for key, stats in statistics.items():
            mean_val = stats["mean"]
            std_val = stats["std"]
            print(f"{key:>{field_width}} - mean: {mean_val:>{field_width}.4f}, std: {std_val:>{field_width}.4f}")

        return statistics
