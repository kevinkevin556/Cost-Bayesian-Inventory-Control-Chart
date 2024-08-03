import numpy as np

from control_chart.shewhart import Shewhart
from meter import Meter
from random_demand import stable, stepwise_increasing

p = 6
c = 5
r = 3
meter = Meter(p, c, r)
demand = stable(mu=1000, sigma=10, n=52)
control_chart = Shewhart(mu=1000, sigma=10, p=p, c=c, r=r)

purchased = []
for d in demand:
    control_chart.purchase()
    control_chart.test(d)
