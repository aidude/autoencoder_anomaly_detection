# input signals to be passed.

import torch
import numpy as np
import matplotlib.pyplot as plt


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def gaussian_sample(self, mu, sigma):
        return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian signal

    def random_sample(self):
        return lambda m, n: torch.rand(m, n)  # random signal


dataDistribution = DataDistribution()
x = dataDistribution.gaussian_sample(2, 0.7)

print(x)
# plt.plot(float(x))
# plt.show()
