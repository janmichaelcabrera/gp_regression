#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import pandas as pd
import scipy.stats as stats
sys.path.append('../../')
import gp_regression as gp

np.random.seed(3)

sigma = 0.01
x = np.linspace(0, 2*np.pi, num=30)
y_true = np.sin(x)
y_obs = y_true + stats.norm.rvs(scale=sigma**2, size=len(x))

cov = gp.covariance_functions(2, 150, 10**-6)

gp_obj = gp.gaussian_process(x, x, y_obs, sigma, cov, cov_func='matern_52')

print(gp_obj.log_marginal_likelihood())

gp_obj.optimize_lml()
print(gp_obj.cov_obj.b, gp_obj.cov_obj.tau_1)

ystar, var = gp_obj.regression()
print(gp_obj.log_marginal_likelihood())

y_high = ystar + np.sqrt(var)*1.96
y_low = ystar - np.sqrt(var)*1.96

samples = gp_obj.rvs(size=100)
prior = gp_obj.prior_rvs(size=100)

plt.figure()
plt.plot(x, prior.T, '-k', alpha=0.1)
plt.show()

plt.figure()
plt.plot(x, samples.T, '-k', alpha=0.1)
plt.plot(x, y_true)
plt.plot(x, ystar)
plt.scatter(x, y_obs)
plt.plot(x, y_high, '-b')
plt.plot(x, y_low, '-b')
plt.show()