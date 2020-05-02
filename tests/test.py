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

def testfit():
    sigma = 0.01
    x = np.linspace(0, 2*np.pi, num=30)
    y_true = np.sin(x)
    y_obs = y_true + stats.norm.rvs(scale=sigma**2, size=len(x))

    cov = gp.covariance_functions(2, 150, 10**-6)

    gp_obj = gp.gaussian_process(x, x, y_obs, sigma, cov, cov_func='squared_exponential')

    print(gp_obj.log_marginal_likelihood())

    gp_obj.optimize_lml()
    print(gp_obj.cov_obj.b, gp_obj.cov_obj.tau_1)

    ystar, var = gp_obj.regression()
    print(gp_obj.log_marginal_likelihood())

    y_high = ystar + np.sqrt(var)*1.96
    y_low = ystar - np.sqrt(var)*1.96

    samples = gp_obj.rvs(size=100)
    prior = gp_obj.prior_rvs(size=100)

    # plt.figure()
    # plt.plot(x, prior.T, '-k', alpha=0.1)
    # plt.show()

    plt.figure()
    plt.plot(x, samples.T, '-k', alpha=0.1)
    plt.plot(x, y_true)
    plt.plot(x, ystar)
    plt.scatter(x, y_obs)
    plt.plot(x, y_high, '-b')
    plt.plot(x, y_low, '-b')
    plt.show()

x = np.linspace(0, 4, num=5)

b, tau_1, tau_2 = 20, 1, 10**6

K_11 = gp.covariance_functions(b, tau_1, tau_2).squared_exponential(x, x)

# say dy/dx @ x=0,4 is 0 

ind_1 = 0
ind_2 = -1

slope_1 = 1
slope_2 = 1

L_1 = np.gradient(K_11, x, axis=0)[ind_1]
L_1_sq = np.gradient(L_1, x)[ind_1]

L_2 = np.gradient(K_11, x, axis=0)[ind_2]
L_2_sq = np.gradient(L_2, x)[ind_2]

L_12 = np.gradient(L_2, x)[ind_1]
L_21 = np.gradient(L_1, x)[ind_2]

L_sq = np.array([[L_1_sq, L_12],[L_21, L_2_sq]])
K_12 = np.row_stack((L_1, L_2))
K_21 = K_12.T

l_mu = np.array([slope_1, slope_2])

L_sq_inv = np.linalg.inv(L_sq)

u_fl = K_21 @ L_sq_inv @ l_mu

K_fl = K_11 - K_21 @ L_sq_inv @ K_12


samples = stats.multivariate_normal.rvs(mean=u_fl, cov=K_fl, size=100)

print(np.gradient(samples[0], x)[ind_1], np.gradient(samples[0], x)[ind_2])

plt.figure()
plt.plot(x, samples.T, '-k', alpha=0.1)
plt.show()