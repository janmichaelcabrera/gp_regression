from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
sys.path.append('../../')
import gp_regression as gp

np.random.seed(3)

# Setup true curve and observations
sigma = 0.5
x = np.linspace(0, 2*np.pi, num=30)
y_true = np.sin(x)
y_obs = y_true + stats.norm.rvs(scale=sigma**2, size=len(x))

# Instantiate GP cov and GP objects
cov = gp.covariance_functions(b=10, tau_1=10, tau_2=10**-6)
gp_obj = gp.gaussian_process(x_star=x, x=x, y=y_obs, sigma_squared=sigma**2, cov_obj=cov, cov_func='squared_exponential')

# Run method for calculating posterior mean and posterior covariances
gp_obj.calc_mean_cov()

# Pass posterior mean
y_untrained = gp_obj.post_mean

# Optimize bandwidht parameters using log-marginal likelihood 
gp_obj.optimize_lml()

# Run method for calculating posterior mean and posterior covariances
gp_obj.calc_mean_cov()

# Pass posterior mean
y_hat = gp_obj.post_mean


plt.figure()
plt.plot(x, y_obs, 'k.', label='Obs')
plt.plot(x, y_true, label='True')
plt.plot(x, y_untrained, label='Untrained GP\n$b=10$; $\\tau_1=10$')
plt.plot(x, y_hat, label='Trained GP\n$b$={:.1f}'.format(gp_obj.cov_obj.b)+'; $\\tau_1$={:.1f}'.format(gp_obj.cov_obj.tau_1))
plt.legend(loc=0)
plt.show()