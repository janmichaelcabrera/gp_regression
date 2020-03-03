#/packages/python/anaconda3/bin

from __future__ import division
import numpy as np
import warnings
from .covariance_functions import *
from scipy.stats import multivariate_normal
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from numpy.linalg import inv
from scipy.optimize import minimize


class gaussian_process:
    """
    This class returns a vector of smoothed values given feature and response vectors
    """
    def __init__(self, x_star, x, y, sigma_squared, cov_obj, cov_func='squared_exponential'):
        """
        Parameters
        ----------
            x: float
                Feature vector

            hyperparams: float (tuple)
                Hyperparameters for a Guassian process

            y: float (Not required for generating random samples)
                Response vector

            x_star: float (Not required for generating random samples)
                Scalar or vector to be evaluated

            cov: str (optional)
                Covariance function to be used. Available functions are:
                    Squared exponential = 'squared_exponential'
                    Matern 5/2 = 'matern_52'
                    Matern 3/2 = 'matern_32'
        """
        self.x_star = x_star
        self.x = x
        self.y = y
        self.sigma_squared = sigma_squared
        self.model_error = self.sigma_squared*np.eye(self.x.shape[0])
        # if np.isscalar(self.sigma_squared) :
        #     self.model_error = self.sigma_squared*np.eye(self.x.shape[0])
        # elif len(self.sigma_squared) == self.x.shape[0]:
        #     self.model_error = np.diag(self.sigma_squared)
        # else:
        #     raise ValueError('Feature shape, %i and error shape, %i mismatch: '%self.x.shape[0], len(self.sigma_squared))
        self.cov_obj = cov_obj
        self.cov_func = cov_func

        self.calc_cov_components()


    def calc_cov_components(self):
        self.C_x_star_x = getattr(self.cov_obj, self.cov_func)(self.x_star, self.x)
        self.C_xx = getattr(self.cov_obj, self.cov_func)(self.x, self.x)
        self.C_star_star = getattr(self.cov_obj, self.cov_func)(self.x_star, self.x_star)

    def calc_mean_cov(self):
        # Calculate weights matrix,  W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}
        self.M = inv(self.C_xx + self.model_error)

        # Calculates posterior variance, var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        self.post_cov = self.C_star_star - self.C_x_star_x @ self.M @ np.transpose(self.C_x_star_x)

        weights = self.C_x_star_x @ self.M
        # Calculate y_star, y^* = W^T y
        self.post_mean = np.transpose(weights) @ self.y

    def regression(self):
        self.calc_mean_cov()
        self.post_var = np.diag(self.post_cov)
        return self.post_mean, self.post_var

    def log_marginal_likelihood(self):
        return multivariate_normal.logpdf(self.y, cov=(self.model_error + self.C_xx))


    def optimize_lml(self, bounds=[(10**-3, 10**3),(10**-3, 10**5)]):
        """
        Returns
        ----------
            hyperparams: float (tuple)
                Returns optimized b and tau_1_squared of hyperparameters. Optimized using BFGS from scipy.optimize.minimize method.
        """
        # Initial params
        b, tau_1 = self.cov_obj.b, self.cov_obj.tau_1
        params = [b, tau_1]

        # Wrapper function used to optimize the log marginal-likelihood
        def func(params):
            # Unpacks parameters to be optimized
            b, tau_1 = params

            temp_cov = covariance_functions(b, tau_1)

            # Evaluates covariance function
            temp_C_xx = getattr(temp_cov, self.cov_func)(self.x, self.x)

            # Evaluates covariance of marginal-likelihood: \sigma^2 I + C
            covariance = self.model_error + temp_C_xx

            # Returns the negative of the marginal log-likelihood multivariate normal (in order to maximize the evaluation)
            return -multivariate_normal.logpdf(self.y, cov=covariance)

        # Uses BFGS optimization function to find optimal b and tau_1_squared.     
        res = minimize(func, [b, tau_1], bounds=bounds)

        # Unpacks results of the optimization step
        self.cov_obj.b, self.cov_obj.tau_1 = res.x

        self.calc_cov_components()

    def rvs(self, size=None):
        if not hasattr(self, 'M'):
            self.calc_mean_cov()
            print('test')
        return stats.multivariate_normal.rvs(mean=self.post_mean, cov=self.post_cov, size=size)