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
    This class instantiates an object for basic mean zero Guassian Process Regression
    """
    def __init__(self, x_star, x, y, sigma_squared, cov_obj, cov_func='squared_exponential'):
        """
        Parameters
        ----------
            x_star: float
                Scalar or vector to be evaluated

            x: float
                Feature vector

            y: float
                Response vector

            sigma_squared: float (scalar)
                Assumed homoscedastic error for error model
                .. math: y_i = f(x_i) + e_i, e_i \\sim N(0, \\sigma^2)

            cov_obj: object
                Covariance function object

            cov_func: str (optional)
                Covariance function to be used. Available functions are:
                    Squared exponential = 'squared_exponential'
                    Matern 5/2 = 'matern_52'
                    Matern 3/2 = 'matern_32'
        Attributes
        ----------
            model_error: float (array_like)
                Matrix representing the error model
                .. math: \\text{model_error} = \\sigma^2 I
        """
        self.x_star = x_star
        self.x = x
        self.y = y
        self.sigma_squared = sigma_squared
        self.model_error = self.sigma_squared*np.eye(self.x.shape[0])
        self.cov_obj = cov_obj
        self.cov_func = cov_func
        self.calc_cov_components()


    def calc_cov_components(self):
        """
        Notes
        ----------
            Calculates the block matrix components of the joint Guassian between the GP prior at x and at x^*
            .. math :: [f(x_1), ... , f(x_N), f(x^*)]^T \\sim \text{N} \\left ( \\begin{bmatrix} \\mathbf{m} \\ m^*\\end{bmatrix}, \\begin{bmatrix} C(\\mathbf{x, x}) & C(\\mathbf{x^*x}) \\ C(\\mathbf{x^*x})^T & C(\\mathbf{x^*, x^*})\\end{bmatrix}\\right)

        Attributes
        ----------
            C_xx: array_like
                Covariance function evaluated with x

            C_x_star_x: array_like
                Covariance function evaluated with x^* and x

            C_star_star: array_like
                Covariance function evaluated with x^*
        """
        self.C_xx = getattr(self.cov_obj, self.cov_func)(self.x, self.x)
        self.C_x_star_x = getattr(self.cov_obj, self.cov_func)(self.x_star, self.x)
        self.C_star_star = getattr(self.cov_obj, self.cov_func)(self.x_star, self.x_star)

    def calc_mean_cov(self):
        """
        Notes
        ----------
            Calculates the posterior mean and covariance of the GP

        Attributes
        ----------
            post_mean: float (vector)
                Posterior mean of the GP

            post_cov: float (array_like)
                Posterior covariance of the GP

        """
        # Calculate weights matrix,  W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}
        M = inv(self.C_xx + self.model_error)

        weights = self.C_x_star_x @ M

        # Calculate y_star, y^* = W^T y
        self.post_mean = np.transpose(weights) @ self.y

        # Calculates posterior variance, var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        self.post_cov = self.C_star_star - self.C_x_star_x @ M @ np.transpose(self.C_x_star_x)

    def regression(self):
        """
        Attributes
        ----------
            post_var: float (vector)
                Diagonal elements of posterior covariance

        Returns
        ----------
            post_mean: float (vector)
                Posterior mean of GP

            post_var: float (vector)
                Diagonal elements of posterior covariance
        """
        self.calc_mean_cov()
        self.post_var = np.diag(self.post_cov)
        return self.post_mean, self.post_var

    def log_marginal_likelihood(self):
        """
        Returns
        ----------
            log_marinal_likelihood: float (scalar)
                Log marginal likelihood evaluated at y with covariance:
                .. math :: \\sigma^2 I + C(x, x)
        """
        return multivariate_normal.logpdf(self.y, cov=(self.model_error + self.C_xx))


    def optimize_lml(self, bounds=[(10**-3, 10**3),(10**-3, 10**5)], method=None):
        """
        Parameters
        ----------
            bounds: sequence, or bounds (optional)
                1. Instance of Bounds class
                2. Sequence of (min, max) pairs for each b and tau_1

            method: str or callable (optional)
                Type of solver (see scipy.optimize.minimize for details)

        Notes
        ----------
            Optimizes hyperparameters of the covariance function by maximimizing the log marginal likelihood. Optimized using BFGS by default from scipy.optimize.minimize method.
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

        # Uses optimization function to find optimal b and tau_1_squared.     
        res = minimize(func, [b, tau_1], bounds=bounds, method=method)

        # Unpacks results of the optimization step
        self.cov_obj.b, self.cov_obj.tau_1 = res.x

        # Recalculates the covariance components
        self.calc_cov_components()

    def rvs(self, size=None):
        """
        Parameters
        -----------
            size: int (optional)
                Number of samples to return. If None, returns only one sample.

        Returns
        ----------
            samples: float (vector or array_like)
                Samples drawn from a multivariate normal described by the posterior GP mean and covariance
        """
        if not hasattr(self, 'post_mean'):
            self.calc_mean_cov()
        return stats.multivariate_normal.rvs(mean=self.post_mean, cov=self.post_cov, size=size)

    def prior_rvs(self, size=None):
        """
        Parameters
        -----------
            size: int (optional)
                Number of samples to return. If None, returns only one sample.

        Returns
        ----------
            samples: float (vector or array_like)
                Samples drawn from a multivariate normal described by the mean zero GP prior
        """
        return stats.multivariate_normal.rvs(mean=np.zeros(len(self.y)), cov=self.C_xx, size=size)