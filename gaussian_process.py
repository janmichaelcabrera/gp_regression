#/packages/python/anaconda3/bin

from __future__ import division
import numpy as np
import warnings
from scipy.stats import multivariate_normal
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from numpy.linalg import inv
from scipy.optimize import minimize


class gaussian_process:
    """
    This class returns a vector of smoothed values given feature and response vectors
    """
    def __init__(self, x, hyperparams=[10, 10, 10**-6], y = [], x_star = [], cov='matern_52'):
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
        self.x = x
        self.hyperparams = hyperparams
        self.cov = cov
        self.y = y
        self.x_star = x_star



    def approx_var(self):
        """
        Returns
        ----------
            variance: float(scalar)
                Approximates the variance for a given problem using residual sum of squared errors. The function runs a Gaussian Process once for the approximation

        Attributes
        ----------
            residuals: float, len(y)
                residuals of prediction from data
                .. math:: r_i = y_i - \\hat{y}_i
        """

        # Evaluate covariance
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)

        # Assume initial estimate for variance
        variance = 1

        # Calculate weights
        weights = C_xx @ inv(C_xx + variance*np.eye(self.x.shape[0]))

        # Calculate predictions from GP
        y_star = np.transpose(weights) @ self.y

        # Calculate residuals from the above prediction
        self.residuals = self.y - y_star

        # Residual sum of squared errors
        rss = (self.residuals**2).sum()

        # Approximate standard error of fit
        return rss/(len(y_star)-1)

    def smoother(self, variance=[]):
        """
        Parameters
        ----------
            variance: float (scalar; optional)
                If not specified, the sample variance is assumed to be one

        Returns
        ----------
            y_star: float, len(x_star)
                Predictor for x_star, 
                .. math:: y^* = W^T y
                .. math:: W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}

            post_var: float, len(x_star)
                Posterior variance given the observed data
                .. math:: var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        """
        # If a sample variance is not spcified, assume it is 1
        if not variance:
            variance = 1

        # Evaluate C(x^*, x)
        C_x_star_x = getattr(covariance_functions, self.cov)(self.x_star, self.x, self.hyperparams)
        # Evaluate C(x, x)
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        # Evaluate C(x^*, x^*)
        C_star_star = getattr(covariance_functions, self.cov)(self.x_star, self.x_star, self.hyperparams)

        # Calculate weights matrix,  W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}
        weights = C_x_star_x @ inv(C_xx + variance*np.eye(self.x.shape[0]))
        
        # Calculate y_star, y^* = W^T y
        y_star = np.transpose(weights) @ self.y

        # Calculates posterior variance, var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        self.post_cov = C_star_star - C_x_star_x @ inv(C_xx + variance*np.eye(self.x.shape[0])) @ np.transpose(C_x_star_x)

        self.post_mean = inv(C_xx + variance*np.eye(self.x.shape[0])) @ C_x_star_x.T

        post_var = np.diag(self.post_cov)
        return y_star, post_var

    def log_marginal_likelihood(self, hyperparams, variance=[]):
        """
        Parameters
        ----------
            hyperparams: float (tuple)
                Hyperparameters for a Guassian process

            variance: float (scalar; optional)
                If not specified, the sample variance is assumed to be one

        Returns
        ----------
            p_y: float
                Returns the log marginal-likelihood of a multivariate gaussian evaluated at the hyperparams
                .. math: y \\sim N(0, \\sigma^2 I + C)
        """
        if not variance:
            variance = 1

        # Unpack hypereparameters
        b, tau_1_squared, tau_2_squared = hyperparams

        # Evaluate C(x, x)
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, hyperparams)

        # Evaluates covariance of marginal-likelihood: \sigma^2 I + C
        covariance = variance*np.eye(self.x.shape[0]) + C_xx
        return multivariate_normal.logpdf(self.y, cov=covariance)

    def optimize_lml(self):
        """
        Returns
        ----------
            hyperparams: float (tuple)
                Returns optimized b and tau_1_squared of hyperparameters. Optimized using BFGS from scipy.optimize.minimize method.
        """
        # Unpacks hyperparameters
        b, tau_1_squared, tau_2_squared = self.hyperparams

        # Wrapper function used to optimize the log marginal-likelihood
        def func(params):
            # Unpacks parameters to be optimized
            b, tau_1_squared = params
            variance = 1

            # Repacks paramers
            hyperparams = b, tau_1_squared, 10**-6

            # Evaluates covariance function
            C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, hyperparams)

            # Evaluates covariance of marginal-likelihood: \sigma^2 I + C
            covariance = variance*np.eye(self.x.shape[0]) + C_xx

            # Returns the negative of the marginal log-likelihood multivariate normal (in order to maximize the evaluation)
            return -multivariate_normal.logpdf(self.y, cov=covariance)

        # Uses BFGS optimization function to find optimal b and tau_1_squared.     
        res = minimize(func, [b, tau_1_squared], bounds=[(0.1, 10**3),(0.1, 10**5)])

        # Unpacks results of the optimization step
        b, tau_1_squared = res.x

        # Updates the model hyperparameters
        self.hyperparams = b, tau_1_squared, tau_2_squared
        return self.hyperparams

    def generate_random_samples(self, mean=[]):
        """
        Parameters
        ----------
            mean: float, vector
                Mean vector for multivariate normal, should have same shape as self.x. If not specified, assumed to be mean zero multivariate normal.

        Returns
        ----------
            fx: float, len(x)
                Returns random samples from a multivariate with specified mean

        Raises
        ----------
            ValueError
                If mean vector shape and x shape do not match
        """

        if not mean:
            mean = np.zeros(self.x.shape[0])

        if mean.shape[0] != self.x.shape[0]:
            raise ValueError('Mean shape, %i  and x shape, %i do not match'%(mean.shape[0], self.x.shape[0]))

        covariance = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        fx = multivariate_normal.rvs(mean=mean, cov=covariance)
        return fx