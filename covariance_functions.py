#/packages/python/anaconda3/bin

import numpy as np
import scipy.stats as stats
from scipy.spatial import distance_matrix


class covariance_functions:
    """
    This class contains covariance function for Gaussian process regression
    """
    def __init__(self, b, tau_1, tau_2=10**-6):
        self.b = b
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def squared_exponential(self, x_1, x_2):
        if len(x_1.shape) == 1:
            X_1 = [[i] for i in x_1]
            X_2 = [[i] for i in x_2]
        else:
            X_1 = x_1
            X_2 = x_2

        D = distance_matrix(X_1, X_2)

        C = self.tau_1*np.exp(-(1/2)*(D/self.b)**2)

        if np.array_equal(x_1, x_2) == True:
            C = C + self.tau_2*np.eye(x_1.shape[0])

        return C

    # def __init__(self):
    #     pass

    # def squared_exponential(x_1, x_2, hyperparams):
    #     """
    #     Parameters
    #     ----------
    #         x: float (vector)
    #             Vector of points

    #     Returns
    #     ----------
    #         C: float (matrix)
    #             Returns a Matern (5,2) square covariance matrix of size(x)
    #             .. math:: C_{SE}(x_1, x_2) = \\tau_1^2 e^{-\\frac{1}{2} (d/b)^2} + \\tau_2^2 \\delta(x_1, x_2)
    #     """

    #     # Unpack hypereparameters
    #     b, tau_1_squared, tau_2_squared = hyperparams

    #     # Initialize covariance matrix
    #     C = np.zeros((x_1.shape[0], x_2.shape[0]))

    #     if len(x_1.shape) == 1:
    #         X_1 = [[i] for i in x_1]
    #         X_2 = [[i] for i in x_2]
    #     else:
    #         X_1 = x_1
    #         X_2 = x_2

    #     C = distance_matrix(X_1, X_2)

    #     C = tau_1_squared*np.exp(-(1/2)*(C/b)**2)

    #     if np.array_equal(x_1, x_2) == True:
    #         C = C + tau_2_squared*np.eye(x_1.shape[0])

    #     return C

    # def matern_32(x_1, x_2, hyperparams):
    #     """
    #     Parameters
    #     ----------
    #         x: float (vector)
    #             Vector of points

    #     Returns
    #     ----------
    #         C: float (matrix)
    #             Returns a Matern (5,2) square covariance matrix of size(x)
    #             .. math:: C_{5,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{5} d / b + (5/3) (d/b)^2 ] e^{-\\sqrt{5} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
    #     """

    #     # Unpack hypereparameters
    #     b, tau_1_squared, tau_2_squared = hyperparams

    #     # Initialize covariance matrix
    #     C = np.zeros((x_1.shape[0], x_2.shape[0]))

    #     if len(x_1.shape) == 1:
    #         X_1 = [[i] for i in x_1]
    #         X_2 = [[i] for i in x_2]
    #     else:
    #         X_1 = x_1
    #         X_2 = x_2

    #     C = distance_matrix(X_1, X_2)

    #     C = tau_1_squared*(1 + np.sqrt(3)*(C/b))*np.exp(-np.sqrt(3)*(C/b))

    #     if np.array_equal(x_1, x_2) == True:
    #         C = C + tau_2_squared*np.eye(x_1.shape[0])

    #     return C

    # def matern_52(x_1, x_2, hyperparams):
    #     """
    #     Parameters
    #     ----------
    #         x: float (vector)
    #             Vector of points

    #     Returns
    #     ----------
    #         C: float (matrix)
    #             Returns a Matern (5,2) square covariance matrix of size(x)
    #             .. math:: C_{5,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{5} d / b + (5/3) (d/b)^2 ] e^{-\\sqrt{5} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
    #     """

    #     # Unpack hypereparameters
    #     b, tau_1_squared, tau_2_squared = hyperparams


    #     # Initialize covariance matrix
    #     C = np.zeros((x_1.shape[0], x_2.shape[0]))

    #     if len(x_1.shape) == 1:
    #         X_1 = [[i] for i in x_1]
    #         X_2 = [[i] for i in x_2]
    #     else:
    #         X_1 = x_1
    #         X_2 = x_2

    #     C = distance_matrix(X_1, X_2)

    #     C = tau_1_squared*(1 + np.sqrt(5)*(C/b) + (5/3)*(C/b)**2)*np.exp(-np.sqrt(5)*(C/b))

    #     if np.array_equal(x_1, x_2) == True:
    #         C = C + tau_2_squared*np.eye(x_1.shape[0])

    #     return C