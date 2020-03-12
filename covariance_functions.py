#/packages/python/anaconda3/bin

import numpy as np
import scipy.stats as stats
from scipy.spatial import distance_matrix


class covariance_functions:
    """
    This class contains covariance functions for FPR
    """
    def __init__(self, b, tau_1, tau_2=10**-6):
        """
        Parameters
        ----------
            b: float (scalar)
                Bandwidth parameter

            tau_1: float (scalar)
                Scale parameter
                
            tau_2: float (scalar)
                Noise parameter
        """
        self.b = b
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def squared_exponential(self, x_1, x_2):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (array_like)
                Returns a Squared Exponential square covariance matrix of size(x)
                .. math:: C_{SE}(x_1, x_2) = \\tau_1^2 e^{-\\frac{1}{2} (d/b)^2} + \\tau_2^2 \\delta(x_1, x_2)
        """
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

    def matern_32(self, x_1, x_2):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (array_like)
                Returns a Matern (3,2) square covariance matrix of size(x)
                .. math:: C_{3,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{3} d / b] e^{-\\sqrt{3} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
        """
        if len(x_1.shape) == 1:
            X_1 = [[i] for i in x_1]
            X_2 = [[i] for i in x_2]
        else:
            X_1 = x_1
            X_2 = x_2

        C = distance_matrix(X_1, X_2)

        C = self.tau_1*(1 + np.sqrt(3)*(C/self.b))*np.exp(-np.sqrt(3)*(C/self.b))

        if np.array_equal(x_1, x_2) == True:
            C = C + self.tau_2*np.eye(x_1.shape[0])

        return C

    def matern_52(self, x_1, x_2):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (array_like)
                Returns a Matern (5,2) square covariance matrix of size(x)
                .. math:: C_{5,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{5} d / b + (5/3) (d/b)^2 ] e^{-\\sqrt{5} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
        """
        if len(x_1.shape) == 1:
            X_1 = [[i] for i in x_1]
            X_2 = [[i] for i in x_2]
        else:
            X_1 = x_1
            X_2 = x_2

        C = distance_matrix(X_1, X_2)

        C = self.tau_1*(1 + np.sqrt(5)*(C/self.b) + (5/3)*(C/self.b)**2)*np.exp(-np.sqrt(5)*(C/self.b))

        if np.array_equal(x_1, x_2) == True:
            C = C + self.tau_2*np.eye(x_1.shape[0])

        return C