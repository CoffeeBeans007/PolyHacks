import math
import numpy as np
import pandas as pd
import scipy


"""
Main functions for statistical analysis of time series data

"""

class StatsMethods:
    def __init__(self):
        pass

    def get_mean(self, data):
        """
        Calculates the mean of a given data set
        :param data: list of data points
        :return: mean of data
        """
        return np.mean(data)
    
    def kalman_meanR(self, data,rho,sigmaM,sigmaR):
    
        """
        Calculate estimated mean-reverting factors of data using Kalman filter

        :param data: list of data points
        :param rho: mean-reverting coefficient
        :param sigmaM: variance of mean-reverting
        :param sigmaR: variance of random walk component

        :return: estimated mean-reverting of data M and R (lists)
        """
        #Kalman gain (prior estimate of the covariance matrix of the current state)
        K1 = (2 * sigmaM**2) / (sigmaR * (math.sqrt(((rho + 1)**2) * sigmaR**2 + 4 * sigmaM**2) + rho * sigmaR + sigmaR) + 2 * (sigmaM**2))
        K2 = (2 * sigmaR) / (math.sqrt(((rho + 1)**2) * sigmaR**2 + 4 * sigmaM**2) - rho * sigmaR + sigmaR)

        K = np.array([K1, K2])

        #Initialize M and R
        M=[0]
        R=[data[0]]

        for i in range(1,len(data)):
            #Predict data[i] using previous M and R

            xhat    = rho*M[i-1]+R[i-1]
            e       = data[i]-xhat

            #Update M and R
            M.append(rho(M[i-1]+K[0]*e))
            R.append(R[i-1]+K[1]*e)

        return M,R




        
