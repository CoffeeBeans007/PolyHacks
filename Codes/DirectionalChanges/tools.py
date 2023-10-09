import re
import os
import sys
import math
import numpy as np
sys.path.append("Codes/DirectionalChanges/vso_dc/trading_model/stock_data")

class MarketL:

    def __init__(self,data):
        self.bid=data["Bid"]
        self.ask=data["Ask"]
        pass
    def Lagrangian(self,surprise,K,H1,H2):

        return(LSpace)
    
    #Total Pointwise-Entropy
    def surprise(self,
                 probabilities):
        """
        This function computes the total pointwise entropy of the market

        Args: probabilities: list of probabilities from K last events
        Ouputs: Total pointwise entropy

        """
        #probabilities is a list of probabilities
        #K is the length of the window
        pointwise_entropy = -np.log(probabilities)
        return(np.sum(pointwise_entropy))
    
    #Entropy rate state transitions
    def EntropyRate(self,probabilities,asymptoticDis):
        """
        This function computes the entropy rate of the market

        Args: 
        probabilities: transition matrix of the market
        asymptoticDis: asymptotic distribution of the market (matrix)

        Ouputs: Entropy rate (1)

        """
        Rate=np.dot(asymptoticDis,probabilities.dot(np.log(probabilities)))
        return(-np.sum(Rate))
    def EntropyRate2(self,probabilities,asymptoticDis):
        # TODO : check dimension of sum for correct axis (supposed to be 2^n-1 for i and j)
        # https://arxiv.org/pdf/1402.2198.pdf
        """
        This function computes the second orders of informativness of the market

        Args: 
        probabilities: transition matrix of the market
        asymptoticDis: asymptotic distribution of the market as array (1,k)

        Ouputs: Entropy rate (1)

        """
        eigenvalues, left_eigenvectors = np.linalg.eig(A.T)
        normalized_left_eigenvectors= left_eigenvectors/left_eigenvectors.sum(axis=0)

        Rate=0

        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                Rate+=normalized_left_eigenvectors[i]*normalized_left_eigenvectors[j]*np.cov(-np.log(probabilities[i]),-np.log(probabilities[j]))

        
        return(Rate)
    def test_EntropyRate2():
        
        # Test case 1
        probabilities = np.array([[0.5, 0.5], [0.3, 0.7]])
        asymptoticDis = np.array([0.4, 0.6])
        expected_output = 0.6931471805599453
        assert np.isclose(EntropyRate2(probabilities, asymptoticDis), expected_output)

        # Test case 2
        probabilities = np.array([[0.2, 0.8], [0.6, 0.4]])
        asymptoticDis = np.array([0.3, 0.7])
        expected_output = 0.6931471805599453
        assert np.isclose(EntropyRate2(probabilities, asymptoticDis), expected_output)

        # Test case 3
        probabilities = np.array([[0.1, 0.9], [0.9, 0.1]])
        asymptoticDis = np.array([0.5, 0.5])
        expected_output = 0.6931471805599453
        assert np.isclose(EntropyRate2(probabilities, asymptoticDis), expected_output)

        # Test case 4
        probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
        asymptoticDis = np.array([0.6, 0.4])
        expected_output = 0.6931471805599453
        assert np.isclose(EntropyRate2(probabilities, asymptoticDis), expected_output)

        # Test case 5
        probabilities = np.array([[0.3, 0.7], [0.5, 0.5]])
        asymptoticDis = np.array([0.7, 0.3])
        expected_output = 0.6931471805599453
        assert np.isclose(EntropyRate2(probabilities, asymptoticDis), expected_output)

        print("All test cases pass")
    
if __name__ == "__main__":
    MarketL.test_EntropyRate2()
    pass
        
    