import numpy as np

np.set_printoptions(superess=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


class FisherLinearDiscriminant:
    def fit(self, X, y):

        # save stuff
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape

        ## calculate class means
        X0 = X[y==0]
        X1 = X[y==1]
        mu0 = x0.mean(0)
        mu1 = x1.mean(0)

        # sigma_w
        Sigma_w = np.empty((self.D, self.D))
        for x0 in X0:
            x0_minus_mu0 = (x0 - mu0).reshape(-1,1)
            Sigma_w += np.dot(x0_minus_mu0, x0_minus_mu0.T)
        for x1 in X1:
            x1_minus_mu1 = (x1 - mu1).reshape(-1,1)
            Sigma_w += np.dot(x1_minus_mu1, x1_minus_mu1.T)
        Sigma_w_inverse = np.linalg.inv(Sigma_w)

        ## beta
        self.beta = np.dot(Sigma_w_inverse,mu1 - mu0)
        self.f = np.dot(X, self.beta)
