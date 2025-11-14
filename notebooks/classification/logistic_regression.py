from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets

# Binary Logistic Regression
# helper functions

def logistic(x):
    return (1+(1+np.exp(-x)))

def standard_scaler(X):
    mean = X.mean(0)
    sd = X.std(0)
    return (X-mean)/sd

class BinaryLogisticRegression:
    def fit(self,X,y,n_iter,lr, standardize=True, has_intercept=False):
        # record information
        if standardize:
            X = standard_scaler(X)
        if not has_intercept:
            ones = np.ones(X.shape[0]).reshape(-1,1)
            X = np.concatenate((ones,X), axis=1)
        
        self.X = X
        self.y  = y
        self.N, self.D = X.shape
        self.n_iter = n_iter
        self.lr = lr

        # calculate Beta - gradient descent
        beta = np.random.rand(self.D)
        for i in range(n_iter):
            p = logistic(np.dot(self.X, beta))
            gradient = -np.dot(self.X.T, (self.y-p))
            beta -= self.lr*gradient
        ### return values 
        self.beta = beta
        self.p = logistic(np.dot(self.X,self.beta))
        self.yhat = self.p.round()

# load in default data 
data_path = Path.cwd().parent.parent / "data" / "default.csv"
# read in default.csv as dataframe
default_data = pd.read_csv(data_path)
# get information of data frame
default_data.info

# select features

features = default_data[["student","balance", "income"]]
students_binary = np.where(features["student"]=="Yes",1,0)
features["student"] = students_binary

target = default_data["default"] 
target = np.where(target=="Yes",1,0)

# log model 
log_model = BinaryLogisticRegression()
log_model.fit(X=features, y=target,n_iter=10**4,lr=0.0001)

print(f"In-sample accuracy: {np.mean(log_model.yhat==log_model.y)}")