from distutils.log import error
import torch
import numpy as np
import random
from sklearn.linear_model import LinearRegression


#DATA GENERATION
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + np.random.randn(100, 1)

#SHUFFLE THE INDICES
idx = np.arange(100)
np.random.shuffle(idx)

#USE FIRST 80 RANDOM INDICES FOR TRAINING
train_idx = idx[:80]

#USE REMAINING INDICES FOR VALIDATION
val_idx = idx[80:]

#GENERATE TRAIN AND VALIDATION SETS
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

#INITIALIZE PARAMETERS A AND B
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

#SET LEARNING RATE
lr = 1e-1

#DEFINE NUMBER OF EPOCHS
n_epochs = 1000

for epoch in range(n_epochs):
    #MODELS PREDICTED OUTPUT
    yhat = a + b * x_train

    #MODEL ERROR
    error = (y_train - yhat)

    #MEAN SQUARE ERROR
    loss = (error ** 2).mean()

    #GRADIENTS
    a_grad = -2*error.mean()
    b_grad = -2* (x_train * error).mean()

    a = a - lr * a_grad
    b = b - lr * b_grad

print(a, b)


linr = LinearRegression()
linr.fit(x_train,y_train)
print(linr.intercept_,linr.coef_[0])