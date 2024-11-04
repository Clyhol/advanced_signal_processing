import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 1000

# preallocate memory
Y = np.zeros(n)
Y_hat = np.zeros(n)
X = np.zeros(n)
X_hat = np.zeros(n)
Z = np.zeros(n)
W = np.zeros(n)
R = np.zeros(n)
B = np.zeros(n)
MSE = np.zeros(n)

h = 0.95
a = 1 
sigma_Z = 0.1
sigma_W = 1
sigma_Y = sigma_Z/(1-h**2)

# generate data and initialize variables
Z = np.random.normal(0, sigma_Z, n)
W = np.random.normal(0, sigma_W, n)

Y[0] = np.random.normal(0,sigma_Y)
Y_hat[0] = 0
R[0] = sigma_Y
X[0] = Y[0] + W[0]

# Kalman filter
for i in range(1, n):
    # generate true state and observation
    Y[i] = h*Y[i-1] + Z[i]
    X[i] = a*Y[i] + W[i]
    
    # prediction step
    Y_hat[i] = h * Y_hat[i-1] # predict the next state
    X_hat[i] = a * Y_hat[i] # predict the next observation
    R[i] = h**2*R[i-1] + sigma_Z**2 # get the error in the prediction
    
    # update step
    B[i] = (a *R[i])/(a**2 * R[i] + sigma_W**2) # kalman gain
    Y_hat[i] = Y_hat[i] + B[i] * (X[i] - X_hat[i]) # update the prediction using the error in the observation prediction
    R[i] = (1 - B[i]*a)*R[i] # update the error in the prediction
    
    MSE[i] = np.mean((Y[:i] - Y_hat[:i])**2)
    
print(f"True state: {Y[n-5:n]}\n Predicted state: {Y_hat[n-5:n]}")
print(f"MSE: {np.mean((Y - Y_hat)**2)}")

plt.plot(MSE)
plt.grid(True)
plt.ylim(0, 1)
#plt.legend()
plt.show()
    
    
    
