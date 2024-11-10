import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union

class ScalarKalmanFilter():
    """
    A class to represent a scalar Kalman filter.

    Attributes:
    -----------
    n : int
        Number of time steps.
    h : float
        State transition coefficient.
    a : Union[int, float]
        Observation coefficient.
    sigma_Z : float
        Standard deviation of the process noise.
    sigma_W : float
        Standard deviation of the observation noise.
    sigma_Y : float
        Initial error covariance.
    Y_hat : np.ndarray
        Predicted state.
    X_hat : np.ndarray
        Predicted observation.
    R : np.ndarray
        Error in the prediction.
    K : np.ndarray
        Kalman gain.
    """

    def __init__(self, n: int, h: float, a: Union[int, float], sigma_Z: float, sigma_W: float):
        """
        Constructs all the necessary attributes for the ScalarKalmanFilter object.

        Parameters:
        -----------
        n : int
            Number of time steps.
        h : float
            State transition coefficient.
        a : Union[int, float]
            Observation coefficient.
        sigma_Z : float
            Standard deviation of the process noise.
        sigma_W : float
            Standard deviation of the observation noise.
        """
        self.n = n
        self.h = h
        self.a = a
        self.sigma_Z = sigma_Z
        self.sigma_W = sigma_W
        self.sigma_Y = sigma_Z/(1-h**2)
        
        #preallocate memory
        self.Y_hat = np.zeros(n) # predicted state
        self.X_hat = np.zeros(n) # predicted observation
        self.R = np.zeros(n) # error in the prediction
        self.K = np.zeros(n) # kalman gain (also called B)
        
        # initialize variables
        self.Y_hat[0] = 0 # initialize to 0
        self.R[0] = self.sigma_Y
        
    def _predict(self, i: int):
        """
        Predicts the next state and observation.

        Parameters:
        -----------
        i : int
            Current time step.
        """
        self.Y_hat[i] = self.h * self.Y_hat[i-1]
        self.X_hat[i] = self.a * self.Y_hat[i]
        self.R[i] = self.h**2 * self.R[i-1] + self.sigma_Z**2

    def _update(self, i: int, X: float):
        """
        Updates the state estimate and error covariance.

        Parameters:
        -----------
        i : int
            Current time step.
        X : float
            Current observation.
        """
        self.K[i] = (self.a * self.R[i])/(self.a**2 * self.R[i] + self.sigma_W**2)
        self.Y_hat[i] = self.Y_hat[i] + self.K[i] * (X - self.X_hat[i])
        self.R[i] = (1-self.K[i] * self.a) * self.R[i]
        
    def fit(self, n: int, X: np.ndarray, Y: np.ndarray = None):
        """
        Fits the Kalman filter to the observations.

        Parameters:
        -----------
        n : int
            Number of time steps.
        X : np.ndarray
            Observations.
        Y : np.ndarray
            True states.
        """
        for i in tqdm(range(1, n), desc="Fitting Kalman filter"):
            self._predict(i)
            self._update(i, X[i])
        if Y is not None:
            print(f"Fitted Kalman filter with MSE: {self._calculate_mse(Y)}")
        else:
            print("Fitted Kalman filter. Could not calculate MSE (no true states provided).")
    
    def get_predicted(self):
        """
        Returns the predicted states.

        Returns:
        --------
        np.ndarray
            Predicted states.
        """
        return self.Y_hat

    def _calculate_mse(self, Y: np.ndarray):
        """
        Calculates the mean squared error of the predictions.

        Parameters:
        -----------
        Y : np.ndarray
            True states.

        Returns:
        --------
        float
            Mean squared error.
        """
        return np.mean((Y - self.Y_hat)**2)
    
class KalmanFilter():
    def __init__(self, n: int, system_features: int, obs_features: int, H: np.ndarray, A: np.ndarray, sigma_Z: float, sigma_W: float, Q_Z: np.ndarray = None, Q_W: np.ndarray = None, Q_Y: np.ndarray = None):
        self.n = n
        self.r = system_features
        self.s = obs_features
        self.H = H # r by r dimension
        self.A = A # s by r dimension
        self.sigma_Z = sigma_Z
        self.sigma_W = sigma_W

        
        if Q_Z is None:
            self.Q_Z = np.eye(self.r) * sigma_Z # process noise covariance
        else:
            self.Q_Z = Q_Z
        if Q_W is None:
            self.Q_W = np.eye(self.s) * sigma_W # observation noise covariance
        else:
            self.Q_W = Q_W
        if Q_Y is None:
            self.Q_Y = Q_Z # initial error covariance (assumption that the initial state is 0 and the uncertainty is the same as the process noise)
        else:
            self.Q_Y = Q_Y
        
        # preallocate memory
        self.Y_hat = np.zeros((n, self.r))
        self.X_hat = np.zeros((n, self.s))
        self.R = np.zeros((n, self.r, self.r))
        self.K = np.zeros((n, self.r, self.s))
        
        # initialize variables
        self.Y_hat[0] = np.zeros(self.r) # initialize to 0
        self.R[0] = self.Q_Y
        
    def _predict(self, i: int):
        self.Y_hat[i] = self.H[i] @ self.Y_hat[i-1]
        self.X_hat[i] = self.A[i] @ self.Y_hat[i]
        self.R[i] = self.H[i] @ self.R[i-1] @ self.H[i].T + self.Q_Z
    
    def _update(self, i: int, X: np.ndarray):
        self.K[i] = self.R[i] @ self.A[i].T @ np.linalg.inv(self.A[i] @ self.R[i] @ self.A[i].T + self.Q_W)
        self.Y_hat[i] = self.Y_hat[i] + self.K[i] @ (X[i] - self.X_hat[i])
        self.R[i] = (np.eye(self.K.shape[1]) - self.K[i] @ self.A[i]) @ self.R[i]
        
    def fit(self, n: int, X: np.ndarray, Y: np.ndarray = None):
        for i in tqdm(range(1, n), desc="Fitting Kalman filter"):
            self._predict(i)
            self._update(i, X[i])
        if Y is not None:
            print(f"Fitted Kalman filter with MSE: {self._calculate_mse(Y)}")
        else:
            print("Fitted Kalman filter. Could not calculate MSE (no true states provided).")
    
    def get_predicted(self):
        return self.Y_hat
    
    def _calculate_mse(self, Y: np.ndarray):
        return np.mean((Y - self.Y_hat)**2)
        
    

if __name__ == "__main__":
    n = 1000000
    h = 0.95
    a = 1
    sigma_Z = 0.1
    sigma_W = 1
    
    # generate data
    Y = np.zeros(n)
    X = np.zeros(n)
    Z = np.random.normal(0, sigma_Z, n)
    W = np.random.normal(0, sigma_W, n)
    for i in range(1, n):
        Y[i] = h * Y[i-1] + Z[i]
        X[i] = a * Y[i] + W[i]
        
    # fit the Kalman filter
    kalman_filter = ScalarKalmanFilter(n, h, a, sigma_Z, sigma_W)
    kalman_filter.fit(n, X, Y)
    