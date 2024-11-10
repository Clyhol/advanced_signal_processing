import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lecture8.Kalman_filters import ScalarKalmanFilter, KalmanFilter

# FILE: Lecture8/test_Kalman_filters.py
class TestScalarKalmanFilter(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.h = 0.95
        self.a = 1
        self.sigma_Z = 0.1
        self.sigma_W = 1
        self.kalman_filter = ScalarKalmanFilter(self.n, self.h, self.a, self.sigma_Z, self.sigma_W)

    def test_initialization(self):
        self.assertEqual(self.kalman_filter.n, self.n)
        self.assertEqual(self.kalman_filter.h, self.h)
        self.assertEqual(self.kalman_filter.a, self.a)
        self.assertEqual(self.kalman_filter.sigma_Z, self.sigma_Z)
        self.assertEqual(self.kalman_filter.sigma_W, self.sigma_W)
        self.assertEqual(self.kalman_filter.Y_hat[0], 0)
        self.assertEqual(self.kalman_filter.R[0], self.sigma_Z/(1-self.h**2))

    def test_predict(self):
        self.kalman_filter._predict(1)
        self.assertAlmostEqual(self.kalman_filter.Y_hat[1], 0)
        self.assertAlmostEqual(self.kalman_filter.X_hat[1], 0)
        self.assertAlmostEqual(self.kalman_filter.R[1], self.h**2 * self.kalman_filter.R[0] + self.sigma_Z**2)

    def test_update(self):
        self.kalman_filter._predict(1)
        self.kalman_filter._update(1, 1.0)
        self.assertNotEqual(self.kalman_filter.Y_hat[1], 0)
        self.assertNotEqual(self.kalman_filter.R[1], self.h**2 * self.kalman_filter.R[0] + self.sigma_Z**2)

    def test_fit(self):
        X = np.random.normal(0, self.sigma_W, self.n)
        self.kalman_filter.fit(self.n, X)
        self.assertEqual(len(self.kalman_filter.Y_hat), self.n)

    def test_get_predicted(self):
        X = np.random.normal(0, self.sigma_W, self.n)
        self.kalman_filter.fit(self.n, X)
        predicted = self.kalman_filter.get_predicted()
        self.assertEqual(len(predicted), self.n)

    def test_calculate_mse(self):
        X = np.random.normal(0, self.sigma_W, self.n)
        Y = np.random.normal(0, self.sigma_W, self.n)
        self.kalman_filter.fit(self.n, X, Y)
        mse = self.kalman_filter._calculate_mse(Y)
        self.assertGreaterEqual(mse, 0)

class TestKalmanFilter(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.system_features = 2
        self.obs_features = 1
        self.H = np.array([np.eye(self.system_features) for _ in range(self.n)])
        self.A = np.array([np.ones((self.obs_features, self.system_features)) for _ in range(self.n)])
        self.sigma_Z = 0.1
        self.sigma_W = 1
        self.kalman_filter = KalmanFilter(self.n, self.system_features, self.obs_features, self.H, self.A, self.sigma_Z, self.sigma_W)

    def test_initialization(self):
        self.assertEqual(self.kalman_filter.n, self.n)
        self.assertEqual(self.kalman_filter.r, self.system_features)
        self.assertEqual(self.kalman_filter.s, self.obs_features)
        self.assertEqual(self.kalman_filter.sigma_Z, self.sigma_Z)
        self.assertEqual(self.kalman_filter.sigma_W, self.sigma_W)
        self.assertTrue(np.array_equal(self.kalman_filter.Y_hat[0], np.zeros(self.system_features)))
        self.assertTrue(np.array_equal(self.kalman_filter.R[0], self.kalman_filter.Q_Y))

    def test_predict(self):
        self.kalman_filter._predict(1)
        self.assertTrue(np.array_equal(self.kalman_filter.Y_hat[1], self.H[1] @ self.kalman_filter.Y_hat[0]))
        self.assertTrue(np.array_equal(self.kalman_filter.X_hat[1], self.A[1] @ self.kalman_filter.Y_hat[1]))
        self.assertTrue(np.array_equal(self.kalman_filter.R[1], self.H[1] @ self.kalman_filter.R[0] @ self.H[1].T + self.kalman_filter.Q_Z))

    def test_update(self):
        X = np.random.normal(0, self.sigma_W, (self.n, self.obs_features))
        self.kalman_filter._predict(1)
        self.kalman_filter._update(1, X)
        self.assertFalse(np.array_equal(self.kalman_filter.Y_hat[1], self.H[1] @ self.kalman_filter.Y_hat[0]))
        self.assertFalse(np.array_equal(self.kalman_filter.R[1], self.H[1] @ self.kalman_filter.R[0] @ self.H[1].T + self.kalman_filter.Q_Z))

    def test_fit(self):
        X = np.random.normal(0, self.sigma_W, (self.n, self.obs_features))
        self.kalman_filter.fit(self.n, X)
        self.assertEqual(len(self.kalman_filter.Y_hat), self.n)

    def test_get_predicted(self):
        X = np.random.normal(0, self.sigma_W, (self.n, self.obs_features))
        self.kalman_filter.fit(self.n, X)
        predicted = self.kalman_filter.get_predicted()
        self.assertEqual(len(predicted), self.n)

    def test_calculate_mse(self):
        X = np.random.normal(0, self.sigma_W, (self.n, self.obs_features))
        Y = np.random.normal(0, self.sigma_W, (self.n, self.system_features))
        self.kalman_filter.fit(self.n, X, Y)
        mse = self.kalman_filter._calculate_mse(Y)
        self.assertGreaterEqual(mse, 0)

if __name__ == '__main__':
    unittest.main()