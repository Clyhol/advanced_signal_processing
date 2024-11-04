import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from tqdm import tqdm

def generate_process(n, h, sigma_Z, sigma_W):
    """Generate the true process Y and observations X."""
    # Initialize arrays
    Z = np.random.normal(0, sigma_Z, n)  # sigma_Z is already variance
    W = np.random.normal(0, sigma_W, n)  # sigma_W is already variance
    Y = np.zeros(n)
    
    # Generate Y(0) ~ N(0, sigma_Y^2)
    sigma_Y = sigma_Z / (1 - h**2)
    Y[0] = np.random.normal(0, sigma_Y)
    
    # Generate AR(1) process
    for i in range(1, n):
        Y[i] = h * Y[i-1] + Z[i]
    
    # Generate observations
    X = Y + W
    
    return Y, X

def compute_theoretical_mse(h, sigma_Z, sigma_W, n):
    """Compute the theoretical MSE for the LMMSE estimator."""
    sigma_Y = sigma_Z / (1 - h**2)
    
    # Compute auto-covariance of Y
    R_Y = np.zeros(n)
    for i in range(n):
        R_Y[i] = sigma_Y * h**i
    
    # Create covariance matrices
    R_YY = toeplitz(R_Y)
    R_XX = R_YY + sigma_W * np.eye(n)
    R_YX = R_YY
    
    # Compute theoretical MSE
    mse = np.trace(R_YY - R_YX @ np.linalg.inv(R_XX) @ R_YX.T) / n
    return mse

def lmmse_estimate(X, h, sigma_Z, sigma_W):
    """Compute LMMSE estimate of Y given observations X."""
    n = len(X)
    sigma_Y = sigma_Z / (1 - h**2)
    
    # Compute auto-covariance of Y
    R_Y = np.zeros(n)
    for i in range(n):
        R_Y[i] = sigma_Y * h**i
    
    # Create covariance matrices
    R_YY = toeplitz(R_Y)
    R_XX = R_YY + sigma_W * np.eye(n)
    R_YX = R_YY
    
    # Compute LMMSE estimate
    Y_est = R_YX @ np.linalg.inv(R_XX) @ X
    
    return Y_est

# Set parameters
h = 0.95
sigma_W = 1  # Already variance (was 1.0)
sigma_Z = 0.1  # Already variance
n_max = 1000
num_realizations = 1

# Initialize arrays for MSE
empirical_mse = np.zeros(n_max)
theoretical_mse = np.zeros(n_max)

# Compute MSE for different n
print("Computing MSE for different n values...")
for n in tqdm(range(1, n_max + 1), desc='Running LMMSE simulation'):
    mse_sum = 0
    
    # Monte Carlo trials
    for _ in range(num_realizations):
        Y, X = generate_process(n, h, sigma_Z, sigma_W)
        Y_est = lmmse_estimate(X, h, sigma_Z, sigma_W)
        mse_sum += np.mean((Y - Y_est)**2)
    
    empirical_mse[n-1] = mse_sum / num_realizations
    theoretical_mse[n-1] = compute_theoretical_mse(h, sigma_Z, sigma_W, n)

# Plot results
plt.figure(figsize=(12, 7))
plt.plot(range(1, n_max + 1), empirical_mse, 'b-', label='Simulated MSE', alpha=0.7)
plt.plot(range(1, n_max + 1), theoretical_mse, 'r--', label='Theoretical MSE')
plt.xlabel('Number of samples (n)')
plt.ylabel('Mean Square Error')
plt.title('LMMSE Performance with No Observation Noise (σ_W = 0)')
plt.legend()
plt.grid(True)

# Add a zoomed inset for the last 100 samples
ax_inset = plt.axes([0.55, 0.55, 0.3, 0.3])
ax_inset.plot(range(900, 1000), empirical_mse[899:999], 'b-', label='Simulated', alpha=0.7)
ax_inset.plot(range(900, 1000), theoretical_mse[899:999], 'r--', label='Theoretical')
ax_inset.set_title('Zoom: n=900 to n=1000')
ax_inset.grid(True)

plt.show()

# Print some example values
print("\nExample MSE values:")
for n in [1, 100, 500, 1000]:
    print(f"n={n:4d}: Theoretical MSE={theoretical_mse[n-1]:.6f}, "
          f"Simulated MSE={empirical_mse[n-1]:.6f}")