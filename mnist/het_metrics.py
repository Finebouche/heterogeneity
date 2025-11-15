import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define the input domain (e.g., x values)
x_domain = np.linspace(-5, 5, 100)  # Adjust as needed

# Define parameter ranges for the two functions
# Function 1 parameters (theta)
theta_min, theta_max = 0, 2 * np.pi
# Function 2 parameters (phi)
phi_min, phi_max = -1, 1

# Number of samples from each parameter space
num_samples = 500  # Increase for better approximation

# Sample parameters uniformly from their ranges
theta_samples = np.random.uniform(theta_min, theta_max, num_samples)
phi_samples = np.random.uniform(phi_min, phi_max, num_samples)

# Define the parametrizable functions
def f_theta(theta, x):
    # Example: Sine function with varying frequency
    return np.sin(theta * x)

def g_phi(phi, x):
    # Example: Linear function with varying slope
    return phi * x

# Evaluate the functions over the input domain
f_values = np.array([f_theta(theta, x_domain) for theta in theta_samples])
g_values = np.array([g_phi(phi, x_domain) for phi in phi_samples])

# Reshape the function outputs to vectors for distance computation
f_vectors = f_values.reshape(num_samples, -1)
g_vectors = g_values.reshape(num_samples, -1)

# Compute pairwise distances between functions
distance_matrix = cdist(f_vectors, g_vectors, metric='euclidean')

# Define a threshold to consider functions as overlapping
overlap_threshold = 5.0  # Adjust based on acceptable similarity

# Identify overlapping function pairs
overlap_indices = np.where(distance_matrix < overlap_threshold)
num_overlaps = len(overlap_indices[0])

# Estimate overlapping volume
total_pairs = num_samples ** 2
overlap_volume = num_overlaps / total_pairs

print(f"Estimated Overlapping Volume: {overlap_volume}")

# Optional: Visualize some overlapping function pairs
import random

# Find indices of overlapping functions
f_overlap_indices = overlap_indices[0]
g_overlap_indices = overlap_indices[1]

# Select a few overlapping pairs to visualize
num_visualizations = 3
selected_indices = random.sample(range(num_overlaps), num_visualizations)

for idx in selected_indices:
    f_idx = f_overlap_indices[idx]
    g_idx = g_overlap_indices[idx]
    plt.figure(figsize=(10, 4))
    plt.plot(x_domain, f_values[f_idx], label=f'f_theta (theta={theta_samples[f_idx]:.2f})')
    plt.plot(x_domain, g_values[g_idx], label=f'g_phi (phi={phi_samples[g_idx]:.2f})')
    plt.legend()
    plt.title('Overlapping Functions')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.grid(True)
    plt.show()