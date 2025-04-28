import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a spherical grid
r = np.logspace(0, 3, 100)
theta = np.linspace(0, np.pi, 30)
phi = np.linspace(0, 2 * np.pi, 30)
R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')

# Convert to Cartesian coordinates
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)
print(X.shape, Y.shape, Z.shape)

# Define a synthetic density function
# density = np.exp(-R**2) * np.abs(np.sin(THETA)) * np.cos(PHI)**2
# density = np.random.rand(len(r), len(theta), len(phi))
density = np.random.random_integers(0, high=10000, size=X.shape)
print(density.shape)

# Plot as scatter plot (each point colored by density)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=np.log10(density.flatten()), cmap='plasma', alpha=0.6)

fig.colorbar(p, ax=ax, label='Density')
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_zscale("log")
ax.set_title('3D Spherical Density Plot (Scatter)')
plt.tight_layout()
plt.show()