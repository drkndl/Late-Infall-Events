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

# For radial L average and plotting 

######################### 2D PLOTTING ########################

    # # Define radial bins
    # num_bins = 20
    # r_bins = np.linspace(R.min(), R.max(), num_bins + 1)
    # r_bincen = 0.5 * (r_bins[:-1] + r_bins[1:])

    # # Flatten arrays
    # R_flat = R.flatten()
    # Lx_flat = Lx.flatten()
    # Ly_flat = Ly.flatten()
    # Lz_flat = Lz.flatten()
    
    # # Bin indices
    # bin_indices = np.digitize(R_flat, r_bins)
    
    # # Initialize arrays
    # Lx_avg = np.zeros(num_bins)
    # Ly_avg = np.zeros(num_bins)
    # Lz_avg = np.zeros(num_bins)

    # for i in range(1, num_bins + 1):
    #     mask = bin_indices == i
    #     if np.any(mask):
    #         print(Lx_flat[mask].shape)
    #         Lx_avg[i-1] = Lx_flat[mask].mean()
    #         Ly_avg[i-1] = Ly_flat[mask].mean()
    #         Lz_avg[i-1] = Lz_flat[mask].mean()

    # # Radial directions in XY plane
    # theta = np.linspace(0, 2 * np.pi, num_bins)
    # x_points = r_bincen * np.cos(theta)
    # y_points = r_bincen * np.sin(theta)
    # print(x_points.shape, y_points.shape)

    # # Interpolate average vector directions to angle theta
    # # (Just for illustration; a more advanced method would project the full vector field)
    # Lx_interp = np.interp(theta, np.arctan2(Ly_avg, Lx_avg), Lx_avg, period=2*np.pi)
    # Ly_interp = np.interp(theta, np.arctan2(Ly_avg, Lx_avg), Ly_avg, period=2*np.pi)

    # # Normalize vectors for display
    # norm = np.sqrt(Lx_interp**2 + Ly_interp**2)
    # Lx_plot = Lx_interp / norm
    # Ly_plot = Ly_interp / norm

    # # Plot using quiver
    # plt.figure(figsize=(6, 6))
    # plt.quiver(x_points/au, y_points/au, Lx_plot, Ly_plot, scale=10, pivot='tip')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Radially Averaged Angular Momenta in XY Plane')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.show()