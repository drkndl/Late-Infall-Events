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


##############   3D PLOTTING  #######################

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plotting with Cartesian equivalent of r_bincen
    # # ax.quiver(xplot/au, yplot/au, zplot/au, Lx_avg, Ly_avg, Lz_avg, length=5, normalize=True, color='red')

    # # Radially plotting (i.e. using r_bincen) averaged angular momenta
    # y_steps = np.linspace(0, yplot.max()/au, num_bins)
    # ax.quiver(r_bincen/au, 0, 0, Lx_avg, Ly_avg, Lz_avg, length=5, normalize=True, edgecolor='black', color='black')

    # # Overplotting the density as well
    # p = ax.scatter(X.flatten()/au, Y.flatten()/au, Z.flatten()/au, c=dens.flatten(), cmap='plasma', s=7, edgecolor='none', alpha=0.1)

    # # Colorbar formatting
    # plt.colorbar(p, pad=0.08, label=r'$\rho [g/cm^3]$') #, shrink=0.85), fraction=0.046)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # ax.set_ylim(yplot.min()/au, yplot.max()/au)
    # # ax.set_zlim(zplot.min()/au, zplot.max()/au)
    # ax.set_title('Radially Averaged Warp L')
    # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    # plt.tight_layout()
    # if savefig==True:
    #     plt.savefig(f"../twist_{it}_3D.png")
    # plt.show()

    ################################### 2D PROJECTION PLOTTING ############################

    # plt.figure(figsize=(8, 6))

    # # XZ projection
    # Q1 = plt.quiver(r_bincen/au, 0, Lx_avg, Lz_avg, angles_deg - np.nanmin(angles_deg), cmap='viridis')
    # plt.colorbar(Q1, label=r'$\hat{L} - \hat{L_{min}}$')
    # plt.xlabel('X')
    # plt.xlim(right = np.max(r_bincen)/au + 5)
    # plt.ylabel('Z')
    # plt.title('Radially Averaged L in XZ Plane')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # if savefig==True:
    #     plt.savefig(f"../twist_{it}_XZ_2D.png")
    # plt.show()

    # plt.figure(figsize=(8, 6))

    # # YZ projection
    # Q2 = plt.quiver(np.zeros(shape=Ly_avg.shape), np.zeros(shape=Lz_avg.shape), Ly_avg, Lz_avg, angles_deg - np.nanmin(angles_deg), cmap='viridis')
    # plt.colorbar(Q2, label=r'$\hat{L} - \hat{L_{min}}$')
    # plt.xlabel('Y [AU]')
    # plt.ylabel('Z [AU]')
    # plt.title('Radially Averaged L in YZ Plane')
    # plt.xlim(-0.02, 0.02)
    # plt.ylim(zplot.min()/au, zplot.max()/au)
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # if savefig==True:
    #     plt.savefig(f"../twist_{it}_YZ_2D.png")
    # plt.show()

    # # XY projection
    # plt.figure(figsize=(8, 6))
    # Q3 = plt.quiver(r_bincen/au, np.zeros(shape=Lz_avg.shape), Lx_avg, Ly_avg, angles_deg - np.nanmin(angles_deg), cmap='viridis')
    # plt.colorbar(Q3, label=r'$\hat{L} - \hat{L_{min}}$')
    # plt.xlabel('X [AU]')
    # plt.ylabel('Y [AU]')
    # plt.title('Radially Averaged L in XY Plane')
    # print(r_bincen.min(), r_bincen.max())
    # plt.xlim(r_bincen.min(), r_bincen.max() + 10*au)
    # plt.ylim(yplot.min()/au, yplot.max()/au)
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # if savefig==True:
    #     plt.savefig(f"../twist_{it}_XY_2D.png")
    # plt.show()


# Roughly the Cartesian central coordinate of the secondary disk obtained through visualization
    # secx = -150       # AU
    # secy = 200        # AU
    # secz = 700        # AU
    # buffer = 400      # AU

    # # Isolating the secondary box and its corresponding density
    # X_secbox, Y_secbox, Z_secbox, rho_c_secbox, secbox_ids = isolate_secondary_box(X_c, Y_c, Z_c, secx * au, secy * au, secz * au, buffer * au, rho_c)

    # # Plotting the secondary densities box 
    # contours_3D(X_secbox/au, Y_secbox/au, Z_secbox/au, rho_c_secbox, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\rho$ Secondary Disk Box', savefig=True, figfolder=f'../secondary_dens_box.png')

    # # Now applying the density threshold to the secondary box to get the secondary disk itself
    # sec_thresh = -15.5   # log of density threshold for which we can see the secondary
    # rho_c_sec, sec_ids = isolate_warp(rho_c_secbox, sec_thresh)

    # # Plotting the secondary densities
    # contours_3D(X_secbox/au, Y_secbox/au, Z_secbox/au, rho_c_sec, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\rho$ Secondary Disk', savefig=False, figfolder=f'../secondary_dens_thresh{sec_thresh}.png')

    # # Plotting the Cartesian secondary disk angular momenta
    # Lx_secbox, Ly_secbox, Lz_secbox = Lx[secbox_ids], Ly[secbox_ids], Lz[secbox_ids]
    # quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, Lx_secbox[sec_ids], Ly_secbox[sec_ids], Lz_secbox[sec_ids], stagger=1, length=10, title="Secondary disk angular momenta", colorbarlabel="logL", savefig=False, figfolder=f'../secondary_L_thresh{sec_thresh}.png', logmag=False)

    # Plotting secondary disk Laplace-Runge-Lenz vector
    # Ax_secbox, Ay_secbox, Az_secbox = Ax[secbox_ids], Ay[secbox_ids], Az[secbox_ids]
    # quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, Ax_secbox[sec_ids], Ay_secbox[sec_ids], Az_secbox[sec_ids], stagger=1, length=10, title=rf'Secondary disk LRL', colorbarlabel=r'$\log(A [g^2cm^3/s^2])$', savefig=False, figfolder=f'../secondary_LRL_thresh{sec_thresh}.png', logmag=False)

    # Plotting secondary disk eccentricity
    # ex_secbox, ey_secbox, ez_secbox = ex[secbox_ids], ey[secbox_ids], ez[secbox_ids]
    # quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, ex_secbox[sec_ids], ey_secbox[sec_ids], ez_secbox[sec_ids], stagger=1, length=10, title=rf'Secondary disk Eccentricity', colorbarlabel=r'$\log(e)$', savefig=False, figfolder=f'../secondary_ecc_thresh{sec_thresh}.png', logmag=False)

    # Characterizing secondary disk eccentricity 
    # e_secbox = e[secbox_ids]
    # print("Min Warp eccentricity: ", np.min(e_secbox[sec_ids]))
    # print("Max Warp eccentricity: ", np.max(e_secbox[sec_ids]))
    # print("Mean Warp eccentricity: ", np.mean(e_secbox[sec_ids]))