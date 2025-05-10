import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
from no_thoughts_just_plots import interactive_2D, contours_3D, cyl_2D_plot, XY_2D_plot, quiver_plots
import astropy.constants as c
au = c.au.cgs.value


folder = Path("leon_snapshot/")         # Folder with the output files
it = 600                                # FARGO snapshot

############# theta = 100, r = 250, phi = 225 ###########
domains = get_domain_spherical(folder)

# Converting spherical coords to Cartesian and Cylindrical coords
THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
RCYL = R * np.sin(THETA)
ZCYL = R * np.cos(THETA)
#print(X.shape, Y.shape, RCYL.shape, ZCYL.shape)

rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values 
vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta
print(np.shape(rho), np.shape(vrad))

vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

# Cartesian velocities
vx = vrad * np.sin(THETA) * np.cos(PHI) + vthe * np.cos(THETA) * np.cos(PHI) - vphi * np.sin(PHI)
vy = vrad * np.sin(THETA) * np.sin(PHI) + vthe * np.cos(THETA) * np.sin(PHI) + vphi * np.cos(PHI)
vz = vrad * np.cos(THETA) - vthe * np.sin(THETA)
# print(np.max(domains["r"]/au), np.min(domains["r"]/au))
# print(np.max(np.abs(X)/au), np.min(np.abs(X)/au))
# print(np.max(np.abs(R)/au), np.min(np.abs(R)/au))
############## Plotting ###################

# Plot r-theta slice (flipping theta to match physics convention of spherical coords)
labels = [r'$\pi - \theta$ [deg]',r'$\log r$ [AU]',r'$\phi$ [deg]']
# interactive_2D(np.log10(rho[::-1,:,:]), (1,0), np.log10(domains['r'] / au), np.rad2deg(domains['theta']), labels)
# interactive_2D(np.log10(vsph[::-1,:,:]), (1,0), np.log10(domains['r'] / au), np.rad2deg(domains['theta']), labels)

# Plot r-phi slice (flipping theta to match physics convention of spherical coords)
# interactive_2D(np.log10(rho[::-1,:,:]), (1,2), np.log10(domains['r'] / au), np.rad2deg(domains['phi']), labels)
# interactive_2D(np.log10(vsph[::-1,:,:]), (1,2), np.log10(domains['r'] / au), np.rad2deg(domains['phi']), labels)

# Plot phi-theta slice (flipping theta to match physics convention of spherical coords)
# interactive_2D(np.log10(rho[::-1,:,:]), (2,0), np.rad2deg(domains['phi']), np.rad2deg(domains['theta']), labels)
# interactive_2D(np.log10(vsph[::-1,:,:]), (2,0), np.rad2deg(domains['phi']), np.rad2deg(domains['theta']), labels)
irad=-1
# print(np.shape(rho[:, :, 1]), np.shape(R[:, :, 1]/au), np.shape(THETA[:, :, 1]))

# for iphi in range(255):

#     iphi_deg = np.round(np.rad2deg(domains["phi"][iphi]), 2)
#     print(iphi_deg)
#     plt.figure()
#     plt.pcolormesh(np.log10(R[::-1, :, iphi]/au), THETA[:, :, iphi], np.log10(rho[::-1, :, iphi]), cmap="inferno", vmin=-19, vmax=-11, rasterized=True)
#     plt.gca().set_aspect("equal")
#     plt.xlabel("R / AU")
#     plt.ylabel(r'$\pi - \theta$ [deg]')
#     # plt.title()
#     plt.colorbar(label = r'$\log(\rho)$')
#     plt.tight_layout()
#     plt.savefig(f"gifs/phi_{iphi}.png")
    # plt.show()


# for irad in range(250):

#     fig, ax = plt.subplots()
#     c = ax.pcolormesh(PHI[:, irad, :], THETA[:, irad, :], np.log10(rho[::-1, irad, :]), cmap="inferno", vmin=-19, vmax=-11, rasterized=True)
#     ax.set_aspect((domains['phi'][-1] - domains['phi'][0]) / (domains['theta'][-1] - domains['theta'][0]))
#     plt.xlabel(r"$\phi$")
#     plt.ylabel(r'$\pi - \theta$')
#     # plt.title()
#     plt.colorbar(c, label = r'$\log(\rho)$')
#     plt.tight_layout()
#     plt.savefig(f"gifs2/rad_{irad}.png")
#     plt.close()


# Plot 3D contours up to a certain radial extent as defined by irad
# irad = np.where(domains["r"]/au < 300)[0][-1]
# fig = plt.figure(figsize=(10, 7))
# contours_3D(X /au, Y /au, ZCYL /au, np.log10(rho), fig, xlabel="X [AU]", ylabel="Y [AU]", zlabel="Z [AU]", colorbarlabel=r"$\log \rho (g/cm^3)$", title="Density contour")
# contours_3D(X[:, :irad, :]/au, Y[:, :irad, :]/au, ZCYL[:, :irad, :]/au, np.log10(rho[:, :irad, :]), fig, xlabel="X [AU]", ylabel="Y [AU]", zlabel="Z [AU]", colorbarlabel=r"$\log \rho (g/cm^3)$", title="Density contour")

itheta = 50
itheta_deg = np.round(np.rad2deg(domains["theta"][itheta]), 2)
iphi = 0
irad = -1
irad = np.where(domains["r"]/au < 1000)[0][-1]
print(irad)

# cyl_2D_plot(rho, RCYL, ZCYL, irad, iphi, title=rf'Density R-Z Plane $\phi = $ {np.round(domains["phi"][iphi], 2)}', colorbarlabel=r"$\rho (g/cm^{3})$", savefig=False, figfolder=folder / f"dens_cyl_phi{iphi}_rad{irad}.png")

# XY_2D_plot(rho, X, Y, irad, itheta, title=rf'Density X-Y Plane $\theta = $ {itheta_deg}', colorbarlabel=r"$\log(\rho)$", savefig=True, figfolder=folder / f"dens_xy_theta{itheta}_rad{irad}.png")

quiver_plots(X, Y, vx, vy, itheta, irad, title=rf'Velocity X-Y Plane $\theta = $ {itheta_deg}', savefig=False, figfolder=folder / f"vel_xy_theta{itheta}_rad{irad}.png")
