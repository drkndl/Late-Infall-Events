import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
from analysis import sph_to_cart, vel_sph_to_cart, calc_cell_volume, calc_mass
from no_thoughts_just_plots import interactive_2D, contours_3D, cyl_2D_plot, XY_2D_plot, quiver_plots, interactive_interp_3d
import astropy.constants as c
au = c.au.cgs.value


folder = Path("../cloud_nodisk_it450_rotX45/")         # Folder with the output files
sim_name = str(folder).split('/')[1]                 # Simulation name (for plot labelling)
it = 450                                             # FARGO snapshot

############# theta = 175, r = 150, phi = 100 ###########
domains = get_domain_spherical(folder)

# 3D meshgrid of Cartesian and cylindrical coordinates from the given spherical coordinates
THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)
# print(np.max(X /au), np.min(X/au))

rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values 
vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

# Cartesian velocities
vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
mass = calc_mass(rho, cell_volume)

# Load density values at multiple iterations 
rho_allit = []
mass_allit = []

for i in range(0, it+1, 10):     # loading density every 10 iterations
    rho_i = get_data(folder, "dens", i, domains)
    mass_i = calc_mass(rho_i, cell_volume)
    rho_allit.append(rho_i)
    mass_allit.append(mass_i)

rho_allit = np.asarray(rho_allit)
mass_allit = np.asarray(mass_allit)

###################################################### Plotting #######################################################


labels = [r'$\pi - \theta$ [deg]',r'$\log r$ [AU]',r'$\phi$ [deg]']
labels_allit = [r"Time"] + labels

iphi_deg9 = np.round(np.rad2deg(domains["phi"][9]), 2)
print(iphi_deg9)

iphi_deg22 = np.round(np.rad2deg(domains["phi"][22]), 2)
print(iphi_deg22)

# rebfeknfek


################## Plot r-theta slice (flipping theta to match physics convention of spherical coords)

# rho plot at 1 iteration
# interactive_2D(np.log10(rho[::-1,:,:]), [r'$\phi$ [deg]'], (1,0), np.log10(domains['r'] / au), np.rad2deg(domains['theta']), labels, title=rf"{sim_name}: $\rho$ $(r, \theta)$ it={it}")

# mass plot at 1 iteration (mass shape is (ntheta-1, nr-1, nphi-1))
# interactive_2D(np.log10(mass[::-1,:,:]), [r'$\phi$ [deg]'], (1,0), np.log10(domains['r'] / au)[:-1], np.rad2deg(domains['theta'])[:-1], labels, vmin=20, title=rf"{sim_name}: $M$ $(r, \theta)$ it={it}")

# rho plot at all iterations
interactive_2D(np.log10(rho_allit[:,::-1,:,:]), [r"Time", r'$\phi$ [deg]'], (2,1), np.log10(domains['r'] / au), np.rad2deg(domains['theta']), labels_allit, title=rf"{sim_name}: $\rho$ $(r, \theta)$ Time Evolution")

# mass plot at all iterations
interactive_2D(np.log10(mass_allit[:,::-1,:,:]), [r"Time", r'$\phi$ [deg]'], (2,1), np.log10(domains['r'] / au)[:-1], np.rad2deg(domains['theta'])[:-1], labels_allit, vmin=20, title=rf"{sim_name}: $M$ $(r, \theta)$ Time Evolution")

# velocity plot at 1 iteration
# interactive_2D(np.log10(vsph[::-1,:,:]), (1,0), np.log10(domains['r'] / au), np.rad2deg(domains['theta']), labels)

################### Plot r-phi slice (flipping theta to match physics convention of spherical coords)

# rho plot at 1 iteration
# interactive_2D(np.log10(rho[::-1,:,:]), [r'$\pi - \theta$ [deg]'], (1,2), np.log10(domains['r'] / au), np.rad2deg(domains['phi']), labels, title=rf"{sim_name}: $\rho$ $(r, \phi)$ it={it}")

# # mass plot at 1 iteration (mass shape is (ntheta-1, nr-1, nphi-1))
# interactive_2D(np.log10(mass[::-1,:,:]), [r'$\pi - \theta$ [deg]'], (1,2), np.log10(domains['r'] / au)[:-1], np.rad2deg(domains['phi'])[:-1], labels, vmin=20, title=rf"{sim_name}: $M$ $(r, \phi)$ it={it}")

# # rho plot at all iterations
# interactive_2D(np.log10(rho_allit[:,::-1,:,:]), [r"Time", r'$\pi - \theta$ [deg]'], (2,3), np.log10(domains['r'] / au), np.rad2deg(domains['phi']), labels_allit, title=rf"{sim_name}: $\rho$ $(r, \phi)$ Time Evolution")

# # mass plot at all iterations
# interactive_2D(np.log10(mass_allit[:,::-1,:,:]), [r"Time", r'$\pi - \theta$ [deg]'], (2,3), np.log10(domains['r'] / au)[:-1], np.rad2deg(domains['phi'])[:-1], labels_allit, vmin=20, title=rf"{sim_name}: $\rho$ $(r, \phi)$ Time Evolution")

# velocity plot at 1 iteration
# interactive_2D(np.log10(vsph[::-1,:,:]), (1,2), np.log10(domains['r'] / au), np.rad2deg(domains['phi']), labels)

#################### Plot phi-theta slice (flipping theta to match physics convention of spherical coords)

# rho plot at 1 iteration
# interactive_2D(np.log10(rho[::-1,:,:]), [r'$\log r$ [AU]'], (2,0), np.rad2deg(domains['phi']), np.rad2deg(domains['theta']), labels, title=rf"{sim_name}: $\rho$ $(\theta, \phi)$ it={it}")

# mass plot at 1 iteration (mass shape is (ntheta-1, nr-1, nphi-1))
# interactive_2D(np.log10(mass[::-1,:,:]), [r'$\log r$ [AU]'], (2,0), np.log10(domains['phi'])[:-1], np.rad2deg(domains['theta'])[:-1], labels, vmin=15, title=rf"{sim_name}: $M$ $(\theta, \phi)$ it={it}")

# rho plot at all iterations
# interactive_2D(np.log10(rho_allit[:,::-1,:,:]), [r"Time", r'$\log r$ [AU]'], (3,1), np.log10(domains['phi']), np.rad2deg(domains['theta']), labels_allit, title=rf"{sim_name}: $\rho$ $(\theta, \phi)$ Time Evolution")

# mass plot at all iterations
# interactive_2D(np.log10(mass_allit[:,::-1,:,:]), [r"Time", r'$\log r$ [AU]'], (3,1), np.log10(domains['phi'])[:-1], np.rad2deg(domains['theta'])[:-1], labels_allit, vmin=20, title=rf"{sim_name}: $\rho$ $(\theta, \phi)$ Time Evolution")

# velocity plot at 1 iteration
# interactive_2D(np.log10(vsph[::-1,:,:]), (2,0), np.rad2deg(domains['phi']), np.rad2deg(domains['theta']), labels)

wekfjwkf
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
irad = np.where(domains["r"]/au < 500)[0][-1]
# irad = -1
# fig = plt.figure(figsize=(10, 7))
# contours_3D(X /au, Y /au, ZCYL /au, np.log10(rho), fig, xlabel="X [AU]", ylabel="Y [AU]", zlabel="Z [AU]", colorbarlabel=r"$\log \rho (g/cm^3)$", title="Density contour")
# contours_3D(X[:, :irad, :]/au, Y[:, :irad, :]/au, ZCYL[:, :irad, :]/au, np.log10(rho[:, :irad, :]), xlabel="X [AU]", ylabel="Y [AU]", zlabel="Z [AU]", colorbarlabel=r"$\log \rho (g/cm^3)$", title="Density contour", savefig=False, figfolder="")

itheta = 62
itheta_deg = np.round(np.rad2deg(domains["theta"][itheta]), 2)
iphi = 0
irad = -1
irad = np.where(domains["r"]/au < 2000)[0][-1]
# print(irad)

cyl_2D_plot(rho, RCYL, ZCYL, irad, iphi, title=rf'Density R-Z Plane $\phi = $ {np.round(domains["phi"][iphi], 2)}', colorbarlabel=r"$\rho (g/cm^{3})$", savefig=True, figfolder=folder / f"dens_cyl_phi{iphi}_rad{irad}.png", showfig=True)

XY_2D_plot(rho, X, Y, irad, itheta, title=rf'Density X-Z Plane $\theta = $ {itheta_deg}', colorbarlabel=r"$\log(\rho)$", savefig=True, figfolder=folder / f"dens_xz_theta{itheta}_rad{irad}.png", showfig=True)

# quiver_plots(X, Y, vx, vy, itheta, irad, title=rf'Velocity X-Y Plane $\theta = $ {itheta_deg}', savefig=True, figfolder=f"../vel_xy_theta{itheta}_rad{irad}.png")

# Rmax = 50       # Maximum radius of the Cartesian box for interactive_interp_3d in AU
# interactive_interp_3d(np.log10(rho), Rmax, colorbarlabel=r"$\log \rho (g/cm^3)$", title="Density", idxnames=['X [au]', 'Y [au]', 'Z [au]'])
# interactive_interp_3d(np.log10(vsph), Rmax, colorbarlabel=r"$\log v (cm/s)$", title="Velocity", idxnames=['X [au]', 'Y [au]', 'Z [au]'])