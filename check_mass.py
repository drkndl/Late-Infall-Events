import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file
from analysis import calc_cell_volume, calc_mass, calc_surfdens
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import astropy.constants as c
import pandas as pd
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def surf_dens_profile(sigma0, p, r0, rc, rout, plot=True):
    """
    Calculating the radial surface density profile of the disk
    """

    cutoff_width_out = 0.05 * rout                                                 # As given in line 11 of condinit.c
    z0 = (rc - rout) / cutoff_width_out                                            # As given in line 475 of condinit.c
    sigma = sigma0 * np.power(rc / r0, -p) * 1. / (1. + np.exp(z0))                # See line 517 of condinit.c
    # sigma = sigma0 * np.power(rc / r0, -p)

    if plot:
        plt.plot(np.log10(rc[62]/au), np.log10(sigma[62]), label="Analytical")
        # plt.plot(np.log10(r / au), sigma_r)
        plt.xlabel(r"$\log(r)$")
        plt.ylabel(r"$\log(\Sigma(r))$")
        plt.title("Disk surface density profile")
        plt.legend()
        plt.show()

    return sigma 


def dens_profile(sigma, h0, r0, rc, f, zc, plot=True):
    """
    Calculating the radial mass density profile of the disk
    """

    Hc = h0 * rc * np.power(rc / r0, f)
    rho = sigma / (np.sqrt(2 * np.pi) * Hc) * np.exp(-zc**2 / (2 * Hc**2))

    if plot:
        plt.plot(np.log10(rc[62]/au), np.log10(rho[62]), label="Analytical")
        # plt.plot(np.log10(r/au), np.log10(rho_r))
        # plt.plot(np.log10(r / au), rho_r)
        plt.xlabel(r"$\log(r)$")
        plt.ylabel(r"$\log(\rho(r))$")
        plt.title("Disk mass density profile")
        plt.legend()
        plt.show()

    return rho


def sph_cell_area_2D(r, theta):
    """
    Calculating the 2D cell volume: Note: Formula WRONG (02.09)
    """

    # Finding dr, dphi, dtheta and making them 3D arrays
    dr = np.diff(r)
    dtheta = np.diff(theta)
    dR = dr[None, :]
    dTheta = dtheta[:, None]

    # We are finding the volume at the centre, so centering the cells
    r_c = 0.5 * (r[:-1] + r[1:])
    theta_c = 0.5 * (theta[:-1] + theta[1:])

    # Creating a meshgrid of the centered cells
    Theta_c, R_c = np.meshgrid(theta_c, r_c, indexing='ij')

    # Finding cell volumes
    cell_area = 2 * np.pi * R_c * np.sin(Theta_c) * dR * dTheta 

    return cell_area


###################################### Numerically integrating for the disk mass ##########################################


# Disk parameters from corresponding nocloud_nocomp par file
folder = Path("nocloud_nocomp_it10/")                                  # Folder with the output files
it = 10                                                                # FARGO snapshot of interest
sim_params = load_par_file(f"{folder}/{folder}.par")                   # Loading simulation parameters from the .par file

R0 = 5.2 * au                         # As defined in FARGO3D [cm]
Rin = 10. * au                        # Disk inner radius in cm (corresponds to Ymin in mesh parameters) 
Rout = 100. * au                      # Disk outer radius in cm (corresponds to Rout in disk parameters)
sigma0 = sim_params['Sigma0']         # Surface density at R0 in g/cm^2
p = sim_params['SigmaSlope']          # Negative surface density power law slope
f = sim_params['FlaringIndex']        # Flaring index
h0 = sim_params['AspectRatio']        # Aspect ratio 
theta_min = 0.17453292519943          # Theta lower limit (corresponds to Zmin in mesh params, 10 deg)
theta_max = 2.96705972839036          # Theta upper limit (corresponds to Zmax in mesh params, 170 deg)

theta = np.linspace(theta_min, theta_max, sim_params['Nz'])                           # Theta array
r = np.logspace(np.log10(Rin / au), np.log10(Rout / au), sim_params['Ny']) * au       # Radius array

# Centering the cells
r_c = 0.5 * (r[:-1] + r[1:])
theta_c = 0.5 * (theta[:-1] + theta[1:])

# Converting to cylindrical coordinates
THETA, R = np.meshgrid(theta_c, r_c, indexing="ij")
RCYL = R * np.sin(THETA)
ZCYL = R * np.cos(THETA)

# Calculating and plotting the surface density profile
sigma_r = surf_dens_profile(sigma0, p, R0, RCYL, Rout, plot=False)

# Calculating and plotting the density profile
rho_r = dens_profile(sigma_r, h0, R0, RCYL, f, ZCYL, plot=False)

# Calculating total disk mass
vol2D = sph_cell_area_2D(r, theta)
disk_mass_theoretical = np.sum(sigma_r * vol2D)


#################################### Adding up mass from the simulation ######################################


disk_folder = Path("../nocloud_nocomp_it10/")                      # Folder with the output files
disk_fig_imgs = Path("nocloud_nocomp_it10/imgs/")                  # Folder to save images
disk_it = 10                                                       # FARGO snapshot of interest

domains = get_domain_spherical(disk_folder)
disk_rho = get_data(disk_folder, "dens", disk_it, domains)         # Load 3D array of density values            
# THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
# X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
disk_mass = calc_mass(disk_rho, cell_volume)
disk_surf_dens = calc_surfdens(disk_rho, domains["theta"], domains["r"], domains["phi"])
disk_mass_simulation = np.sum(disk_mass)


################################################ Comparison plots ##############################################


# Surface density plots
fig, ax = plt.subplots()
ax.plot(np.log10(domains["r"]/au), np.log10(disk_surf_dens), label="Simulation")
ax.plot(np.log10(RCYL[62]/au), np.log10(sigma_r[62]), label="Analytical")
# ax.plot(np.log10(r / au), sigma_r, label="Analytical 2")
ax.set_xlabel(r"$\log(r)$")
ax.set_ylabel(r"$\log(\Sigma(r))$")
ax.set_title("Disk surface density profile")
ax.legend()

# Inset axis zooming in to the first 100 AU
inset_ax = inset_axes(ax, width="45%", height="45%", loc='upper right')
x1, x2 = np.min(np.log10(RCYL[62]/au)), np.log10(150)
inset_ax.set_xlim(x1, x2)
inset_ax.plot(np.log10(domains["r"]/au), np.log10(disk_surf_dens))
inset_ax.plot(np.log10(RCYL[62]/au), np.log10(sigma_r[62]))
# mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")
plt.savefig(f'{disk_fig_imgs}/checkmass_surfdens_it{it}.png')
plt.show()


# Mass density plots
fig, ax = plt.subplots()
ax.plot(np.log10(domains["r"]/au), np.log10(disk_rho[62, :, 0]), label="Simulation")
ax.plot(np.log10(RCYL[62]/au), np.log10(rho_r[62]), label="Analytical")
# ax.plot(np.log10(r/au), np.log10(rho_r), label)
# ax.plot(np.log10(r / au), rho_r)
ax.set_xlabel(r"$\log(r)$")
ax.set_ylabel(r"$\log(\rho(r))$")
ax.set_title("Disk mass density profile")
ax.legend()

# Inset axis zooming in to the first 100 AU
inset_ax = inset_axes(ax, width="45%", height="45%", loc='upper right')
x1, x2 = np.min(np.log10(RCYL[62]/au)), np.log10(150)
inset_ax.set_xlim(x1, x2)
inset_ax.plot(np.log10(domains["r"]/au), np.log10(disk_rho[62, :, 0]))
inset_ax.plot(np.log10(RCYL[62]/au), np.log10(rho_r[62]))
# mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")
plt.savefig(f'{disk_fig_imgs}/checkmass_massdens_it{it}.png')
plt.show()

print(f"Theoretical disk mass: {disk_mass_theoretical:.2e} g or {(disk_mass_theoretical / Msun):.3f} Msun")
print(f"Simulation disk mass: {disk_mass_simulation:.2e} g or {(disk_mass_simulation / Msun):.3f} Msun")


################################## Now calculating cloudlet mass from the simulation ###################################


# We are using the iras04125_lowres_it450_nocomp simulation for mass estimation
# Assumption: All of the cloudlet mass is accreted onto primary star at the end of the simulation

# Disk parameters from corresponding iras04125_lowres_it450_nocomp.par file
cloud_folder = Path("../iras04125_lowres_it450_nocomp/")                         # Folder with the output files
cloud_fig_imgs = Path("iras04125_lowres_it450_nocomp/imgs/")                     # Folder to save images
cloud_it = 10                                                                    # FARGO snapshot of interest
cloud_sim_name = str(cloud_fig_imgs).split('/')[0]                               # Simulation name (for plot labels)
cloud_sim_params = load_par_file(f"{cloud_sim_name}/{cloud_sim_name}.par")       # Loading simulation parameters from the .par file
# print(cloud_sim_params)

cloud_mass_theoretical = cloud_sim_params["CloudletMass"]

cloud_domains = get_domain_spherical(cloud_folder)
cloud_rho = get_data(cloud_folder, "dens", cloud_it, cloud_domains)         # Load 3D array of density values            
# THETA, R, PHI = np.meshgrid(cloud_domains["theta"], cloud_domains["r"], cloud_domains["phi"], indexing="ij")
# X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

cell_volume = calc_cell_volume(cloud_domains["theta"], cloud_domains["r"], cloud_domains["phi"])
cloud_mass = calc_mass(cloud_rho, cell_volume)
cloud_mass_simulation = np.sum(cloud_mass)
# cloud_surf_dens = calc_surfdens(cloud_rho, cloud_domains["theta"], cloud_domains["r"], cloud_domains["phi"])


print(f"Theoretical cloud mass: {cloud_mass_theoretical:.2e} g or {(cloud_mass_theoretical / Msun):.3f} Msun")
print(f"Simulation cloud mass: {cloud_mass_simulation:.2e} g or {(cloud_mass_simulation / Msun):.3f} Msun")




