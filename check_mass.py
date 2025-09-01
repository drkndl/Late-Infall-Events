import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file
from analysis import calc_cell_volume, calc_mass, centering, sph_to_cart, vel_sph_to_cart, calc_simtime, calc_surfdens
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


###################################### Numerically integrating for the disk mass ##########################################


# Disk parameters from corresponding nocloud_nocomp par file
folder = Path("nocloud_nocomp/")                                  # Folder with the output files
it = 10                                                           # FARGO snapshot of interest
sim_params = load_par_file(f"{folder}/{folder}_it{it}.par")       # Loading simulation parameters from the .par file

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

# Converting to cylindrical coordinates
THETA, R = np.meshgrid(theta, r, indexing="ij")
RCYL = R * np.sin(THETA)
ZCYL = R * np.cos(THETA)

# Calculating and plotting the surface density profile
sigma_r = surf_dens_profile(sigma0, p, R0, RCYL, Rout, plot=False)

# Calculating and plotting the density profile
rho_r = dens_profile(sigma_r, h0, R0, RCYL, f, ZCYL, plot=False)


#################################### Adding up mass from the simulation ######################################


sim_folder = Path("../nocloud_nocomp_it10/")         # Folder with the output files
fig_imgs = Path("nocloud_nocomp_it10/imgs/")         # Folder to save images
it = 10                                                       # FARGO snapshot of interest
sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)

domains = get_domain_spherical(sim_folder)
rho = get_data(sim_folder, "dens", it, domains)         # Load 3D array of density values            
vphi = get_data(sim_folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
vrad = get_data(sim_folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
vthe = get_data(sim_folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

# THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
# X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
mass = calc_mass(rho, cell_volume)
surf_dens = calc_surfdens(rho, domains["theta"], domains["r"], domains["phi"])


################################################ Comparison plots ##############################################


# Surface density plots
plt.plot(np.log10(domains["r"]/au), np.log10(surf_dens), label="Simulation")
plt.plot(np.log10(RCYL[62]/au), np.log10(sigma_r[62]), label="Analytical")
# plt.plot(np.log10(r / au), sigma_r, label="Analytical 2")
plt.xlabel(r"$\log(r)$")
plt.ylabel(r"$\log(\Sigma(r))$")
plt.title("Disk surface density profile")
plt.legend()
plt.savefig(f'{fig_imgs}/checkmass_surfdens_it{it}.png')
plt.show()

# Mass density plots
print(rho.shape, RCYL.shape)
# wednwe
plt.plot(np.log10(domains["r"]/au), np.log10(rho[62, :, 0]), label="Simulation")
plt.plot(np.log10(RCYL[62]/au), np.log10(rho_r[62]), label="Analytical")
# plt.plot(np.log10(r/au), np.log10(rho_r), label)
# plt.plot(np.log10(r / au), rho_r)
plt.xlabel(r"$\log(r)$")
plt.ylabel(r"$\log(\rho(r))$")
plt.title("Disk mass density profile")
plt.legend()
plt.savefig(f'{fig_imgs}/checkmass_massdens_it{it}.png')
plt.show()

