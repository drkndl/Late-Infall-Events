# Post-processing on FARGO3D output files 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
from no_thoughts_just_plots import quiver_plot_3d, contours_3D, plot_surf_dens, plot_twist_arrows, plot_total_disks_bonanza, cyl_2D_plot, XY_2D_plot, velocity_streamlines, plotly_quiver3D, vel_cyl_2D, plot_disk_sep, interactive_2D
import astropy.constants as c
import pandas as pd
# import pyvista as pv 

# pv.global_theme.allow_empty_mesh = True
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def sph_to_cart(THETA, R, PHI):
    """
    Creates a meshgrid of Cartesian and/or cylindrical coordinates given spherical coordinates 

    Inputs:
    ------
    THETA:   3D meshgrid of theta values with shape (n_theta, n_r, n_phi)
    R:       3D meshgrid of r values with shape (n_theta, n_r, n_phi)
    PHI:     3D meshgrid of phi values with shape (n_theta, n_r, n_phi)

    Outputs:
    -------
    X:       3D meshgrid of Cartesian X values with shape (n_theta, n_r, n_phi)
    Y:       3D meshgrid of Cartesian Y values with shape (n_theta, n_r, n_phi)
    ZCYL:    3D meshgrid of Cartesian (or Cylindrical) Z values with shape (n_theta, n_r, n_phi)
    RCYL:    3D meshgrid of Cylindrical R values with shape (n_theta, n_r, n_phi)
    """

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    RCYL = R * np.sin(THETA)
    ZCYL = R * np.cos(THETA)

    return X, Y, ZCYL, RCYL 


def cart_to_sph(xx, yy, zz):
    """
    Convert data in Cartesian coordinates to spherical coordinates. This function is adapted from test_interp_3d.py by Kees Dullemond

    Inputs:
    ------
    xx:   3D meshgrid of Cartesian X values with shape (n_theta, n_r, n_phi)
    yy:   3D meshgrid of Cartesian Y values with shape (n_theta, n_r, n_phi)
    zz:   3D meshgrid of Cartesian Z values with shape (n_theta, n_r, n_phi)

    Outputs:
    -------
    r_box:       3D meshgrid of spherical R values with shape (n_theta, n_r, n_phi)
    rc_box:      3D meshgrid of Cylindrical R values with shape (n_theta, n_r, n_phi)
    theta_box:   3D meshgrid of spherical theta values with shape (n_theta, n_r, n_phi)
    phi_box:     3D meshgrid of spherical phi values with shape (n_theta, n_r, n_phi)
    """

    r_box     = np.sqrt(xx**2 + yy**2 + zz**2)
    rc_box    = np.sqrt(xx**2 + yy**2)
    theta_box = np.pi/2-np.arctan(zz/(rc_box+1e-99))
    phi_box   = np.arctan2(yy,xx)
    phi_box[phi_box<0]+=np.pi*2

    return r_box, rc_box, theta_box, phi_box


def vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI):
    """
    Converts velocities in spherical coordinates to Cartesian coordinates

    Inputs:
    ------
    vthe:    3D meshgrid of polar velocities with shape (n_theta, n_r, n_phi)
    vrad:    3D meshgrid of azimuthal velocities with shape (n_theta, n_r, n_phi)
    vphi:    3D meshgrid of radial velocities with shape (n_theta, n_r, n_phi)
    THETA:   3D meshgrid of theta values with shape (n_theta, n_r, n_phi)
    PHI:     3D meshgrid of phi values with shape (n_theta, n_r, n_phi)

    Outputs:
    -------
    vx:    3D meshgrid of Cartesian X velocities with shape (n_theta, n_r, n_phi)
    vy:    3D meshgrid of Cartesian Y velocities with shape (n_theta, n_r, n_phi)
    vz:    3D meshgrid of Cartesian Z velocities with shape (n_theta, n_r, n_phi)
    """

    vx = vrad * np.sin(THETA) * np.cos(PHI) + vthe * np.cos(THETA) * np.cos(PHI) - vphi * np.sin(PHI)
    vy = vrad * np.sin(THETA) * np.sin(PHI) + vthe * np.cos(THETA) * np.sin(PHI) + vphi * np.cos(PHI)
    vz = vrad * np.cos(THETA) - vthe * np.sin(THETA)

    return vx, vy, vz


def centering(data):
    """
    Center the data from cell walls to cell centres

    Inputs:
    ------
    data:      3D array with shape (n_theta, n_r, n_phi)

    Outputs:
    -------
    data_c:    Centered 3D array with shape (n_theta-1, n_r-1, n_phi-1)
    """

    data_c = 0.5 * (data[:-1, :, :] + data[1:, :, :])
    data_c = 0.5 * (data_c[:, :-1, :] + data_c[:, 1:, :])
    data_c = 0.5 * (data_c[:, :, :-1] + data_c[:, :, 1:])

    return data_c 



def calc_cell_volume(theta, r, phi):
    """
    Calculates the cell volumes of an (theta,r,phi) grid in cm^3 using the formula for an elemental volume in 3D spherical coordinates dV = r^2 sin(theta) dr dphi dtheta
    
    Inputs:
    ------
    theta:    1D array of polar angles
    r:        1D array of radii
    phi:      1D array of azimuthal angles

    Outputs:
    -------
    cell_vol: 3D array of cell volumes with shape (n_theta-1, n_r-1, n_phi-1)
    """

    # Finding dr, dphi, dtheta and making them 3D arrays
    dr = np.diff(r)
    dtheta = np.diff(theta)
    dphi = np.diff(phi)
    dR = dr[None, :, None]
    dTheta = dtheta[:, None, None]
    dPhi = dphi[None, None, :]

    # We are finding the volume at the centre, so centering the cells
    r_c = 0.5 * (r[:-1] + r[1:])
    theta_c = 0.5 * (theta[:-1] + theta[1:])
    phi_c = 0.5 * (phi[:-1] + phi[1:])

    # Creating a meshgrid of the centered cells
    Theta_c, R_c, Phi_c = np.meshgrid(theta_c, r_c, phi_c, indexing='ij')

    # Finding cell volumes
    cell_vol = (R_c**2) * np.sin(Theta_c) * dR * dTheta * dPhi
    
    return cell_vol 


def calc_mass(rho, cell_vol):
    """
    Calculates the mass of each cell in g as m = density x volume

    Inputs:
    -------
    rho:      3D array of density in g/cm^3 with shape (n_theta, n_r, n_phi)
    cell_vol: 3D array of cell volumes in cm^3 with shape (n_theta-1, n_r-1, n_phi-1)

    Outputs:
    --------
    mass:     3D array of mass in g with shape (n_theta-1, n_r-1, n_phi-1)
    """

    # Centering the rho grid so that we take the density at the centre of the cells
    rho_c = centering(rho)
    mass = rho_c * cell_vol
    return mass


def calc_angular_momentum(mass, x, y, z, vx, vy, vz):
    """
    Calculates the angular momentum across each cell in (g cm^2/s) as L = m(r x v)

    Inputs:
    -------
    mass:     3D array of mass in g with shape (n_theta-1, n_r-1, n_phi-1)
    x:        3D meshgrid of Cartesian X values with shape (n_theta, n_r, n_phi)
    y:        3D meshgrid of Cartesian Y values with shape (n_theta, n_r, n_phi)
    z:        3D meshgrid of Cartesian Z values with shape (n_theta, n_r, n_phi)
    vx:       3D meshgrid of Cartesian X velocities with shape (n_theta, n_r, n_phi)
    vy:       3D meshgrid of Cartesian Y velocities with shape (n_theta, n_r, n_phi)
    vz:       3D meshgrid of Cartesian Z velocities with shape (n_theta, n_r, n_phi)

    Outputs:
    -------
    Lx:       3D array of x-component of angular momentum with shape (n_theta-1, n_r-1, n_phi-1)
    Ly:       3D array of y-component of angular momentum with shape (n_theta-1, n_r-1, n_phi-1)
    Lz:       3D array of z-component of angular momentum with shape (n_theta-1, n_r-1, n_phi-1)
    """

    # Calculating r x v
    r_vec = np.stack((x, y, z), axis=-1)
    v_vec = np.stack((vx, vy, vz), axis=-1)
    L_vec = np.cross(r_vec, v_vec, axis=-1)

    # Finding x, y, z angular momentum components
    Lx = L_vec[:, :, :, 0]
    Ly = L_vec[:, :, :, 1]
    Lz = L_vec[:, :, :, 2] 

    # Averaging the angular momenta to cell centres (because mass is defined at cell centre)
    Lx = centering(Lx)
    Ly = centering(Ly)
    Lz = centering(Lz)

    # Calculate angular momentum components by multiplying with mass 
    Lx = mass * Lx
    Ly = mass * Ly
    Lz = mass * Lz

    return Lx, Ly, Lz


def calc_LRL(mass, Mstar, vx, vy, vz, Lx, Ly, Lz, X, Y, Z):
    """
    Calculating the Laplace-Runge-Lenz vector A = p x L - m * k * r^ where k is the strength of the central force k = GMm for gravitational forces

    Inputs:
    ------
    mass:     3D array of gas mass in g with shape (n_theta-1, n_r-1, n_phi-1)
    Mstar:    Mass of the central star in g (float)
    vx:       3D meshgrid of Cartesian X velocities with shape (n_theta-1, n_r-1, n_phi-1)
    vy:       3D meshgrid of Cartesian Y velocities with shape (n_theta-1, n_r-1, n_phi-1)
    vz:       3D meshgrid of Cartesian Z velocities with shape (n_theta-1, n_r-1, n_phi-1)
    Lx:       Angular momentum array in x-direction with size (theta-1, n_r-1, n_phi-1)
    Ly:       Angular momentum array in y-direction with size (theta-1, n_r-1, n_phi-1)
    Lz:       Angular momentum array in z-direction with size (theta-1, n_r-1, n_phi-1)
    X:        3D meshgrid of Cartesian X values with shape (n_theta-1, n_r-1, n_phi-1)
    Y:        3D meshgrid of Cartesian Y values with shape (n_theta-1, n_r-1, n_phi-1)
    Z:        3D meshgrid of Cartesian Z values with shape (n_theta-1, n_r-1, n_phi-1)

    Outputs:
    -------
    Ax:       3D array of X values of Laplace-Runge-Lenz vector with shape (n_theta-1, n_r-1, n_phi-1)
    Ay:       3D array of Y values of Laplace-Runge-Lenz vector with shape (n_theta-1, n_r-1, n_phi-1)
    Az:       3D array of Z values of Laplace-Runge-Lenz vector with shape (n_theta-1, n_r-1, n_phi-1)
    """

    # Defining r, v and L vectors 
    r = np.stack((X, Y, Z), axis=-1)  
    v = np.stack((vx, vy, vz), axis=-1)
    L = np.stack((Lx, Ly, Lz), axis=-1)

    # Calculating the coordinates' unit vector
    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / r_mag

    # Calculating the linear momentum vector
    p = mass[..., np.newaxis] * v

    # Calculating the Laplace-Runge-Lenz vector
    A = np.cross(p, L) - G * mass[..., np.newaxis]**2 * Mstar * r_hat
    
    # Finding x, y, z angular momentum components
    Ax = A[:, :, :, 0]
    Ay = A[:, :, :, 1]
    Az = A[:, :, :, 2]

    return Ax, Ay, Az


def calc_eccen(Ax, Ay, Az, mass, Mstar):
    """
    Calculating the eccentricity by way of the Laplace-Runge-Lenz vector with e = A / |m k|

    Inputs:
    ------
    Ax:       3D array of Laplace-Runge-Lenz vectors in x direction
    Ay:       3D array of Laplace-Runge-Lenz vectors in y direction
    Az:       3D array of Laplace-Runge-Lenz vectors in z direction
    mass:     3D array of particle masses 
    Mstar:    Mass of the primary star (float)

    Outputs:
    -------
    ex:       3D array of eccentricity in x direction
    ey:       3D array of eccentricity in y direction
    ez:       3D array of eccentricity in z direction
    """

    k = G * Mstar * mass
    ex = Ax / (mass * k)
    ey = Ay / (mass * k)
    ez = Az / (mass * k)

    return ex, ey, ez


def scale_height(r, h0, R0, f):
    """
    Calculates the scale height of the disk at a given radius r0

    Inputs:
    ------
    r:            Radius at which pressure scale height is calcualted [cm] (float)
    h0:           Aspect ratio (float)
    R0:           Radius at which aspect ratio is defined (FARGO3D standard) [cm] (float)
    f:            Disk flaring index (float)

    Outputs:
    -------
    Hc:           Scale height at given radius r0 [cm]
    """

    Hc = h0 * r * np.power(r / R0, f)             
    return Hc 


def omega_kepler(Mstar, r):
    """
    Calculates the Keplerian velocities for an array of radii
    
    Inputs:
    ------
    Mstar:   Mass of the star [g]
    r:       1D array of radii [cm]

    Outputs:
    -------
    omega_k: 1D array of Keplerian velocities [/s]
    """

    omega_k = np.sqrt(G * Mstar / r**3)
    return omega_k


def calc_surfdens(dens, theta, r, phi):
    """
    Calculates surface density
    """

    # Azimuthally averaging density
    dens_phiavg = np.mean(dens, axis=-1)   # shape (Ntheta, Nr)

    # Checking if broadcasting is okay
    dtheta = np.gradient(theta)            # shape (Ntheta,)
    theta_grid = theta[:, np.newaxis]      # shape (Ntheta, 1)
    r_grid = r[np.newaxis, :]              # shape (1, Nr)
    dtheta_grid = dtheta[:, np.newaxis]    # shape (Ntheta, 1)

    # Integrand in the surface density formula
    surf_dens_theta = dens_phiavg * r_grid * np.sin(theta_grid) * dtheta_grid  # shape (Ntheta, Nr)

    # Integrating over theta (axis=0) to get surface density as function of r
    surf_dens = np.sum(surf_dens_theta, axis=0)  # shape (Nr,)

    return surf_dens



def calc_simtime(it, ninterm=ninterm, dt=dt, stoky=stoky):
    """
    Given an iteration, calculate simulation time in kyr

    Inputs:
    ------
    it:              Simulation iteration (int)
    ninterm:         Time steps between outputs (found in .par --> output control parameters) (int)
    dt:              Time step length (found in .par --> output control parameters) (sec, float)
    stoky:           Number of seconds in 1 kyr (float)

    Outputs:
    -------
    t_ky:            Iteration in simulation time (kyr, float)
    """

    t_ky = it * ninterm * dt / stoky
    return t_ky



def isolate_disk(X, Y, Z, cenX, cenY, cenZ, buffer, dens, vx, vy, vz, Lx, Ly, Lz, threshold):
    """
    A crude way to isolate the disk: first isolating the box surrounding the disk of interest (primary or secondary) and then applying a density threshold to capture the highest densities in the simulation output to obtain the disk and reject the background

    Inputs:
    ------
    X:            3D meshgrid of Cartesian X values with shape (n_theta-1, n_r-1, n_phi-1)
    Y:            3D meshgrid of Cartesian Y values with shape (n_theta-1, n_r-1, n_phi-1)
    Z:            3D meshgrid of Cartesian Z values with shape (n_theta-1, n_r-1, n_phi-1) 
    cenX:         Stellar X-coordinate (float)
    cenY:         Stellar Y-coordinate (float)
    cenZ:         Stellar Z-coordinate (float) 
    buffer:       Size of the box to be isolated around the star 
    dens:         3D array of centered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    vx:           3D meshgrid of Cartesian X velocities with shape (n_theta-1, n_r-1, n_phi-1)
    vy:           3D meshgrid of Cartesian Y velocities with shape (n_theta-1, n_r-1, n_phi-1)
    vz:           3D meshgrid of Cartesian Z velocities with shape (n_theta-1, n_r-1, n_phi-1) 
    Lx:           Angular momentum array in x-direction with size (theta-1, n_r-1, n_phi-1)
    Ly:           Angular momentum array in y-direction with size (theta-1, n_r-1, n_phi-1)
    Lz:           Angular momentum array in z-direction with size (theta-1, n_r-1, n_phi-1)        
    threshold:    Threshold density by which we filter the dens array

    Outputs:
    -------
    disk_dens:    3D array of filtered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    disk_vx:      3D array of filtered x velocities in cm/s with shape (n_theta-1, n_r-1, n_phi-1)
    disk_vy:      3D array of filtered y velocities in cm/s with shape (n_theta-1, n_r-1, n_phi-1)
    disk_vz:      3D array of filtered z velocities in cm/s with shape (n_theta-1, n_r-1, n_phi-1)
    disk_Lx:      3D array of filtered x angular momenta in CGS with shape (n_theta-1, n_r-1, n_phi-1)
    disk_Ly:      3D array of filtered y angular momenta in CGS with shape (n_theta-1, n_r-1, n_phi-1)
    disk_Lz:      3D array of filtered z angular momenta in CGS with shape (n_theta-1, n_r-1, n_phi-1)
    ids:          3D array of indices corresponding to disk_dens
    """

    # First isolating the box around the disk of interest to avoid capturing points outside disk that meet density threshold
    box_mask = (
        (X >= cenX - buffer) & (X <= cenX + buffer) &
        (Y >= cenY - buffer) & (Y <= cenY + buffer) &
        (Z >= cenZ - buffer) & (Z <= cenZ + buffer)
    )
    box_dens = np.where(box_mask, dens, np.nan)
    box_vx = np.where(box_mask, vx, np.nan)
    box_vy = np.where(box_mask, vy, np.nan)
    box_vz = np.where(box_mask, vz, np.nan)
    box_Lx = np.where(box_mask, Lx, np.nan)
    box_Ly = np.where(box_mask, Ly, np.nan)
    box_Lz = np.where(box_mask, Lz, np.nan)

    # Filtering densities greater than a given threshold; values below the threshold are designated nan
    dens_mask = dens > 10**threshold
    disk_dens = np.where(dens_mask, box_dens, np.nan)
    disk_vx = np.where(dens_mask, box_vx, np.nan)
    disk_vy = np.where(dens_mask, box_vy, np.nan)
    disk_vz = np.where(dens_mask, box_vz, np.nan)
    disk_Lx = np.where(dens_mask, box_Lx, np.nan)
    disk_Ly = np.where(dens_mask, box_Ly, np.nan)
    disk_Lz = np.where(dens_mask, box_Lz, np.nan)

    # Also finding the corresponding x, y, z indices of the filtered densities
    ids = ~np.isnan(disk_dens)

    return disk_dens, disk_vx, disk_vy, disk_vz, disk_Lx, disk_Ly, disk_Lz, ids


def calc_L_average(Lx, Ly, Lz, mass):
    """
    Calculates the mass-averaged angular momentum averaged across the theta and phi directions to find the radial Ls

    Inputs:
    ------
    Lx:       Angular momentum array in x-direction with size (theta-1, r-1, phi-1)
    Ly:       Angular momentum array in y-direction with size (theta-1, r-1, phi-1)
    Lz:       Angular momentum array in z-direction with size (theta-1, r-1, phi-1)
    mass:     3D mass array (theta-1, r-1, phi-1)

    Outputs:
    -------
    Lx_avg:      1D array of radially averaged Lx
    Ly_avg:      1D array of radially averaged Ly
    Lz_avg:      1D array of radially averaged Lz
    """

    # Mass averaged angular momentum vectors for each shell L(r)
    Lx_avg = np.nansum(Lx * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    Ly_avg = np.nansum(Ly * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    Lz_avg = np.nansum(Lz * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))

    return Lx_avg, Ly_avg, Lz_avg


# def calc_theta_mask(theta, r, max_height, min_height=None):
#     """
#     Mask theta values only up to a certain height. Useful to calculate accretion rates and eccentricities at different heights, or filter out non-disk material

#     Inputs:
#     ------
#     max_height:      1D array of len (nr) of maximum heights to be considered at each radial point (default value is 4 scale heights to capture the disk)
#     """

#     z = r * np.cos(theta)                          # 2D array of disk heights at all radii

#     # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
#     if min_height == None:
#         theta_mask = np.abs(z) <= max_height
#         theta_sel = theta[theta_mask]
#         # print(np.round(np.degrees(theta_sel), 1))

#     elif min_height != None:
#         theta_mask = (np.abs(z) >= min_height) & (np.abs(z) <= max_height)
#         theta_sel = theta[theta_mask]

#     return theta_mask


def calc_e_average(ex, ey, ez, mass):
    """
    Calculates the mass-averaged eccentricities averaged across 4Hc in theta and (-pi, pi) in phi to get the radial profile of eccentricities
    Then masks the eccentricities only up to disk radii 

    Inputs:
    ------
    ex:       Eccentricity in x-direction with size (theta-1, r-1, phi-1)
    ey:       Eccentricity in y-direction with size (theta-1, r-1, phi-1)
    ez:       Eccentricity in z-direction with size (theta-1, r-1, phi-1)
    mass:     3D mass array (theta-1, r-1, phi-1)
    r_mask:   1D Boolean mask array for radii only up to the extent of the disk i.e. True if 
    """ 

    # Mass-weighted average in ([-4Hc,4Hc], [-pi,pi]) of eccentricities
    ex_avg = np.nansum(ex * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    ey_avg = np.nansum(ey * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    ez_avg = np.nansum(ez * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    eavg_disk = np.sqrt(ex_avg**2 + ey_avg**2 + ez_avg**2)

    # eavg_disk = eavg[r_mask[:-1]]

    return eavg_disk


def ini_cloudlet_pos(distIni, dens, dens_thresh, r, phi):
    """
    Find the initial position of the cloudlet in phi (to calculate nudge/whirl of the disk when the cloudlet hits)

    Inputs:
    ------
    distIni:        Initial distance of the cloudlet [cm]
    dens:           3D array of centered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    dens_thresh:    Density threshold to identify cloudlet centre in phi at the distIni ring [g/cm^3]

    Outputs:
    -------
    ini_cloud_phi:  Initial centre of cloudlet phi position [deg]
    """

    ir = np.argmin(np.abs(r - distIni))
    dens_r = dens[:, ir, :]          
    dens_phi = dens_r.sum(axis=0)

    # Identify cloudlet material
    mask = dens_phi > dens_thresh

    # Circular (angular) mean weighted by density
    sin_mean = np.sum(dens_phi[mask] * np.sin(phi[mask])) / np.sum(dens_phi[mask])
    cos_mean = np.sum(dens_phi[mask] * np.cos(phi[mask])) / np.sum(dens_phi[mask])

    ini_cloud_phi = np.arctan2(sin_mean, cos_mean)  # radians in [-pi, pi]
    ini_cloud_phi = np.degrees(ini_cloud_phi)
    
    return ini_cloud_phi


def calc_inc_twist(Lx_avg, Ly_avg, Lz_avg, R, savefig, plot=True):
    """
    Calculates the twist and inclination of the warped disk from the radial profile of angular momenta
    
    Inputs:
    ------
    Lx_avg:      1D array of radially averaged Lx
    Ly_avg:      1D array of radially averaged Ly
    Lz_avg:      1D array of radially averaged Lz
    R:           1D array of radii 
    savefig:     Boolean to save the plot if True
    plot:        Boolean to plot inclination / twist vs R if True

    Outputs:
    -------
    inc_deg:       1D array of radial profile of warp inclination in degrees
    twist_deg:     1D array of radial profile of warp twist in degrees
    """

    # Obtain radius values for plotting
    Rc = 0.5 * (R[1:] + R[:-1])

    Lavg_mag = np.sqrt(Lx_avg**2 + Ly_avg**2 + Lz_avg**2)

    # Calculating warped disk twist
    # twist_rad = np.arccos(Lx_avg / Lxy_proj)
    twist_abs = np.arctan2(Ly_avg, Lx_avg)
    twist_rad = twist_abs - twist_abs[0]
    twist_deg = np.degrees(twist_rad)

    # Calculating warped disk inclination (Kimmig & Dullemond 2024)
    inc = np.arccos(Lz_avg / Lavg_mag)
    inc_deg = np.degrees(inc)

    if plot:

        plt.plot(Rc / au, twist_deg)
        plt.xlabel("R [AU]")
        plt.ylabel("Twist (deg)")
        plt.show()

        plt.plot(Rc / au, inc_deg)
        plt.xlabel("R [AU]")
        plt.ylabel("Inclination (deg)")
        plt.show()
    
    return inc_deg, twist_deg


def isolate_outer_disk(R, RCYL, ZCYL, r_break, dens, Lx, Ly, Lz, threshold):
    """
    A crude way to isolate the disk: first isolating the box surrounding the disk of interest (primary or secondary) and then applying a density threshold to capture the highest densities in the simulation output to obtain the disk and reject the background

    Inputs:
    ------
    X:            3D meshgrid of Cartesian X values with shape (n_theta-1, n_r-1, n_phi-1)
    Y:            3D meshgrid of Cartesian Y values with shape (n_theta-1, n_r-1, n_phi-1)
    Z:            3D meshgrid of Cartesian Z values with shape (n_theta-1, n_r-1, n_phi-1) 
    dens:         3D array of centered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)        
    threshold:    Threshold density by which we filter the dens array

    Outputs:
    -------
    outer_dens:   3D array of filtered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    ids:          3D array of indices corresponding to disk_dens
    """

    outer_mask_geom = (R > r_break) & (RCYL < np.max(RCYL)) & (np.abs(ZCYL) < np.max(np.abs(ZCYL)))
    outer_dens_box = np.where(outer_mask_geom, dens, np.nan)
    outer_Lx_box = np.where(outer_mask_geom, Lx, np.nan)
    outer_Ly_box = np.where(outer_mask_geom, Ly, np.nan)
    outer_Lz_box = np.where(outer_mask_geom, Lz, np.nan)

    outer_mask_dense = dens > 10**threshold
    outer_dens = np.where(outer_mask_dense, outer_dens_box, np.nan)
    outer_Lx = np.where(outer_mask_dense, outer_Lx_box, np.nan)
    outer_Ly = np.where(outer_mask_dense, outer_Ly_box, np.nan)
    outer_Lz = np.where(outer_mask_dense, outer_Lz_box, np.nan)

    # Also finding the corresponding x, y, z indices of the filtered densities
    ids = ~np.isnan(outer_dens)

    return outer_dens, outer_Lx, outer_Ly, outer_Lz, ids


def calc_total_L(Lx_avg, Ly_avg, Lz_avg):
    """
    Calculates the total angular momentum of the disk with reference to the z-axis by averaging over the radial profile of the angular momenta

    Inputs:
    ------
    Lx_avg:      1D array of radially averaged Lx
    Ly_avg:      1D array of radially averaged Ly
    Lz_avg:      1D array of radially averaged Lz

    Outputs:
    -------
    Lx_tot:      Total L of warped disk in x-direction (float)
    Ly_tot:      Total L of warped disk in y-direction (float)
    Lz_tot:      Total L of warped disk in z-direction (float)
    """

    Lx_tot = np.sum(Lx_avg)
    Ly_tot = np.sum(Ly_avg)
    Lz_tot = np.sum(Lz_avg)

    return Lx_tot, Ly_tot, Lz_tot


def calc_whirl(Lx_tot, Ly_tot, Lz_tot, ini_cloud_phi):
    """
    
    """

    # Total disk polar angle
    disk_polar = np.arctan2(Ly_tot, Lx_tot)

    # Calculating warped disk whirl (how much the disk as a whole turns around in XY plane when the cloudlet hits at an angle)
    whirl = disk_polar - np.deg2rad(ini_cloud_phi)
    whirl_deg = np.rad2deg(whirl)

    return whirl_deg


def main():

    folder = Path("../cloud_disk_it450_retro_rotY45/")                        # Folder with the output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_retro_rotY45")         # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_disk_it450_retro_rotY45/imgs/")                    # Folder to save images
    it = 450                                                       # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)
    
    # Load some simulation parameters
    b = get_param_value('ImpactParameter', sim_name)

    # Save simulation parameters into dictionary for plotting purposes
    plot_args = {"Time": f"{int(calc_simtime(it))} kyr", "b": b}
    

    ###################### Load data (theta = 175, r = 150, phi = 100) ################################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
    vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
    vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
    vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

    # all_positive = np.all(rho > 0)
    # print(all_positive)

    vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

    THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
    X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

    # Cartesian velocities
    vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)

    # Central coordinates of the primary
    Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation


    ############################# Calculate physical quantities ######################################


    # Interpolate the densities & coordinates to the cell centres so that the array shape matches with mass & L
    rho_c = centering(rho)
    X_c = centering(X)
    Y_c = centering(Y)
    Z_c = centering(ZCYL)
    vx_c = centering(vx)
    vy_c = centering(vy)
    vz_c = centering(vz)

    cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])          # Volume 3D
    mass = calc_mass(rho, cell_volume)                                                      # Mass 3D
    surf_dens = calc_surfdens(rho, domains["theta"], domains["r"], domains["phi"])          # Surface density 3D
    Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)                        # Angular momentum 3D
    Ax, Ay, Az = calc_LRL(mass, Mstar, vx_c, vy_c, vz_c, Lx, Ly, Lz, X_c, Y_c, Z_c)         # Laplace-Runge-Lenz vector 3D
    ex, ey, ez = calc_eccen(Ax, Ay, Az, mass, Mstar)                                        # Eccentricity 3D
    e = np.sqrt(ex**2 + ey**2 + ez**2)                                                      # Absolute eccentricity 3D 

    print(Lx.shape, Ly.shape, Lz.shape)
    print(e.shape)

    ########################### Isolating the warp in the primary disk ###############################


    # Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
    # Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
    warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
    warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
    rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 

    # Find the radial extent of the warp
    r_warp_extent = np.sqrt(X_c[warp_ids]**2 +  Y_c[warp_ids]**2 + Z_c[warp_ids]**2) / au
    mask = (domains["r"]/au >= r_warp_extent.min()) & (domains["r"]/au <= r_warp_extent.max())
    r_select = domains["r"][mask]
    print(sim_name, r_select/au)

    plot_args[r"$\mathrm{Box}_{\mathrm{prim}}$"] = f"{2 * warp_buffer} AU"
    plot_args[r"$\rho_{\mathrm{prim}} \geq$"] = fr"$10^{{{warp_thresh}}} g/cm^3$"
    

    #################################### Plotting warp/disk properties #####################################


    # 2D visualizations
    # irad = -1
    irad = np.where(domains["r"]/au < 350)[0][-1]
    iphi = 0
    itheta = int(7*len(domains["theta"])/15)
    # print(itheta)
    itheta_deg = np.round(np.rad2deg(domains["theta"][itheta]), 2)
    print(itheta_deg)

    rho_phiavg = np.mean(rho, axis=2)

    # Azimuthally averaged density RZ plot
    # cyl_2D_plot(rho_phiavg, RCYL, ZCYL, irad, iphi, title=rf'{sim_name}: $\phi$ Averaged Density R-Z Plane', colorbarlabel=r"$\rho (g/cm^{3})$", savefig=True, figfolder=f'{fig_imgs}/dens_phiavg_cyl_phi{iphi}_rad{irad}_it{it}.png', showfig=True, data_phiavg=True)

    # Density RZ plot
    cyl_2D_plot(rho, RCYL, ZCYL, irad, iphi, vmin=-17, vmax=-11, title=rf'{sim_name}: Density R-Z Plane $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$\rho (g/cm^{3})$", savefig=True, figfolder=f'{fig_imgs}/dens_cyl_phi{iphi}_rad{irad}_it{it}.png', showfig=True, data_phiavg=False)

    # XY_2D_plot(vrad * 1e-5, X, Y, irad, itheta, vmin=-0.5, vmax=0.5, title=rf'{sim_name}: X-Y Radial Velocity $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$v_{rad} (km/s)$", savefig=True, figfolder=f'{fig_imgs}/vrad_xy_phi{iphi}_rad{irad}_it{it}.png', showfig=True)

    # Radial velocity RZ plot (multiplying vrad by 1e-5 to convert cm/s to km/s)
    # vel_cyl_2D(vrad * 1e-5, rho, RCYL, ZCYL, irad, iphi, title=rf'{sim_name}: Radial Velocities R-Z Plane $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$\mathrm{v_{rad} (\rm km/s)}$", savefig=True, figfolder=f'{fig_imgs}/radvel_cyl_phi{iphi}_rad{irad}_it{it}.png', showfig=True, acc=False, data_phiavg=False)
    # edjwfk
 
    # Rho * Radial velocity RZ plot (multiplying vrad by 1e-5 to convert cm/s to km/s)
    # vel_cyl_2D(vrad, rho, RCYL, ZCYL, irad, iphi, title=rf'{sim_name}: $\rho v_{{rad}}$ R-Z Plane $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$\mathrm{\rho v_{rad} (\rm g \rm km/s)}$", savefig=True, figfolder=f'{fig_imgs}/rhoradvel_cyl_phi{iphi}_rad{irad}_it{it}.png', showfig=True, acc=True, data_phiavg=False)

    # Comparing subsonic vs supersonic regions in meridional flows
    h0 = get_param_value("AspectRatio", sim_name)      # Aspect ratio
    f = get_param_value("FlaringIndex", sim_name)      # Flaring index
    R0 = 5.2 * au                                      # As defined in FARGO3D [cm]

    omega_k = omega_kepler(Mstar, RCYL)             # Array of Keplerian velocities (s^-1)
    Hc_arr = scale_height(RCYL, h0, R0, f)          # Array of pressure scale heights (cm)
    cs = omega_k * Hc_arr                             # Array of local isothermal speed of sound (cm/s)
    # print(np.mean(cs), np.mean(omega_k), np.mean(Hc_arr))
    # print(np.shape(omega_k), np.shape(Hc_arr))
    # print(np.max(vrad / cs), np.min(vrad / cs), np.mean(vrad / cs))
    # print(vrad / cs)

    # vrad / cs RZ plot 
    # vel_cyl_2D(vrad / cs, rho, RCYL, ZCYL, irad, iphi, title=rf'{sim_name}: $v_{{rad}} / c_s$ R-Z Plane $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$\mathrm{v_{rad} / c_s}$", savefig=True, figfolder=f'{fig_imgs}/radvelbycs_cyl_phi{iphi}_rad{irad}_it{it}.png', showfig=True, acc=False, data_phiavg=False)

    XY_2D_plot(rho, X, Y, irad, itheta, vmin=-17, vmax=-11, title=rf'{sim_name}: Density X-Y Plane $\theta = $ {itheta_deg}$^{{\circ}}$', colorbarlabel=r"$\log(\rho)$", savefig=True, figfolder=f'{fig_imgs}/dens_xy_theta{itheta}_rad{irad}_it{it}.png', showfig=True)

    # Plotting the 3D warp/disk densities 
    contours_3D(X_c/au, Y_c/au, Z_c/au, np.log10(rho_c_warp), r_select, plot_args, colorbarlabel=r'$\log(\rho) [g/cm^3]$', title=rf'{sim_name}: Initial & Outer Disks: $\log(\rho)$', savefig=True, figfolder=f'{fig_imgs}/warp_dens_thresh{warp_thresh}_it{it}.png', showfig=True)
    
    # Another way to plot the warp/disk densities
    # contours_3D(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, rho_c[warp_ids], fig, colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

    # Plotting the disk velocities
    # azim=-90, elev=-108 (Top-down)
    # quiver_plot_3d(X_c/au, Y_c/au, Z_c/au, vx_c_warp, vy_c_warp, vz_c_warp, stagger=5, length=30, title=f"{sim_name}: velocities {int(calc_simtime(it))} kyr", colorbarlabel="velocities", savefig=True, figfolder=f'{fig_imgs}/warp_vel_thresh{warp_thresh}_it{it}.png', azim=-90, elev=-170, logmag=True, ignorecol=True)

    # Nstagger = int(len(X_c[warp_ids])/500)
    # Nstagger = 100
    # stagger_ids = np.linspace(1000, len(X_c[warp_ids])-1, Nstagger, dtype=int)
    # plotly_quiver3D(X_c[warp_ids][stagger_ids]/au, Y_c[warp_ids][stagger_ids]/au, Z_c[warp_ids][stagger_ids]/au, (vx_c[warp_ids]/vsph.mean())[stagger_ids], (vy_c[warp_ids]/vsph.mean())[stagger_ids], (vz_c[warp_ids]/vsph.mean())[stagger_ids], savefig=False, figfolder=f'{fig_imgs}/plotly_vel_thresh{warp_thresh}_it{it}.png', showfig=True)

    # Disk velocity streamline plots at different theta values
    # targets = np.array([60, 70, 80, 90, 100, 110])   
    # ithetas = [np.abs(np.rad2deg(domains["theta"]) - t).argmin() for t in targets]
    # velocity_streamlines(X_c, Y_c, vx_c_warp, vy_c_warp, rho_c_warp, ithetas, irad, savefig=True, figfolder=f'{fig_imgs}/vel_streamlines_thresh{warp_thresh}_it{it}.png', showfig=True)

    # quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, vx_c[warp_ids], vy_c[warp_ids], vz_c[warp_ids], stagger=10, length=30, title=f"{sim_name}: velocities {int(calc_simtime(it))} kyr", colorbarlabel="velocities", savefig=False, figfolder=f'../warp_vel_thresh{warp_thresh}_it{it}.png', azim=-90, elev=-178, logmag=True, ignorecol=False)

    # Plotting the Cartesian warp/disk angular momenta
    # quiver_plot_3d(X_c/au, Y_c/au, Z_c/au, Lx_c_warp, Ly_c_warp, Lz_c_warp, stagger=10, length=2, title="Disk Angular Momenta", colorbarlabel="logL", savefig=False, figfolder=f'../warp_L_thresh{warp_thresh}_it{it}.png', logmag=True, ignorecol=True)

    # Another way to plot the Cartesian warp/disk angular momenta
    # quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Lx[warp_ids], Ly[warp_ids], Lz[warp_ids], stagger=100, length=3, title="Warp Angular Momenta", colorbarlabel="logL", savefig=True, figfolder=f'{fig_imgs}/warp_L_thresh{warp_thresh}_it{it}.png', logmag=True)

    # Calculating the radial profile of warp/disk inclination and precession according to Kimmig & Dullemond (2024)
    Lx_warp_avg, Ly_warp_avg, Lz_warp_avg = calc_L_average(Lx_c_warp, Ly_c_warp, Lz_c_warp, mass)
    inc, twist = calc_inc_twist(Lx_warp_avg, Ly_warp_avg, Lz_warp_avg, domains["r"], savefig=False, plot=False)
    # print(np.min(twist), np.max(twist), np.mean(twist))
    # print(np.min(inc), np.max(inc), np.mean(inc))

    # print(X)
    # irad_e = np.where(domains["r"]/au < 200)[0][-1]
    # XY_2D_plot(e, X_c, Y_c, irad_e, itheta, vmin=0, vmax=0.2, title=rf'{sim_name}: X-Y Eccentricity $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$e$", savefig=True, figfolder=f'{fig_imgs}/e_xy_phi{iphi}_rad{irad_e}_it{it}.png', showfig=True)

    # iphi = 50
    # cyl_2D_plot(10**e, R_c, Z_c, irad_e, iphi, vmin=0, vmax=1, title=rf'{sim_name}: Eccentricity R-Z Plane $\phi$ = {np.round(np.degrees(domains["phi"][iphi]), 2)}$^{{\circ}}$', colorbarlabel=r"$e$", savefig=True, figfolder=f'{fig_imgs}/e_cyl_phi{iphi}_rad{irad_e}_it{it}.png', showfig=True, data_phiavg=False)

    # Characterizing warp eccentricity 
    # print("Min Warp eccentricity: ", np.min(e[warp_ids]))
    # print("Max Warp eccentricity: ", np.max(e[warp_ids]))
    # print("Mean warp eccentricity: ", np.mean(e[warp_ids]))

    ############### Plotting azimuthally and polar averaged eccentricity (of the disk only, not the whole sim space!) as a radial profile
    # Mass averaged angular momentum vectors for each shell L(r)
    # ex_avg = np.nansum(ex * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    # ey_avg = np.nansum(ey * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    # ez_avg = np.nansum(ez * mass, axis=(0,2)) / np.sum(mass, axis=(0,2))
    # eavg = np.sqrt(ex_avg**2 + ey_avg**2 + ez_avg**2)

    print(np.isnan(Lx_c_warp), np.isnan(Ly_c_warp), np.isnan(Lz_c_warp))
    print(np.sum(np.isnan(Lx_c_warp)), np.sum(np.isnan(Ly_c_warp)), np.sum(np.isnan(Lz_c_warp)))
    print(np.shape(Lx_c_warp), np.shape(Ly_c_warp), np.shape(Lz_c_warp))
    
    Ax_warp, Ay_warp, Az_warp = calc_LRL(mass, Mstar, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, X_c, Y_c, Z_c)
    ex_warp, ey_warp, ez_warp = calc_eccen(Ax_warp, Ay_warp, Az_warp, mass, Mstar)
    eavg_disk = calc_e_average(ex_warp, ey_warp, ez_warp, mass)
    print(eavg_disk)
    print(eavg_disk.shape)
    print(np.isnan(eavg_disk))

    # Radial profile of mass-averaged eccentricities
    fig, ax = plt.subplots()
    ax.plot(np.log10(r_select/au), eavg_disk[mask[:-1]], lw=2.5)
    # ax.plot(np.log10(domains["r"][:-1]/au), eavg_disk)
    ax.set_xlabel("log(R [AU])")
    ax.set_ylabel("e")
    fig.tight_layout()
    plt.savefig(f"{fig_imgs}/e_vs_r_it{it}.png")
    plt.show()


    ###################################### Isolating the outer disk ############################################
    

    # Finding the radial separation between the inner and outer disks at the discontinuity of dinc/dr
    r_break = plot_disk_sep(domains["r"], inc, title=rf"{sim_name}: Inclination Gradient", savefig=True, figfolder=f'{fig_imgs}/dinc_dr.png', showfig=True)
    # print("r_break:", r_break/au)

    # Isolating the outer disk using r_break and a density threshold
    R_c = centering(R)
    RCYL_c = centering(RCYL)
    outer_thresh = -17
    outer_rho, Lx_outer, Ly_outer, Lz_outer, outer_ids = isolate_outer_disk(R_c, RCYL_c, Z_c, r_break, rho_c, Lx, Ly, Lz, threshold=outer_thresh)
    r_outer_extent = np.sqrt(X_c[outer_ids]**2 +  Y_c[outer_ids]**2 + Z_c[outer_ids]**2) / au
    mask = (domains["r"]/au >= r_outer_extent.min()) & (domains["r"]/au <= r_outer_extent.max())
    r_outer = domains["r"][mask]

    # Plotting outer disk in 3D
    # contours_3D(X_c/au, Y_c/au, Z_c/au, np.log10(outer_rho), r_outer, sim_params=None, colorbarlabel=r'$\log(\rho) [g/cm^3]$', title=rf'{sim_name}: Outer Disk: $\log(\rho)$', savefig=True, figfolder=f'{fig_imgs}/outer_dens_thresh{outer_thresh}_it{it}.png', showfig=True)


    ###################### Disk morphology properties separating inner and outer disks #############################


    ######### Total disk properties 

    # Calculating and plotting the total angular momentum of the warped disk
    Lx_disk, Ly_disk, Lz_disk = calc_total_L(Lx_warp_avg, Ly_warp_avg, Lz_warp_avg)

    # Initial cloudlet position
    cloud_dist = get_param_value("DistIni", sim_name)
    rho0 = get_data(folder, "dens", 0, domains)         # Load 3D array of density values at first iteration
    cloud_phi = ini_cloudlet_pos(cloud_dist, rho0, 1e-17, domains["r"], domains["phi"])
    # print("CLOUD_PHI:", cloud_phi)

    whirl = calc_whirl(Lx_disk, Ly_disk, Lz_disk, cloud_phi)
    # print("TOTAL DISK WHIRL: ", whirl)

    # Calculating and plotting the radial profile of warp/disk precession as a quiver plot
    plot_twist_arrows(Lx_warp_avg, Ly_warp_avg, Lz_warp_avg, domains["r"], r_select, plot_args, title=f"{sim_name}: Disk Twist", savefig=True, figfolder=f'{fig_imgs}/warp_twist_arrows_it{it}_dens{warp_thresh}.png', showfig=True)
    
    ######### Outer disk properties 

    # Radially averaged outer disk momenta
    Lx_outer_avg, Ly_outer_avg, Lz_outer_avg = calc_L_average(Lx_outer, Ly_outer, Lz_outer, mass)
    # plot_twist_arrows(Lx_outer_avg, Ly_outer_avg, Lz_outer_avg, domains["r"], r_outer, sim_params=None, title=f"{sim_name}: Outer Disk Twist", savefig=True, figfolder=f'{fig_imgs}/outer_twist_arrows_it{it}_dens{outer_thresh}.png', showfig=True)

    # Total outer disk momenta
    Lx_outer_disk, Ly_outer_disk, Lz_outer_disk = calc_total_L(Lx_outer_avg, Ly_outer_avg, Lz_outer_avg)
    outer_whirl = calc_whirl(Lx_outer_disk, Ly_outer_disk, Lz_outer_disk, cloud_phi)
    # print("OUTER DISK WHIRL: ", outer_whirl)

    # plotly_quiver3D(np.logspace(1, np.log10(r_select.max()/au), len(Lx_warp_avg)), np.zeros(len(Lx_warp_avg)), np.zeros(len(Lx_warp_avg)), Lx_warp_avg/Lx_disk, Ly_warp_avg/Ly_disk, Lz_warp_avg/Lz_disk, savefig=False, figfolder=f'{fig_imgs}/plotly_vel_thresh{warp_thresh}_it{it}.png', showfig=True)

    # quiver_plot_3d(np.array([Px]), np.array([Py]), np.array([Pz]), np.array([Lx_disk]), np.array([Ly_disk]), np.array([Lz_disk]), stagger=1, length=0.05, title=f"{sim_name} Total disk angular momentum", colorbarlabel="logL", savefig=False, figfolder=f'{fig_imgs}/{sim_name}_totalL.png', logmag=True)

    # plot_total_disks_bonanza(X_c/au, Y_c/au, Z_c/au, rho_c_warp, None, Px, Py, Pz, Lx_disk, Ly_disk, Lz_disk, sim_params, length=150, Rwarp=r_select, azim=-106, elev=36, colorbarlabel=r'$\rho_{norm}$', title=rf'{sim_name} Disk $\rho$ and L, t = {int(it * dt * ninterm / stoky)} kyr', savefig=True, figfolder=f'{fig_imgs}/total_bonanza_it{it}_dens{warp_thresh}.png', showfig=True)

    # Plotting warp surface density
    # plot_surf_dens(X_c, Y_c, Z_c, surf_dens, warp_ids, domains["r"], savefig=False, figfolder=f'../warp_L_thresh{warp_thresh}_it{it}.png', showfig=True)

    # Plotting warp Laplace-Runge-Lenz vector
    # quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Ax[warp_ids], Ay[warp_ids], Az[warp_ids], stagger=500, length=10, title=rf'Warp LRL', colorbarlabel=r'$\log(A [g^2cm^3/s^2])$', savefig=True, figfolder=f'{fig_imgs}/warp_{it}_LRL.png', logmag=True)

    # Plotting warp eccentricity
    # quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, ex[warp_ids], ey[warp_ids], ez[warp_ids], stagger=70, length=30, title=rf'Warp Eccentricity', colorbarlabel=r'$e$', savefig=True, figfolder=f'../warp_{it}_ecc.png', logmag=False)
   

    ############################### Loading / calculating companion properties #####################################



    # Loading central coordinates of the companion for the given iteration it in the simulation
    # df_planet = pd.read_table(folder / "planet0.dat", header = None) #, sep='\s')
    # comp_cenx = df_planet[1].iloc[it]        # X-coordinate of companion
    # comp_ceny = df_planet[2].iloc[it]        # Y-coordinate of companion
    # comp_cenz = df_planet[3].iloc[it]        # Z-coordinate of companion
    # comp_cen_vx = df_planet[4].iloc[it]      # X velocity of companion
    # comp_cen_vy = df_planet[5].iloc[it]      # Y velocity of companion
    # comp_cen_vz = df_planet[6].iloc[it]      # Z velocity of companion
    # Mcomp = df_planet[7].iloc[it]            # Mass of the companion in g
    
    # # Transforming our (X, Y, Z) and our (vx, vy, vz) to have the companion at the centre of the grid
    # comp_X = X - comp_cenx
    # comp_Y = Y - comp_ceny
    # comp_Z = ZCYL - comp_cenz
    # comp_vx = vx - comp_cen_vx
    # comp_vy = vy - comp_cen_vy
    # comp_vz = vz - comp_cen_vz 

    # # Centering these new grids
    # comp_Xc = centering(comp_X)
    # comp_Yc = centering(comp_Y)
    # comp_Zc = centering(comp_Z)
    # comp_vx_c = centering(comp_vx)
    # comp_vy_c = centering(comp_vy)
    # comp_vz_c = centering(comp_vz)

    # # Angular momentum, eccentricity of the secondary disk is to be calculated with respect to the companion star
    # Lx_comp, Ly_comp, Lz_comp = calc_angular_momentum(mass, comp_X, comp_Y, comp_Z, comp_vx, comp_vy, comp_vz)
    # Ax_comp, Ay_comp, Az_comp = calc_LRL(mass, Mcomp, comp_vx_c, comp_vy_c, comp_vz_c, Lx_comp, Ly_comp, Lz_comp, comp_Xc, comp_Yc, comp_Zc)
    # ex_comp, ey_comp, ez_comp = calc_eccen(Ax_comp, Ay_comp, Az_comp, mass, Mcomp)
    # e_comp = np.sqrt(ex_comp**2 + ey_comp**2 + ez_comp**2)


    ####################################### Isolating the companion disk ###########################################


    # comp_thresh = -17.5   # log of density threshold for which we can see the companion
    # comp_buffer = 70    # Isolates a box of 2 * comp_buffer around the star (AU)
    # rho_c_compmask, vx_c_compmask, vy_c_compmask, vz_c_compmask, Lx_c_compmask, Ly_c_compmask, Lz_c_compmask, comp_ids = isolate_disk(comp_Xc, comp_Yc, comp_Zc, 0, 0, 0, comp_buffer * au, rho_c, comp_vx_c, comp_vy_c, comp_vz_c, Lx_comp, Ly_comp, Lz_comp, comp_thresh)  

    # # Find the radial extent of the companion
    # r_comp_extent = np.sqrt(X_c[comp_ids]**2 +  Y_c[comp_ids]**2 + Z_c[comp_ids]**2) / au
    # comp_mask = (domains["r"]/au >= r_comp_extent.min()) & (domains["r"]/au <= r_comp_extent.max())
    # r_comp_select = domains["r"][comp_mask]

    # plot_args[r"$\mathrm{Box}_{\mathrm{sec}}$"] = f"{2 * comp_buffer} AU"
    # plot_args[r"$\rho_{\mathrm{sec}} \geq$"] = fr"$10^{{{comp_thresh}}} g/cm^3$"


    ####################################### Plotting the companion properties #####################################


    # Plotting the companion densities (with the grid centre being the primary star!)
    # contours_3D(X_c/au, Y_c/au, Z_c/au, rho_c_compmask, plot_args, colorbarlabel=r'$\rho [g/cm^3]$', title=r'Secondary disk: $\rho$', savefig=False, figfolder=f'{fig_imgs}/comp_dens_thresh{comp_thresh}_it{it}.png', showfig=True)

    # Plotting the companion densities (with the grid centre being the companion star!)
    # contours_3D(comp_Xc/au, comp_Yc/au, comp_Zc/au, rho_c_compmask, colorbarlabel=r'$\rho [g/cm^3]$', title=rf'{sim_name} Secondary $\rho$ above $\rho = 10^{{{comp_thresh}}} g/cm^3$, t = {int(it * dt * ninterm / stoky)} kyr', savefig=False, figfolder=f'{fig_imgs}/comp_dens_thresh{comp_thresh}_it{it}.png', showfig=True)

    # Plotting the Cartesian companion angular momenta (with the grid centre being the primary star!)
    # quiver_plot_3d(X_c[comp_ids]/au, Y_c[comp_ids]/au, Z_c[comp_ids]/au, Lx_comp[comp_ids], Ly_comp[comp_ids], Lz_comp[comp_ids], stagger=1, length=20, title=f"{sim_name}: Companion angular momenta, t = {int(it * dt * ninterm / stoky)} kyr", colorbarlabel="logL", savefig=False, figfolder=f'{fig_imgs}/comp_L_thresh{comp_thresh}_it{it}.png', logmag=True)

    # Visualizing both primary and secondary disks and their total angular momenta
    # Lx_comp_avg, Ly_comp_avg, Lz_comp_avg = calc_L_average(Lx_c_compmask, Ly_c_compmask, Lz_c_compmask)
    # Lx_comp_disk, Ly_comp_disk, Lz_comp_disk = calc_total_L(Lx_comp_avg, Ly_comp_avg, Lz_comp_avg)

    # plot_total_disks_bonanza(X_c/au, Y_c/au, Z_c/au, rho_c_warp, rho_c_compmask, np.array([Px / au, comp_cenx / au]), np.array([Py / au, comp_ceny / au]), np.array([Pz / au, comp_cenz / au]), np.array([Lx_disk, Lx_comp_disk]), np.array([Ly_disk, Ly_comp_disk]), np.array([Lz_disk, Lz_comp_disk]), plot_args, length=150, colorbarlabel=r'$\rho_{norm}$', title=r'Disks $\rho$ and L', savefig=False, figfolder=f'{fig_imgs}/total_bonanza_it{it}.png', showfig=True)


    ####################################################################################################


if __name__ == "__main__":
    main()

