import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import astropy.constants as c
au = c.au.cgs.value


def calc_cell_volume(theta, r, phi):
    """
    Calculates the cell volumes of an (theta,r,phi) grid in cm^3 using the fomula V = r^2 sin(theta) dr dphi dtheta
    
    Inputs:
    ------
    theta:    1D array of polar angles
    r:        1D array of radii
    phi:      1D array of azimuthal angles

    Outputs:
    -------
    cell_vol: 3D array of cell volumes with shape (theta-1, r-1, phi-1)
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
    Calculates the mass of each cell in g as m = rho * volume

    Inputs:
    -------
    rho:      3D array of density in g/cm^3 with shape (theta, r, phi)
    cell_vol: 3D array of cell volumes in cm^3 with shape (theta-1, r-1, phi-1)

    Outputs:
    --------
    mass:     3D array of mass in g with shape (theta-1, r-1, phi-1)
    """

    # Centering the rho grid so that we take the density at the centre of the cells
    rho_c = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
    rho_c = 0.5 * (rho_c[:, :-1, :] + rho_c[:, 1:, :])
    rho_c = 0.5 * (rho_c[:, :, :-1] + rho_c[:, :, 1:])
    mass = rho_c * cell_vol
    return mass


def calc_angular_momentum(mass, x, y, z, vx, vy, vz):
    """
    Calculates the angular momentum across each cell in (g cm^2/s) as L = m(r x v)

    Inputs:
    -------
    mass:     3D array of mass in g with shape (theta-1, r-1, phi-1)

    """

    r_vec = np.stack((x, y, z), axis=-1)
    v_vec = np.stack((vx, vy, vz), axis=-1)
    L_vec = np.cross(r_vec, v_vec, axis=-1)
    return L_vec




folder = Path("leon_snapshot/")         # Folder with the output files
it = 600                                # FARGO snapshot

############# theta = 100, r = 250, phi = 225 ###########
domains = get_domain_spherical(folder)
rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values
vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
RCYL = R * np.sin(THETA)
ZCYL = R * np.cos(THETA)

vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

# Cartesian velocities
vx = vrad * np.sin(THETA) * np.cos(PHI) + vthe * np.cos(THETA) * np.cos(PHI) - vphi * np.sin(PHI)
vy = vrad * np.sin(THETA) * np.sin(PHI) + vthe * np.cos(THETA) * np.sin(PHI) + vphi * np.cos(PHI)
vz = vrad * np.cos(THETA) - vthe * np.sin(THETA)

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
mass = calc_mass(rho, cell_volume)
calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)