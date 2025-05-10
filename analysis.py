import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import astropy.constants as c
au = c.au.cgs.value


def calc_cell_volume(theta, r, phi):
    """
    Calculates the cell volumes of an (theta,r,phi) grid 
    theta = 100, r = 250, phi = 225
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
    Calculates the mass of each cell 
    """

    # Centering the rho grid so that we take the density at the centre of the cells
    rho_c = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
    rho_c = 0.5 * (rho_c[:, :-1, :] + rho_c[:, 1:, :])
    rho_c = 0.5 * (rho_c[:, :, :-1] + rho_c[:, :, 1:])
    mass = rho_c * cell_vol
    return mass

folder = Path("leon_snapshot/")         # Folder with the output files
it = 600                                # FARGO snapshot

############# theta = 100, r = 250, phi = 225 ###########
domains = get_domain_spherical(folder)
rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
mass = calc_mass(rho, cell_volume)
print(mass)