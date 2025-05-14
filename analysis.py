import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import matplotlib.pyplot as plt
from no_thoughts_just_plots import quiver_plot_3d, contours_3D
import astropy.constants as c
au = c.au.cgs.value


def sph_to_cart(THETA, R, PHI):
    """
    Creates a meshgrid of Cartesian and/or cylindrical coordinates given spherical coordinates 
    """

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    RCYL = R * np.sin(THETA)
    ZCYL = R * np.cos(THETA)

    return X, Y, ZCYL, RCYL 


def cart_to_sph(xx, yy, zz):
    """
    Convert Cartesian coordinates to spherical coordinates
    """

    r_box     = np.sqrt(xx**2+yy**2+zz**2)
    rc_box    = np.sqrt(xx**2+yy**2)
    theta_box = np.pi/2-np.arctan(zz/(rc_box+1e-99))
    phi_box   = np.arctan2(yy,xx)
    phi_box[phi_box<0]+=np.pi*2

    return r_box, rc_box, theta_box, phi_box


def vel_sph_to_cart(vthe, vrad, vphi, THETA, R, PHI):
    """
    Converts velocities in spherical coordinates to Cartesian coordinates
    """

    vx = vrad * np.sin(THETA) * np.cos(PHI) + vthe * np.cos(THETA) * np.cos(PHI) - vphi * np.sin(PHI)
    vy = vrad * np.sin(THETA) * np.sin(PHI) + vthe * np.cos(THETA) * np.sin(PHI) + vphi * np.cos(PHI)
    vz = vrad * np.cos(THETA) - vthe * np.sin(THETA)

    return vx, vy, vz


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


def calc_eccent():
    return 


def isolate_warp(dens, threshold):
    """
    A crude way to isolate the warp in the primary disk: applying a density threshold
    """

    # Filtering densities greater than a given threshold
    mask = dens > 10**threshold
    warp_dens = np.where(mask, dens, np.nan)

    # Also finding the corresponding x, y, z indices of the filtered densities
    x, y, z = np.indices(dens.shape)
    warpx = np.where(mask, x, -1)
    warpy = np.where(mask, y, -1)
    warpz = np.where(mask, z, -1)

    return warp_dens, warpx, warpy, warpz



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
L = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)
Lx = L[:, :, :, 0]
Ly = L[:, :, :, 1]
Lz = L[:, :, :, 2]


# Isolating the warp in the primary disk
threshold = -14
rho_warp = isolate_warp(rho, threshold)

# quiver_plot_3d(X[::7, :20:7, ::7]/au, Y[::7, :20:7, ::7]/au, ZCYL[::7, :20:7, ::7]/au, Lx[::7, :20:7, ::7], Ly[::7, :20:7, ::7], Lz[::7, :20:7, ::7])
# quiver_plot_3d(X[50, :170:5, ::5]/au, Y[50, :170:5, ::5]/au, ZCYL[50, :170:5, ::5]/au, Lx[50, :170:5, ::5], Ly[50, :170:5, ::5], Lz[50, :170:5, ::5])

# fig = plt.figure(figsize=(10, 7))
# contours_3D(X[::5, :20:5, ::5]/au, Y[::5, :20:5, ::5]/au, ZCYL[::5, :20:5, ::5]/au, np.log10(rho[::5, :20:5, ::5]), fig, xlabel='x', ylabel='y', zlabel='z', colorbarlabel='dens', title='dens')
# print(L[0, 0, 0, :].shape)
# print(L.shape)
# print(L[0, 0, 0, :])
# print(L[0,0,0,-1])
# print(L[:, :, :, 0], L[:, :, :, 0].shape)
# print(L[:, :, :, 0], L[:, :, :, 1].shape)

# Plotting the warp
fig = plt.figure(figsize=(10, 7))
# contours_3D(X[::2, ::2, ::2]/au, Y[::2, ::2, ::2]/au, ZCYL[::2, ::2, ::2]/au, np.log10(rho_warp[::2, ::2, ::2]), fig, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\log(\rho)$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

contours_3D(X/au, Y/au, ZCYL/au, rho_warp, fig, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

quiver_plot_3d(X/au, Y/au, ZCYL/au, Lx, Ly, Lz)