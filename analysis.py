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

    r_box     = np.sqrt(xx**2 + yy**2 + zz**2)
    rc_box    = np.sqrt(xx**2 + yy**2)
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


def centering(data):
    """
    Center the data from cell walls to cell centres

    Inputs:
    ------
    data:      3D array with shape (theta, r, phi)

    Outputs:
    -------
    data_c:    Centered 3D array with shape (theta-1, r-1, phi-1)
    """

    data_c = 0.5 * (data[:-1, :, :] + data[1:, :, :])
    data_c = 0.5 * (data_c[:, :-1, :] + data_c[:, 1:, :])
    data_c = 0.5 * (data_c[:, :, :-1] + data_c[:, :, 1:])

    return data_c 



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
    # rho_c = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
    # rho_c = 0.5 * (rho_c[:, :-1, :] + rho_c[:, 1:, :])
    # rho_c = 0.5 * (rho_c[:, :, :-1] + rho_c[:, :, 1:])
    rho_c = centering(rho)
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

    # Find angular momentum components
    Lx = L_vec[:, :, :, 0]
    Ly = L_vec[:, :, :, 1]
    Lz = L_vec[:, :, :, 2] 

    # Average the angular momenta to cell centres (because mass is defined at cell centre)
    Lx = centering(Lx)
    Ly = centering(Ly)
    Lz = centering(Lz)

    # Find angular momentum components and multiply by 
    Lx = mass * Lx
    Ly = mass * Ly
    Lz = mass * Lz

    return Lx, Ly, Lz


def calc_L_average(Lx, Ly, Lz):
    """
    Calculates the angular momentum averaged across the theta and phi directions to find the radial Ls

    Inputs:
    ------
    Lx:       Angular momentum array in x-direction with size (theta, r, phi)
    Ly:       Angular momentum array in y-direction with size (theta, r, phi)
    Lz:       Angular momentum array in z-direction with size (theta, r, phi)
    """

    Lr, _, Ltheta, Lphi = cart_to_sph(Lx, Ly, Lz)
    print(Lr.shape, Ltheta.shape, Lphi.shape)



    return Lr, Ltheta, Lphi


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
    ids = ~np.isnan(warp_dens)

    return warp_dens, ids



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

# Interpolate the densities & coordinates to the cell centres so that the array shape matches with mass & L
rho_c = centering(rho)
X_c = centering(X)
Y_c = centering(Y)
Z_c = centering(ZCYL)

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
# print(cell_volume.max(), cell_volume.min(), cell_volume.mean())
mass = calc_mass(rho, cell_volume)
Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)
print(Lx.shape, Ly.shape, Lz.shape)
Lrad, Lthe, Lphi = calc_L_average(Lx, Ly, Lz)

########################### Isolating the warp in the primary disk ###############################

threshold = -14   # log of density threshold for which we can see the warp in the primary
rho_warp, warp_ids = isolate_warp(rho_c, threshold)    # Note that I am using centered densities to match the indices corresponding to the warp with the angular momenta indices

# Plotting the warp densities 
fig = plt.figure(figsize=(10, 7))
contours_3D(X_c/au, Y_c/au, Z_c/au, rho_warp, fig, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

# Another way to plot the warp densities
# fig = plt.figure(figsize=(10, 7))
# contours_3D(X[warp_ids]/au, Y[warp_ids]/au, ZCYL[warp_ids]/au, rho[warp_ids], fig, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

# Plotting the Cartesian warp angular momenta
quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Lx[warp_ids], Ly[warp_ids], Lz[warp_ids], stagger=70, title="Warp angular momenta", colorbarlabel="logL", savefig=False, figfolder=None)

# Plotting the spherical warp angular momenta
# quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Lrad[warp_ids], Lthe[warp_ids], Lphi[warp_ids], stagger=100, title="Warp angular momenta", savefig=False, figfolder=None)

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

