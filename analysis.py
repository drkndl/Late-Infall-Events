# Post-processing on FARGO3D output files 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import matplotlib.pyplot as plt
from no_thoughts_just_plots import quiver_plot_3d, contours_3D
import astropy.constants as c
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)


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
    Calculates the cell volumes of an (theta,r,phi) grid in cm^3 using the fomula for an elemental volume in 3D spherical coordinates dV = r^2 sin(theta) dr dphi dtheta
    
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


def isolate_warp(dens, vx, vy, vz, Lx, Ly, Lz, threshold):
    """
    A crude way to isolate the warp in the primary disk: applying a density threshold to capture the highest densities in the simulation output

    Inputs:
    ------
    dens:         3D array of centered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    threshold:    Threshold density by which we filter the dens array

    Outputs:
    -------
    warp_dens:    3D array of filtered density in g/cm^3 with shape (n_theta-1, n_r-1, n_phi-1)
    ids:          3D array of indices corresponding to warp_dens
    """

    # Filtering densities greater than a given threshold; values below the threshold are designated nan
    mask = dens > 10**threshold
    warp_dens = np.where(mask, dens, np.nan)
    warp_vx = np.where(mask, vx, np.nan)
    warp_vy = np.where(mask, vy, np.nan)
    warp_vz = np.where(mask, vz, np.nan)
    warp_Lx = np.where(mask, Lx, np.nan)
    warp_Ly = np.where(mask, Ly, np.nan)
    warp_Lz = np.where(mask, Lz, np.nan)

    # Also finding the corresponding x, y, z indices of the filtered densities
    ids = ~np.isnan(warp_dens)

    return warp_dens, warp_vx, warp_vy, warp_vz, warp_Lx, warp_Ly, warp_Lz, ids


def isolate_secondary_box(X, Y, Z, secX, secY, secZ, buffer, dens):
    """
    Isolates the secondary simply by zooming into the simulation box around (secX, secY, secZ) obtained by visualization

    Inputs:
    ------
    X: 
    Y: 
    Z: 
    """

    # Logical mask to select points within the box
    mask = (
        (X >= secX - buffer) & (X <= secX + buffer) &
        (Y >= secY - buffer) & (Y <= secY + buffer) &
        (Z >= secZ - buffer) & (Z <= secZ + buffer)
    )

    # Extract the density values within the box
    sec_dens = dens[mask]

    # Extract corresponding coordinates
    X_sec = X[mask]
    Y_sec = Y[mask]
    Z_sec = Z[mask]

    return X_sec, Y_sec, Z_sec, sec_dens, mask



def plot_L_average(Lx, Ly, Lz, R, savefig, it):
    """
    Calculates the angular momentum averaged across the theta and phi directions to find the radial Ls

    Inputs:
    ------
    Lx:       Angular momentum array in x-direction with size (theta, r, phi)
    Ly:       Angular momentum array in y-direction with size (theta, r, phi)
    Lz:       Angular momentum array in z-direction with size (theta, r, phi)
    """

    # Obtain radius values for plotting
    Rc = 0.5 * (R[1:] + R[:-1])

    # Angular momentum vectors for each shell L(r)
    Lx_avg = np.nansum(Lx, axis=(0,2))
    Ly_avg = np.nansum(Ly, axis=(0,2))
    Lz_avg = np.nansum(Lz, axis=(0,2))
    Lavg_mag = np.sqrt(Lx_avg**2 + Ly_avg**2 + Lz_avg**2)

    # Calculating warp twist
    twist_rad = np.arccos(Lx_avg / Lavg_mag)
    twist_deg = np.degrees(twist_rad)
    twist = np.nanmax(twist_deg) - np.nanmin(twist_deg)
    print("Twist: ", twist)

    plt.plot(Rc / au, twist_deg)
    plt.xlabel("R [AU]")
    plt.ylabel("Twist (deg)")
    plt.show()

    # Calculating warp inclination (Kimmig & Dullemond 2024)
    inc = np.arccos(Lz_avg / Lavg_mag)
    inc_deg = np.degrees(inc)
    print(np.nanmax(inc_deg), np.nanmin(inc_deg), np.nanmean(inc_deg))

    plt.plot(Rc / au, inc_deg)
    plt.xlabel("R [AU]")
    plt.ylabel("Inclination (deg)")
    plt.show()
    
    return 


folder = Path("leon_snapshot/")         # Folder with the output files
it = 600                                # FARGO snapshot

###################### Load data (theta = 100, r = 250, phi = 225) ################################


domains = get_domain_spherical(folder)
rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

# Cartesian velocities
vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)


############################# Calculate physical quantities ######################################


# Interpolate the densities & coordinates to the cell centres so that the array shape matches with mass & L
rho_c = centering(rho)
X_c = centering(X)
Y_c = centering(Y)
Z_c = centering(ZCYL)
vx_c = centering(vx)
vy_c = centering(vy)
vz_c = centering(vz)

cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
mass = calc_mass(rho, cell_volume)
Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)
Ax, Ay, Az = calc_LRL(mass, Mstar, vx_c, vy_c, vz_c, Lx, Ly, Lz, X_c, Y_c, Z_c)
ex, ey, ez = calc_eccen(Ax, Ay, Az, mass, Mstar)
e = np.sqrt(ex**2 + ey**2 + ez**2)


########################### Isolating the warp in the primary disk ###############################


# Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
# Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
warp_thresh = -14   # log of density threshold for which we can see the warp in the primary
rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_warp(rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 
print(rho_c_warp.shape, vx_c_warp.shape, vy_c_warp.shape, Lx_c_warp.shape, Ly_c_warp.shape)

#################################### Plotting warp properties #####################################


# Plotting the warp densities 
# contours_3D(X_c/au, Y_c/au, Z_c/au, rho_c_warp, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{warp_thresh}}} g/cm^3$', savefig=False, figfolder=f'../warp_dens_thresh{warp_thresh}_it{it}.png')

# Another way to plot the warp densities
# contours_3D(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, rho_c[warp_ids], fig, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\log(\rho)$ above $\rho = 10^{{{threshold}}} g/cm^3$')

# Plotting the Cartesian warp angular momenta
# quiver_plot_3d(X_c/au, Y_c/au, Z_c/au, Lx_c_warp, Ly_c_warp, Lz_c_warp, stagger=5, length=2, title="Warp angular momenta 2", colorbarlabel="logL", savefig=False, figfolder=f'../warp_L_thresh{warp_thresh}_it{it}.png', logmag=True, ignorecol=True)

# Another way to plot the Cartesian warp angular momenta
# quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Lx[warp_ids], Ly[warp_ids], Lz[warp_ids], stagger=70, length=3, title="Warp angular momenta", colorbarlabel="logL", savefig=False, figfolder=f'../warp_L_thresh{warp_thresh}_it{it}.png', logmag=True)

# Plotting the radially averaged Cartesian warp angular momenta and the 2D projections
plot_L_average(Lx_c_warp, Ly_c_warp, Lz_c_warp, domains["r"], True, it)

# Plotting radially averaged L for ALL radii (a.k.a a mess)
# plot_L_average(Lx, Ly, Lz, X_c, Y_c, Z_c, rho_c)

# Plotting warp Laplace-Runge-Lenz vector
# quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, Ax[warp_ids], Ay[warp_ids], Az[warp_ids], stagger=70, length=3, title=rf'Warp LRL', colorbarlabel=r'$\log(A [g^2cm^3/s^2])$', savefig=True, figfolder=f'../warp_{it}_LRL.png', logmag=True)

# Plotting warp eccentricity
# quiver_plot_3d(X_c[warp_ids]/au, Y_c[warp_ids]/au, Z_c[warp_ids]/au, ex[warp_ids], ey[warp_ids], ez[warp_ids], stagger=70, length=3, title=rf'Warp Eccentricity', colorbarlabel=r'$e$', savefig=True, figfolder=f'../warp_{it}_ecc.png', logmag=False)

# Characterizing warp eccentricity 
# print("Min Warp eccentricity: ", np.min(e[warp_ids]))
# print("Max Warp eccentricity: ", np.max(e[warp_ids]))
# print("Mean warp eccentricity: ", np.mean(e[warp_ids]))

no_secondary_all_my_homies_hate_secondary

######## Isolating the secondary disk box: (X=-150AU,Y=300,Z=600) (R=?,theta=30°,phi=115°) ##########

# Roughly the Cartesian central coordinate of the secondary disk obtained through visualization
secx = -150       # AU
secy = 300        # AU
secz = 600        # AU
buffer = 100      # AU

# Isolating the secondary box and its corresponding density
X_secbox, Y_secbox, Z_secbox, rho_c_secbox, secbox_ids = isolate_secondary_box(X_c, Y_c, Z_c, secx * au, secy * au, secz * au, buffer * au, rho_c)

# Plotting the secondary densities box 
contours_3D(X_secbox/au, Y_secbox/au, Z_secbox/au, rho_c_secbox, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\rho$ Secondary Disk Box', savefig=True, figfolder=f'../secondary_dens_box.png')

# Now applying the density threshold to the secondary box to get the secondary disk itself
sec_thresh = -15.5   # log of density threshold for which we can see the secondary
rho_c_sec, sec_ids = isolate_warp(rho_c_secbox, sec_thresh)

# Plotting the secondary densities
contours_3D(X_secbox/au, Y_secbox/au, Z_secbox/au, rho_c_sec, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'$\rho$ Secondary Disk', savefig=True, figfolder=f'../secondary_dens_thresh{sec_thresh}.png')

# Plotting the Cartesian secondary disk angular momenta
Lx_secbox, Ly_secbox, Lz_secbox = Lx[secbox_ids], Ly[secbox_ids], Lz[secbox_ids]
quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, Lx_secbox[sec_ids], Ly_secbox[sec_ids], Lz_secbox[sec_ids], stagger=1, length=10, title="Secondary disk angular momenta", colorbarlabel="logL", savefig=True, figfolder=f'../secondary_L_thresh{sec_thresh}.png', logmag=True)

# Plotting secondary disk Laplace-Runge-Lenz vector
Ax_secbox, Ay_secbox, Az_secbox = Ax[secbox_ids], Ay[secbox_ids], Az[secbox_ids]
quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, Ax_secbox[sec_ids], Ay_secbox[sec_ids], Az_secbox[sec_ids], stagger=1, length=10, title=rf'Secondary disk LRL', colorbarlabel=r'$\log(A [g^2cm^3/s^2])$', savefig=True, figfolder=f'../secondary_LRL_thresh{sec_thresh}.png', logmag=True)

# Plotting secondary disk eccentricity
ex_secbox, ey_secbox, ez_secbox = ex[secbox_ids], ey[secbox_ids], ez[secbox_ids]
quiver_plot_3d(X_secbox[sec_ids]/au, Y_secbox[sec_ids]/au, Z_secbox[sec_ids]/au, ex_secbox[sec_ids], ey_secbox[sec_ids], ez_secbox[sec_ids], stagger=1, length=10, title=rf'Secondary disk Eccentricity', colorbarlabel=r'$\log(e)$', savefig=True, figfolder=f'../secondary_ecc_thresh{sec_thresh}.png', logmag=False)

# Characterizing secondary disk eccentricity 
e_secbox = e[secbox_ids]
print("Min Warp eccentricity: ", np.min(e_secbox[sec_ids]))
print("Max Warp eccentricity: ", np.max(e_secbox[sec_ids]))
print("Mean Warp eccentricity: ", np.mean(e_secbox[sec_ids]))


####################################################################################################

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

