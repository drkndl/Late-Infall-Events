import matplotlib.pyplot as plt
import numpy as np
from viewarr import *
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
#from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as c
import imageio
import re
import os

au = c.au.cgs.value


def cyl_2D_plot(data, RCYL, ZCYL, irad, iphi, title, colorbarlabel, savefig, figfolder, showfig):
    """
    Plot 2D vertical projection of a physical quantity at a particular azimuth angle and range of radii
    
    Inputs:
    ------
    data:            3D array of physical quantity
    RCYL, ZCYL:      2D mesh of cylindrical coordinates' R and Z values
    irad:            Index of final radius to be plotted (int)
    iphi:            Index of azimuth angle to be plotted
    title:           Plot title (str)
    colorbarlabel:   Colour bar label
    savefig:         if True, image is saved (bool)
    figfolder:       Path where the image is to be saved (path)  
    
    Outputs:
    -------
    plot
    """

    plt.figure()
    plt.pcolormesh(RCYL[..., :irad, iphi]/au, ZCYL[..., :irad, iphi]/RCYL[..., :irad, iphi], np.log10(data[...,:irad, iphi]), cmap="Spectral_r", vmin=-19, vmax=-11, rasterized=True)
    plt.xlabel("rcyl / AU")
    plt.ylabel("z / r")
    plt.xscale("log")
    plt.ylim(-1,1)
    plt.title(title)
    plt.colorbar(label = colorbarlabel)
    # Save the figure?
    if savefig == True:
        plt.savefig(figfolder)

    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()


# 2D X-Y plot along the midplane
def XY_2D_plot(data, X, Y, irad, itheta, title, colorbarlabel, savefig, figfolder, showfig):
    """
    Plot 2D colormesh of a physical quantity along the X-Y plane for given polar angle theta and range of radii
    
    Inputs:
    ------
    data:            3D array of physical quantity
    X, Y:            3D mesh of X, Y coordinates
    irad:            Index of final radius to be plotted (int)
    itheta:          Index of polar angle to be plotted
    title:           Plot title (str)
    colorbarlabel:   Colour bar label
    savefig:         if True, image is saved (bool)
    figfolder:       Path where the image is to be saved (path)
    
    Outputs:
    -------
    plot
    """

    plt.figure()
    plt.pcolormesh(X[itheta, :irad, ...]/au, Y[itheta, :irad, ...]/au, np.log10(data[itheta, :irad, ...]), cmap="Spectral_r", vmin=-19, vmax=-11, rasterized=True)
    plt.gca().set_aspect("equal")
    plt.xlabel("x / AU")
    plt.ylabel("y / AU")
    plt.title(title)
    plt.colorbar(label = colorbarlabel)
    plt.tight_layout()

    # Save the figure?
    if savefig == True:
        plt.savefig(figfolder)

    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()


def interactive_2D(data, indices, x, y, idxnames):
    """
    Plots a 2D interactive slice from a 3D array using the slicearr() func from viewarr.py

    Inputs:
    ------
    data:         3D array of physical quantity
    indices:      Tuple of indices of data to be plotted 
    x:            x-axis array
    y:            y-axis array
    idxnames:     List of plot labels of each index of data
    """

    slicearr(data,indices=indices,x=x,y=y,zmin=-19,zmax=None,idxnames=idxnames,idxvals=None,idxformat='')
    plt.ioff()   # Added because interactive plot does not work properly otherwise
    plt.show()


def contours_3D(X, Y, Z, data, Rwarp, sim_params, colorbarlabel, title, savefig, figfolder, showfig=True, azim=-62, elev=-29):
    """
    Plot a 3D contour plot of a FARGO scalar field (dens/vx/energy etc) in Cartesian coords
    
    Inputs:
    ------
    X:                       3D array of Cartesian X meshgrid
    Y:                       3D array of Cartesian Y meshgrid
    Z:                       3D array of Cartesian Z meshgrid
    data:                    3D array of quantity to be visualized
    Rwarp:                   Radial extent of the warped disk
    sim_params:              Dictionary of simulation parameterss
    colorbarlabel:           Colour bar label (str)
    title:                   Image title (str)
    savefig:                 Boolean to save figure if True
    figfolder:               Folder in which to save the figure if savefig=True
    showfig:                 Boolean to show figure if True (default=True)
    azim:                    Azmiuthal camera angle for plot (Default=-62 for warp view) (degrees)
    elev:                    Elevation camera angle for plot (Default=-41 for warp view) (degrees)

    Issues:
    ------
    1. Too much whitespace around plot and labels
    """

    # ax = fig.add_subplot(111, figsize=(8,6), projection='3d')
    ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')

    p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=data.flatten(), cmap='plasma', s=7, edgecolor='none', alpha=0.5)

    # Colorbar formatting
    plt.colorbar(p, pad=0.08, label=colorbarlabel) #, shrink=0.85), fraction=0.046)

    # Adding a textbox to show simulation parameters
    param_text = '\n'.join(f'{key}: {value}' for key, value in sim_params.items())
    props = dict(boxstyle='round', facecolor='white', pad=0.6, alpha=0.3)   
    ax.text2D(0.9, 0.1, param_text, transform=ax.transAxes, fontsize=9, horizontalalignment='center', verticalalignment='center', bbox=props)

    # Plot formatting
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")
    ax.set_xlim(-Rwarp[-1]/au - 5, Rwarp[-1]/au + 5)
    ax.set_ylim(-Rwarp[-1]/au - 5, Rwarp[-1]/au + 5)
    ax.set_zlim(-30, 30)
    ax.set_title(title, pad=30)

    # Initial camera position of the 3D plot (default: elev=-41, azim=-62 for best view of warp)
    ax.view_init(elev=elev, azim=azim)   
    
    plt.tight_layout()

    # Save the figure?
    if savefig == True:
        plt.savefig(figfolder)

    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()


def quiver_plots(X, Y, v_x, v_y, itheta, irad, title, savefig, figfolder):
    """
    2D plots of vectors such as velocities

    Inputs:
    ------
    X:               3D array of Cartesian X meshgrid
    Y:               3D array of Cartesian Y meshgrid
    v_x:             3D array of Cartesian X velocities
    v_y:             3D array of Cartesian Y velocities
    title:           Plot title (str)
    irad:            Index of final radius to be plotted (int)
    itheta:          Index of polar angle to be plotted
    savefig:         if True, image is saved (bool)
    figfolder:       Path where the image is to be saved (path)
    """

    plt.figure(figsize=(10,8))
    vtot = np.log10(np.sqrt(v_x**2 + v_y**2))
    # vtot = np.sqrt(v_x**2 + v_y**2)

    # Note that I am plotting every second value in phi since otherwise, the plot gets too busy
    Q = plt.quiver(X[itheta, :irad, ::2]/au, Y[itheta, :irad, ::2]/au, v_x[itheta, :irad, ::2], v_y[itheta, :irad, ::2], vtot[itheta, :irad, ::2], cmap='viridis', edgecolor="black", alpha=0.6, angles='xy', scale_units='xy', pivot='tip', scale=10**3.2)
    plt.colorbar(Q, label='log(v)') 

    # Set labels and limits
    plt.gca().set_aspect("equal")
    plt.xlabel("x / AU")
    plt.ylabel("y / AU")
    plt.title(title)
    plt.tight_layout()
    if savefig == True:
        plt.savefig(figfolder)
    plt.show()


def interactive_interp_3d(data, Rmax, colorbarlabel, title, idxnames):
    """
    Plots data available in spherical coordinates on a Cartesian box that we can zoom in and out of

    Inputs:
    ------

    """
    # Model in (r,theta,phi)

    rin = 10
    rout= Rmax
    nr  = 250
    r   = rin * (rout/rin)**np.linspace(0,1,nr)

    th0 = 0.1
    th1 = np.pi-th0
    nt  = 100
    theta = np.linspace(th0,th1,nt)

    nph = 225
    phi = np.linspace(0,2*np.pi,nph)

    tt,rr, pp = np.meshgrid(theta,r,phi,indexing='ij')
    # Rmax = 3500.       # Maximum radius of the Cartesian box (AU)
    nx   = 200
    ny   = 202
    nz   = 204
    x    = np.linspace(-Rmax,Rmax,nx)
    y    = np.linspace(-Rmax,Rmax,ny)
    z    = np.linspace(-Rmax,Rmax,nz)
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    print(xx.shape, yy.shape, zz.shape)
    print(np.min(xx), np.max(xx))

    # Now compute r, theta and phi for each grid cell in the cartesian box
    r_box     = np.sqrt(xx**2+yy**2+zz**2)
    rc_box    = np.sqrt(xx**2+yy**2)
    theta_box = np.pi/2-np.arctan(zz/(rc_box+1e-99))
    phi_box   = np.arctan2(yy,xx)
    phi_box[phi_box<0]+=np.pi*2

    interp = RegularGridInterpolator((theta, r, phi), data, fill_value=0.,bounds_error=False)

    # Now map the model onto (x,y,z) box
    data_cart  = interp((theta_box, r_box, phi_box))

    # Now view
    slicearr(data_cart, indices=(1,0), idxnames=idxnames)
    plt.title(title)
    # plt.colorbar(label=colorbarlabel)
    plt.ioff()
    plt.show()
    slicearr(data_cart, indices=(0,2), idxnames=idxnames)
    plt.title(title)
    # plt.colorbar(label=colorbarlabel)
    plt.ioff()
    plt.show()


def quiver_plot_3d(X, Y, Z, dx, dy, dz, stagger, length, title, colorbarlabel, savefig, figfolder, showfig=True, ignorecol=False, logmag=True):
    """
    Plots quiver plot in 3D 
    
    Inputs:
    ------
    X:              Array of X coordinates of the arrows
    Y:              Array of Y coordinates of the arrows
    Z:              Array of Z coordinates of the arrows
    dx:             Array of X direction data of the arrows
    dy:             Array of Y direction data of the arrows
    dz:             Array of Z direction data of the arrows
    stagger:        Plot only every stagger points so that the plot doesn't look so busy
    length:         Length of the arrow
    title:          Plot title
    colorbarlabel:  Plot colorbar label
    savefig:        Boolean to save figure if True
    figfolder:      Directory in which the figure is to be saved
    showfig:        Boolean to show figure if True (default=True)
    ignorecols:     Boolean to remove arrow colours if True
    """

    ax = plt.figure(figsize=(6,6)).add_subplot(projection='3d')

    if logmag == True:
        # Map arrows to colormap according to the log of the magnitude of (dx, dy, dz)
        o = np.log10(np.sqrt(dx**2 + dy**2 + dz**2))
    else:
        o = np.sqrt(dx**2 + dy**2 + dz**2)
    # o = o.ravel()
    norm = colors.Normalize()
    norm.autoscale(o)
    cmap = cm.plasma
    
    # Define colour array
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot the arrows
    if ignorecol == True:
        Q = ax.quiver(X[::stagger, ::stagger, ::stagger], Y[::stagger, ::stagger, ::stagger], Z[::stagger, ::stagger, ::stagger], dx[::stagger, ::stagger, ::stagger], dy[::stagger, ::stagger, ::stagger], dz[::stagger, ::stagger, ::stagger], length=length, pivot='tip', alpha=0.8, arrow_length_ratio = 0.5, normalize=True) 
    else:
        print(X, Y, Z, dx, dy, dz)
        Q = ax.quiver(X[::stagger], Y[::stagger], Z[::stagger], dx[::stagger], dy[::stagger], dz[::stagger], length=length, pivot='tip', alpha=0.8, colors=cmap(norm(o)), arrow_length_ratio = 0.5, normalize=True) 
        plt.colorbar(sm, label=colorbarlabel)

    # Best initial camera projection to see the arrows properly 
    ax.view_init(elev=8, azim=-66)

    # Plot formatting
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")
    ax.set_title(title)
    plt.tight_layout()

    # Save the figure?
    if savefig:
        plt.savefig(figfolder)
    
    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()


def plot_surf_dens(X, Y, Z, surf_dens, warp_ids, r, savefig, figfolder, showfig):
    """
    X:            3D array of X coordinates
    Y:            3D array of Y coordinates
    Z:            3D array of Z coordinates
    surf_dens:    1D array of surface density at each radial point
    warp_ids:     IDs corresponding to the warp
    r:            1D array of radial points
    savefig:      Boolean to save figure if True
    figfolder:    Directory in which figure is to be saved
    showfig:      Boolean to show figure if True
    """

    r_warp_extent = np.sqrt(X[warp_ids]**2 +  Y[warp_ids]**2 + Z[warp_ids]**2) / au
    mask = (r/au >= r_warp_extent.min()) & (r/au <= r_warp_extent.max())
    r_selected = r[mask]
    surface_density_selected = surf_dens[mask]

    plt.plot(r_selected/au, surface_density_selected)
    plt.xlabel("R [AU]")
    plt.ylabel(r"$\Sigma$")
    plt.tight_layout()

    # Save the figure?
    if savefig:
        plt.savefig(figfolder)
    
    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()



def plot_twist_arrows(Lx_avg, Ly_avg, Lz_avg, R, Rwarp, sim_params, title, savefig, showfig, figfolder):   
    """
    Function to plot the warped disk twist/precession through a 3D quiver plot as a function of radius
    """

    # Obtain radius values for plotting
    Rc = 0.5 * (R[1:] + R[:-1])

    # Magnitude and XY projection of L(r)
    Lavg_mag = np.sqrt(Lx_avg**2 + Ly_avg**2 + Lz_avg**2)
    Lxy_proj = np.sqrt(Lx_avg**2 + Ly_avg**2)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')

    # Defining arrow colours according to twist value
    c = np.arccos(Lx_avg / Lxy_proj)
    
    # Define colour array
    # c = np.arccos(Lx_avg / Lxy_proj)
    c = np.concatenate((c, np.repeat(c, 2)))          # Repeat for each body line and two head lines of arrow
    c = plt.cm.viridis(c)                             # Colormap

    # Radially plotting averaged angular momenta (assuming R along X-axis and keeping Y- and Z- axes 0)
    # ax.quiver(Rc/au, 0, 0, Lx_avg, Ly_avg, Lz_avg, arrow_length_ratio=1, length=1, normalize=True, pivot='tip', color="black")          # Workaround to add arrowheads (fuck matplotlib 3D quiver plots)
    q = ax.quiver(Rc/au, 0, 0, Lx_avg, Ly_avg, Lz_avg, length=3, normalize=True, pivot='tip', color="black")
    # plt.colorbar(q, label=r"Normalized warp precession")

    # Overplotting the density as well
    # if plot_density:
    #     p = ax.scatter(X.flatten()/au, Y.flatten()/au, Z.flatten()/au, c=dens.flatten(), cmap='plasma', s=7, edgecolor='none', alpha=0.1)
    #     # Colorbar formatting
    #     plt.colorbar(p, pad=0.08, label=r'$\rho [g/cm^3]$') #, shrink=0.85), fraction=0.046)

    # Adding a textbox to show simulation parameters
    param_text = '\n'.join(f'{key}: {value}' for key, value in sim_params.items())
    props = dict(boxstyle='round', facecolor='white', pad=0.6, alpha=0.3)   
    ax.text2D(0.9, 0.9, param_text, transform=ax.transAxes, fontsize=9, horizontalalignment='center', verticalalignment='center', bbox=props)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_xlim(0, Rwarp.max()/au + 5)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_title(title)
    # ax.view_init(elev=36, azim=-68)
    ax.view_init(elev=71, azim=-179)
    # ax.view_init(elev=5, azim=0)
    # ax.view_init(elev=42, azim=7)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.tight_layout()
    
    # Save the figure?
    if savefig:
        plt.savefig(figfolder)
    
    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()


def plot_total_disks_bonanza(X, Y, Z, p_dens, s_dens, LX, LY, LZ, Ldx, Ldy, Ldz, sim_params, length, colorbarlabel, title, figfolder, azim=-62, elev=-41, savefig=False, showfig=True):
    """
    A plot of primary + secondary densities and their total disk angular momenta
    """

    ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')

    # Colorbar formatting for the densities
    # all_values = np.concatenate([p_dens, s_dens])
    all_values = p_dens
    vmin = all_values.min()
    vmax = all_values.max()
    norm = Normalize(vmin=all_values.min(), vmax=all_values.max())
    cmap = plt.cm.Spectral

    # Plotting the densities
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=p_dens.flatten(), cmap=cmap, s=7, edgecolor='none', alpha=0.3)
    # ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=s_dens.flatten(), cmap=cmap, s=7, edgecolor='none', alpha=0.3)

    # Plotting the angular momenta
    ax.quiver(LX, LY, LZ, Ldx, Ldy, Ldz, length=length, pivot='tail', alpha=0.8, color="black", arrow_length_ratio = 0.5, normalize=True)

    # Create a ScalarMappable for colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_clim(vmin, vmax)
    sm.set_array([])  # Dummy array for colorbar

    # Colorbar formatting
    plt.colorbar(sm, pad=0.08, label=colorbarlabel) #, shrink=0.85), fraction=0.046)

    # Adding a textbox to show simulation parameters
    # param_text = '\n'.join(f'{key}: {value}' for key, value in sim_params.items())
    # props = dict(boxstyle='round', facecolor='white', pad=0.6, alpha=0.3)   
    # ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9, horizontalalignment='center', verticalalignment='center', bbox=props)

    # Plot formatting
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)
    ax.set_title(title, pad=30)

    # Initial camera position of the 3D plot (default: elev=-41, azim=-62 for best view of warp)
    ax.view_init(elev=elev, azim=azim)   
    
    plt.tight_layout()

    # Save the figure?
    if savefig == True:
        plt.savefig(figfolder)

    # Display the figure?
    if showfig:
        plt.show()
    else:
        plt.close()



def extract_it_number(filename):
    """
    Sort the image files for GIF in order of iteration
    """

    match = re.search(r'_it(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # or raise an error if this shouldn't happen
    


def make_evol_GIF(directory, fname, gif_name, delete_files=True):
    """
    Creates a time evolution GIF out of images in a given directory

    Inputs:
    ------
    directory:      folder where the images are stored (str/Path)
    fname:          part of the name of the images (e.g. 'warp_dens_thresh' or 'warp_twist_arrow') (str)
    gif_name:       name of the output GIF (str)
    delete_files:   deletes images once GIF is made if True (default=True)
    """

    # Get all the files required to make the GIF, sorted by iteration number
    filenames = sorted([f for f in os.listdir(directory) if fname in f], key=extract_it_number)
    # filenames = sorted([f for f in os.listdir(directory) if fname in f])
    print(filenames)

    # Create GIF
    with imageio.get_writer(f'{directory}/{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(f'{directory}/{filename}')
            writer.append_data(image)

    # Remove image files from the folder
    if delete_files:
        for file in filenames:
            os.remove(f'{directory}/{file}') 

    return