import matplotlib.pyplot as plt
import numpy as np
from viewarr import *
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d
#from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as c
au = c.au.cgs.value


def cyl_2D_plot(data, RCYL, ZCYL, irad, iphi, title, colorbarlabel, savefig, figfolder):
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
    plt.pcolormesh(RCYL[..., :irad, iphi]/au, ZCYL[..., :irad, iphi]/RCYL[..., :irad, iphi], np.log10(data[...,:irad, iphi]), cmap="viridis", vmin=-19, vmax=-11, rasterized=True)
    plt.xlabel("rcyl / AU")
    plt.ylabel("z / r")
    plt.xscale("log")
    plt.ylim(-1,1)
    plt.title(title)
    plt.colorbar(label = colorbarlabel)
    if savefig == True:
        plt.savefig(figfolder)
    plt.show()
    

# 2D X-Y plot along the midplane
def XY_2D_plot(data, X, Y, irad, itheta, title, colorbarlabel, savefig, figfolder):
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
    plt.pcolormesh(X[itheta, :irad, ...]/au, Y[itheta, :irad, ...]/au, np.log10(data[itheta, :irad, ...]), cmap="viridis", vmin=-19, vmax=-11, rasterized=True)
    plt.gca().set_aspect("equal")
    plt.xlabel("x / AU")
    plt.ylabel("y / AU")
    plt.title(title)
    plt.colorbar(label = colorbarlabel)
    plt.tight_layout()
    if savefig == True:
        plt.savefig(figfolder)
    plt.show()


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


def contours_3D(X, Y, Z, data, fig, xlabel, ylabel, zlabel, colorbarlabel, title):
    """
    Plot a 3D contour plot of a FARGO scalar field (dens/vx/energy etc) in Cartesian coords
    
    Inputs:
    ------
    X:                       3D array of Cartesian X meshgrid
    Y:                       3D array of Cartesian Y meshgrid
    Z:                       3D array of Cartesian Z meshgrid
    data:                    3D array of quantity to be visualized
    fig:                     Plot artist/class
    xlabel/ylabel/zlabel:    Axis labels (str)
    colorbarlabel:           Colour bar label (str)
    title:                   Image title (str)

    Issues:
    ------
    1. Too much whitespace around plot and labels
    """

    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=data.flatten(), cmap='plasma', s=7, edgecolor='none', alpha=0.3)

    # Colorbar formatting
    fig.colorbar(p, shrink=0.65, pad=0.04, label=colorbarlabel) # fraction=0.046

    # Plot formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Initial camera position of the 3D plot adjusted for best view of warp
    ax.view_init(elev=-41, azim=-62)   
        
    fig.tight_layout()
    plt.show()


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


def quiver_plot_3d(X, Y, Z, dx, dy, dz, stagger, title, colorbarlabel, savefig, figfolder):
    """
    Plots quiver plot in 3D 
    
    Inputs:
    ------
    stagger: Plot only every stagger points so that the plot doesn't look so busy
    """

    ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')

    # Map arrows to colormap according to the log of the magnitude of (dx, dy, dz)
    o = np.log10(np.sqrt(dx**2 + dy**2 + dz**2))
    norm = colors.Normalize()
    norm.autoscale(o)
    cmap = cm.plasma
    
    # Define colour array
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot the arrows
    Q = ax.quiver(X[::stagger], Y[::stagger], Z[::stagger], dx[::stagger], dy[::stagger], dz[::stagger], length=3, pivot='tip', alpha=0.8, color=cmap(norm(o)), arrow_length_ratio = 0.5, normalize=True)
    plt.colorbar(sm, label=colorbarlabel)

    # Best initial camera projection to see the arrows properly 
    ax.view_init(elev=8, azim=-66)

    # Plot formatting
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")
    ax.set_title(title)
    plt.tight_layout()
    if savefig == True:
        plt.savefig(figfolder)
    plt.show()
