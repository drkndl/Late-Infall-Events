import matplotlib.pyplot as plt
import numpy as np
from viewarr import *
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
#from matplotlib.ticker import LinearLocator
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
    Plot a 3D contour plot of a FARGO scalar field (dens/vx/enery etc) in Cartesian coords
    
    Inputs:
    ------
    X:                       3D array of Cartesian X meshgrid
    X:                       3D array of Cartesian X meshgrid
    X:                       3D array of Cartesian X meshgrid
    data:                    3D array of quantity to be visualized
    fig:                     Plot artist/class
    xlabel/ylabel/zlabel:    Axis labels (str)
    colorbarlabel:           Colour bar label (str)
    title:                   Image title (str)
    """

    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X[::5].flatten(), Y[::5].flatten(), Z[::5].flatten(), c=data[::5].flatten(), cmap='plasma', alpha=0.05)

    fig.colorbar(p, ax=ax, label=colorbarlabel)
    ax.set_xlabel(xlabel)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_zscale("log")
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def quiver_plots(X, Y, v_x, v_y, itheta, irad, title, savefig, figfolder):

    plt.figure(figsize=(10,8))
    vtot = np.log10(np.sqrt(v_x**2 + v_y**2))
    # vtot = np.sqrt(v_x**2 + v_y**2)
    Q = plt.quiver(X[itheta, :irad, ::2]/au, Y[itheta, :irad, ::2]/au, v_x[itheta, :irad, ::2], v_y[itheta, :irad, ::2], vtot[itheta, :irad, ::2], cmap='viridis', edgecolor="black", alpha=0.6, angles='xy', scale_units='xy', scale=10**3.2)
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