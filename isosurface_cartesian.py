import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
from analysis import sph_to_cart, vel_sph_to_cart, calc_cell_volume, calc_mass
from no_thoughts_just_plots import interactive_2D, contours_3D, cyl_2D_plot, XY_2D_plot, quiver_plots, interactive_interp_3d
import astropy.constants as c
au = c.au.cgs.value
from scipy.interpolate import RegularGridInterpolator
from viewarr import *


#folder  = Path("../cloud_disk_it450_cmass10_Rout30_rotX45/")         # Folder with the output files
folder  = Path("../cloud_disk_it450_Rout30_rotX45/")         # Folder with the output files
it      = 450                                             # FARGO snapshot

domains = get_domain_spherical(folder)
theta   = domains['theta']
r       = domains['r']
phi     = domains['phi']

rho     = get_data(folder, "dens", it, domains)         # Load 3D array of density values 

def interp_3d(data,theta,r,phi,x,y,z,fillcenter=0.0):
    """
    Map data from the 3D spherical grid onto a cartesian regular gridded box.

    Arguments:

      data        The 3D data array data[theta,r,phi]
      theta       The 1D polar grid
      r           The 1D radial grid
      phi         The 1D azimuthal grid
      x           A 1D array of the x-coordinate
      y           A 1D array of the y-coordinate
      z           A 1D array of the z-coordinate

    Returns:

      data_cart   The mapped data onto the cartesian coordinates
    """
    
    nth,nr,nph = data.shape
    assert nth==len(theta)
    assert nr==len(r)
    assert nph==len(phi)
    
    # To ensure that we have no "empty wedge" near the extrema of phi:
    
    data_ext           = np.zeros((nth,nr,nph+2))
    data_ext[:,:,1:-1] = data[:,:,:]
    data_ext[:,:,0]    = data_ext[:,:,-2]
    data_ext[:,:,-1]   = data_ext[:,:,1]
    phi_ext            = np.zeros(nph+2)
    phi_ext[1:-1]      = phi[:]
    phi_ext[0]         = phi_ext[-2] - 2*np.pi
    phi_ext[-1]        = phi_ext[1]  + 2*np.pi
    
    # Setting up the cartesian box
    
    xx,yy,zz           = np.meshgrid(x,y,z,indexing='ij')
    
    # Now compute r, theta and phi for each grid cell in the cartesian box
    
    r_box              = np.sqrt(xx**2+yy**2+zz**2)
    rc_box             = np.sqrt(xx**2+yy**2)
    theta_box          = np.pi/2-np.arctan(zz/(rc_box+1e-99))
    phi_box            = np.arctan2(yy,xx)
    #phi_box[phi_box<0]+=np.pi*2

    # Now set up the interpolation

    interp             = RegularGridInterpolator((theta, r, phi_ext), data_ext, fill_value=0.,bounds_error=False)
    
    # Now map the model onto (x,y,z) box

    data_cart          = interp((theta_box,r_box,phi_box))

    # For cleanliness, put values values inside the inner radius to a constant value

    mask               = r_box<r.min()
    data_cart[mask]    = fillcenter

    return data_cart

#rmax = r.max() * 1     # Full scale
rmax = r.max() * 0.1  # Moderate zoom-in
#rmax = r.max() * 0.01  # Strong zoom-in
nx   = 100
ny   = 102   # Using ny!=nx!=nz to make it easier to figure out what is x, what is y and what 
nz   = 104   # is z. Later they can be made equal.
x    = np.linspace(-rmax,rmax,nx)
y    = np.linspace(-rmax,rmax,ny)
z    = np.linspace(-rmax,rmax,nz)

rho_cart = interp_3d(rho,theta,r,phi,x,y,z)

interactive_2D(np.log10(rho_cart+1e-22), [r"Time", r'$\phi$ [deg]'], title="blah", indices)
