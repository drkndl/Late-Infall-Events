import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from viewarr import *

# Model in (r,theta,phi)

rin = 1
rout= 10
nr  = 10
r   = rin * (rout/rin)**np.linspace(0,1,nr)

th0 = 0.1
th1 = np.pi-th0
nt  = 12
theta = np.linspace(th0,th1,nt)

nph = 14
phi = np.linspace(0,2*np.pi,nph)

rr,tt,pp = np.meshgrid(r,theta,phi,indexing='ij')

rho = np.exp(-(np.pi/2-tt)**2/2/0.3**2) * 1/rr

# Now make a (x,y,z) box

#size = 11.
size = 5.
nx   = 100
ny   = 102
nz   = 104
x    = np.linspace(-size,size,nx)
y    = np.linspace(-size,size,ny)
z    = np.linspace(-size,size,nz)
xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
#pts  = np.zeros((nx,ny,nz,3))
#pts[:,:,:,0] = xx
#pts[:,:,:,1] = yy
#pts[:,:,:,2] = zz

# Now compute r, theta and phi for each grid cell in the cartesian box

r_box     = np.sqrt(xx**2+yy**2+zz**2)
rc_box    = np.sqrt(xx**2+yy**2)
theta_box = np.pi/2-np.arctan(zz/(rc_box+1e-99))
phi_box   = np.arctan2(yy,xx)
phi_box[phi_box<0]+=np.pi*2

# Now set up the interpolation

interp = RegularGridInterpolator((r, theta, phi), rho, fill_value=0.,bounds_error=False)

# Now map the model onto (x,y,z) box

rho_cart  = interp((r_box,theta_box,phi_box))

# Now view

slicearr(rho_cart,indices=(0,1))
plt.ioff()
plt.show()
slicearr(rho_cart,indices=(1,2))
plt.ioff()
plt.show()