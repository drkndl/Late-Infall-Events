import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import astropy.constants as c
from viewarr import *

au = c.au.cgs.value

quantity = "dens"
folder = Path("leon_snapshot/")
it = 600

domains_min = {}
real_to_fargo = {'phi': 'x', 'r':'y', 'theta':'z'}
for key in ["r", "phi", "theta"]:
    domains_min[key] = np.loadtxt(folder / f"domain_{real_to_fargo[key]}.dat")
    if key != "phi": domains_min[key] = domains_min[key][3:-3] # Ghost cells

domains = {}
for key in domains_min.keys():
    domains[key] = (domains_min[key][1:] + domains_min[key][:-1]) / 2

data = np.fromfile(folder / f"gas{quantity}{it}.dat").reshape(domains["theta"].size, domains["r"].size, domains["phi"].size)

# Slice in (r,theta)-plane
floor = 1e-22
slicearr(np.log10(data[::-1,:,:]+floor),indices=(1,0),x=np.log10(domains['r']),y=domains['theta'],zmin=None,zmax=None,idxnames=[r'$\pi-\theta$',r'$r$',r'$\phi$'],idxvals=None,idxformat='')

# Slice in (phi,theta)-plane
floor = 1e-22
slicearr(np.log10(data[::-1,:,:]+floor),indices=(2,0),x=domains['phi'],y=domains['theta'],zmin=None,zmax=None,idxnames=[r'$\pi-\theta$',r'$r$',r'$\phi$'],idxvals=None,idxformat='')
plt.ioff()
plt.show()