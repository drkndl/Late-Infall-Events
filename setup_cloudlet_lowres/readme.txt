This is a test run on the BinAC cluster with low resolution. 
Changes made to setup=cloudlet:

iras04125_c7_highmass_nograv_balanced.par
-----------------------------------------

### Mesh parameters

Nx        80                         Azimuthal number of zones 
Ny        75                         Radial number of zones
Nz        105                        Number of zones in colatitude

### Output control parameters

Ntot          60000                            Total number of time steps (35.578 kyr)


cloudlet.par
------------

### Mesh parameters

Nx        80                          Azimuthal number of zones 
Ny        75                          Radial number of zones
Nz        105                         Number of zones in colatitude
Ymin      1.496e+14                   Inner boundary radius


cps calc:
--------
r	   0.5946219743854434
phi	   0.7308032544688762
theta	   1.9183585429808

https://rometsch.github.io/cps-calc/?grid-type=spherical&spacing-type=log&name=IRAS04152_lowRes&x1N=75&x1min=10&x1max=10000&x1extent=9990&x2N=80&x2min=-3.14&x2max=3.14&x2extent=6.28&x3N=105&x3max=3.14&x3extent=3.14&aspect-ratio=0.03799&flaring-index=0.25&radius=5.2  

