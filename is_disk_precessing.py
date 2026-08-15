import numpy as np 
import matplotlib.pyplot as plt 
import astropy.constants as c

au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g

# Primary star and disk parameters
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
px = 0                    # Primary disk central X coordinate
py = 0                    # Primary disk central Y coordinate
pz = 0                    # Primary disk central Z coordinate
Rpi = 10 * au             # Primary disk inner radius [cm]
Rpo = 50 * au             # Primary disk outer radius [cm]

# Secondary star parameters
Rs = 1.11e16              # Secondary star orbital radius [cm]
Ms = 3.38e32              # Secondary star mass [g]
i = 1.0471975511965976    # Secondary star inclination with respect to primary [rad]

# Disk parameters
num_rings = 10
radii = np.linspace(10 * au, 60 * au, num_rings)       # Radial bins of disks
m0 = 1e28                                              # g, mass of innermost ring
k = 0.3                                                # exponential decay factor
ring_masses = m0 * np.exp(-k * np.arange(num_rings))   # Mass of each ring

phi_arr = np.linspace(0, 2 * np.pi, 100)       # Array of angular velocities
omega = 2 * np.pi / 10                         # One orbit every 10 seconds
time = np.linspace(0, 10, 500)                 # 10 seconds sampled at 500 points

# Orbit in the xy-plane (before inclination)
x = Rs * np.cos(omega * time)
y = Rs * np.sin(omega * time)
z = np.zeros_like(time)

# Rotation matrix to incline orbit about the x-axis
y_inc = y * np.cos(i) - z * np.sin(i)
z_inc = y * np.sin(i) + z * np.cos(i)

# Compute torque magnitude on each ring
# τ_i = (G * M * m_i * r_i^2 / R^3) * sin(2i)
torques = (G * Ms * ring_masses * radii**2 / Rs**3) * np.sin(2 * i)

# Optional: direction (unit vector of torque)
# Assuming orbit is inclined about x-axis: torque vector points in x-direction
torque_vectors = np.array([ [t, 0, 0] for t in torques ])

# Display torque magnitudes
for i, (r, m, t) in enumerate(zip(radii, ring_masses, torques)):
    print(f"Ring {i+1}: Radius = {r:.2e} m, Mass = {m:.2e} kg, Torque = {t:.2e} N·m")

# Plot the orbit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x/au, y_inc/au, z_inc/au, label=f'Orbit with {np.ceil(np.degrees(i))}° inclination')
ax.scatter(0, 0, 0, color='orange', label='Central Mass')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Inclined Orbit')
plt.show()

plt.figure()
plt.plot(radii / au, torques, marker='o')
plt.xlabel('Ring Radius [AU]')
plt.ylabel('Torque (N·m)')
plt.title('Torque on Disk Rings from Inclined Secondary Star')
plt.grid(True)
plt.show()

