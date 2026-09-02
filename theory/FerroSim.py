import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ==========================================
# 1. INPUT PARAMETERS (From custom fluid data)
# ==========================================
# Particles
rho_p = 5323.0          # Particle density (kg/m^3)
vol_frac = 0.0914       # Inlet volume fraction
M_sat_p = 265000.0      # Saturation magnetization of particles (A/m)

# Carrier Fluid (65% Undecane, 35% Hexane approximate values)
rho_undecane = 730.0
rho_hexane = 659.0
sigma_undecane = 1.88e-2 # N/m
sigma_hexane = 1.17e-2   # N/m

# Constants
g = 9.81                # Gravity (m/s^2)
mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability (T.m/A)

# ==========================================
# 2. DERIVED BULK PROPERTIES
# ==========================================
# Base fluid approximations via mass composition
rho_base = (0.65 * rho_undecane) + (0.35 * rho_hexane)
gamma_base = (0.65 * sigma_undecane) + (0.35 * sigma_hexane)

# Bulk Ferrofluid Properties
rho_bulk = (vol_frac * rho_p) + ((1 - vol_frac) * rho_base)
M_sat_bulk = vol_frac * M_sat_p

print(f"Calculated Bulk Density: {rho_bulk:.2f} kg/m^3")
print(f"Calculated Effective Surface Tension: {gamma_base:.4f} N/m")

# ==========================================
# 3. CRITICAL INSTABILITY THRESHOLDS
# ==========================================
# Critical Wavenumber (k_c) and Wavelength (lambda_c)
# k_c = sqrt(rho * g / gamma)
k_c = np.sqrt((rho_bulk * g) / gamma_base)
lambda_c = (2 * np.pi) / k_c

print(f"Critical Wavenumber (k_c): {k_c:.2f} rad/m")
print(f"Predicted Inter-spike Distance: {lambda_c * 1000:.2f} mm")

# Critical Magnetization Threshold
# M_c^2 = (2 / mu_0) * sqrt(rho * g * gamma)
# Assuming high susceptibility for the onset threshold approximation
M_c = np.sqrt((2 / mu_0) * np.sqrt(rho_bulk * g * gamma_base))
print(f"Critical Magnetization (M_c): {M_c:.2f} A/m")

# ==========================================
# 4. 3D SURFACE TOPOLOGY GENERATION
# ==========================================
# Generate a mesh grid matching a typical petri dish field of view (e.g., 40x40mm)
x = np.linspace(-0.02, 0.02, 500)
y = np.linspace(-0.02, 0.02, 500)
X, Y = np.meshgrid(x, y)

# The Rosensweig instability forms a hexagonal lattice pattern.
# We model this by superimposing three plane waves oriented at 120 degrees.
theta1 = 0
theta2 = 2 * np.pi / 3
theta3 = 4 * np.pi / 3

# Amplitude of spikes (this scales non-linearly with applied field H > H_c)
# For the theoretical model, we set an arbitrary small amplitude for the linear regime
A = 0.002 # 2mm spike height 

# Hexagonal surface function Z(x,y)
Z = A * (
    np.cos(k_c * (X * np.cos(theta1) + Y * np.sin(theta1))) +
    np.cos(k_c * (X * np.cos(theta2) + Y * np.sin(theta2))) +
    np.cos(k_c * (X * np.cos(theta3) + Y * np.sin(theta3)))
)

# Optional: Add a macroscopic macroscopic Gaussian deformation 
# caused by the field gradient of a central cylindrical magnet
magnet_gradient = np.exp(-(X**2 + Y**2) / (0.01**2)) 
Z += (0.001 * magnet_gradient)

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X * 1000, Y * 1000, Z * 1000, 
                       cmap=cm.inferno, 
                       linewidth=0, antialiased=True)

ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Spike Height Z (mm)')
ax.set_title('Theoretical Rosensweig Instability Topology')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()