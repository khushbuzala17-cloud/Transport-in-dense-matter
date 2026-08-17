import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


# Constants in CGS units
G = 6.67430e-8  # Gravitational constant (cm^3 g^-1 s^-2)
c = 2.99792458e10  # Speed of light (cm/s)
solar_mass = 1.989e33  # Solar mass (g)


# Load EoS from Excel file
eos_file = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/final_eos_solved_output_NJL.xlsx"
sheet_name = "eta_0.9_NJL"      # Change to your required sheet

eos_data = pd.read_excel(eos_file, sheet_name=sheet_name)


# Extract pressure and energy density
pressure = eos_data["P_total"].values
epsilo = eos_data["eps_total"].values


epsilon = epsilo * 1.78266192e12  # Convert rho to g/cm^3
p = pressure * 1.60218e33  # Convert P to dyn/cm^2


# Interpolation function (no extrapolation)
pressure_to_energy_density = interp1d(p, epsilon, kind="linear", bounds_error=True)


def tov_equations_dimensional(r, state):
    P, m = state
    if r < 1e-5 or m < 1e-6:
        r = max(r, 1e-5)
        m = max(m, 1e-6)

    if P <= 0:
        return [0.0, 0.0]

    try:
        epsilon_val = pressure_to_energy_density(P)
    except ValueError:
        return [0.0, 0.0]
    
    dPdr = - (G * (epsilon_val + P / c**2) * (m + 4 * np.pi * r**3 * P / c**2)) / \
           (r * (r - 2 * G * m / c**2))
    dmdr = 4 * np.pi * r**2 * epsilon_val

    print(f"Radius: {r:.2e} cm, Pressure: {P:.2e} dyn/cm², Mass: {m:.2e} g, Energy Density (epsilon): {epsilon_val:.2e} g/cm³")

    return [dPdr, dmdr]


def stop_integration_out_of_range(r, state):
    P, _ = state
    if P < min(p) or P > max(p):
        return 0
    return 1


stop_integration_out_of_range.terminal = True  
stop_integration_out_of_range.direction = -1


def find_ns_radius(pressure, radius_array):
    mask = (pressure <= 0).astype(int)
    index = np.argmax(mask)
    if index == 0:
        return radius_array[-1], len(radius_array) - 1
    return radius_array[index], index


# Central pressure
P_central = 2.1050e35 
initial_conditions = [P_central, 0.0]

r_array = np.linspace(1e-6, 20e5, 5000)

try:
    sol = solve_ivp(
        tov_equations_dimensional,
        [r_array[0], r_array[-1]],
        initial_conditions,
        method='RK45',
        events=stop_integration_out_of_range,
        dense_output=True,
        atol=1e-6,
        rtol=1e-6
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    Pressure, Mass = sol.y
    radius_solver = sol.t

    radius_ns, index_ns = find_ns_radius(Pressure, radius_solver)

    radius_solver = radius_solver[:index_ns + 1]
    Pressure = Pressure[:index_ns + 1]
    Mass = Mass[:index_ns + 1]

except RuntimeError as e:
    print(f"Solver failed: {e}")
    Pressure = sol.y[0] if 'sol' in locals() else [initial_conditions[0]]
    Mass = sol.y[1] if 'sol' in locals() else [initial_conditions[1]]
    radius_solver = sol.t if 'sol' in locals() else [r_array[0]]

    radius_ns = radius_solver[-1]
    mass_ns = Mass[-1]

    print(f"Stellar parameters at failure:\n"
          f"Radius: {radius_ns / 1e5:.2f} km\n"
          f"Mass: {mass_ns / solar_mass:.2f} M_sun")

    radius_solver = np.array(radius_solver)


# Convert units
radius_km = radius_solver / 1e5
mass_solar = Mass / solar_mass


# ======================================================
# ===== SAVE TO EXCEL WITH MeV/fm^3 UNITS ==============
# ======================================================

# Compute energy density profile matching Pressure array (in CGS)
epsilon_profile_cgs = []
for P in Pressure:
    try:
        epsilon_profile_cgs.append(pressure_to_energy_density(P))
    except:
        epsilon_profile_cgs.append(np.nan)

# Convert to MeV/fm^3 for output
pressure_MeVfm3 = Pressure / 1.60218e33
epsilon_MeVfm3 = np.array(epsilon_profile_cgs) / 1.78266192e12

# Create table with MeV/fm^3 units
output_df = pd.DataFrame({
    "Radius (km)": radius_km,
    "r_cm": radius_km*(1e5),
    "Pressure (MeV/fm^3)": pressure_MeVfm3,
    "Energy Density (MeV/fm^3)": epsilon_MeVfm3,
    "rho (g/cm^3)": epsilon_MeVfm3*(1.78266192e12),
    "Mass (g)": Mass,
    "Mass (M_sun)": mass_solar
})

# File path
output_file = r"/home/khushbu/Desktop/khushbu/phD/EOS codes/max+gibs(paper)/plots/ MR_NJL.xlsx"
output_sheet = "MR_(eta = 0.9))"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    output_df.to_excel(writer, sheet_name=output_sheet, index=False)

print(f"\nData saved successfully to:\n{output_file}\n")


# ======================================================



# Plotting (still uses CGS units for consistency)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(radius_km, Pressure, color='blue', label='Pressure')
axes[0].set_xlabel('Radius (km)')
axes[0].set_ylabel('Pressure (dyn/cm²)')
axes[0].legend()

axes[1].plot(radius_km, mass_solar, color='red', label='Mass')
axes[1].set_xlabel('Radius (km)')
axes[1].set_ylabel('Mass (M_sun)')
axes[1].legend()

print(f"Stellar parameters (physical units):\nRADIUS: {radius_km[-1]:.2f} km\nMASS: {mass_solar[-1]:.2f} M_sun")

plt.tight_layout()
plt.show()
