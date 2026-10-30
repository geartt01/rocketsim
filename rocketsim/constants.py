"""Physical constants, SI units.

Sources: ISO 2533:1975 (Standard Atmosphere); U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562).
"""

G0 = 9.80665  # standard gravity, m/s^2
R_AIR = 287.05287  # specific gas constant of dry air, J/(kg·K)
GAMMA_AIR = 1.4  # ratio of specific heats of air, -
R_EARTH = 6356766.0  # Earth radius for geopotential altitude, m (US Std Atm 1976)

# ISA sea level and layers
T0 = 288.15  # sea-level temperature, K
P0 = 101325.0  # sea-level pressure, Pa
LAPSE_TROPO = -0.0065  # troposphere temperature gradient, K/m
H_TROPOPAUSE = 11000.0  # top of troposphere, geopotential m

# Sutherland's law for air: mu = BETA * T^1.5 / (T + S)
SUTHERLAND_BETA = 1.458e-6  # kg/(m·s·K^0.5)
SUTHERLAND_S = 110.4  # K
