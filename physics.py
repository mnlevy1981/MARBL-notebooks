"""
Python module containing functions to convert from one physical quantity to another.
E.g. depth (m) -> pressure (bars)
"""

import numpy as np

def depth_to_pressure(depth):
    """
    ! !DESCRIPTION:
    !  This function computes pressure in bars from depth in meters
    !  using a mean density derived from depth-dependent global
    !  average temperatures and salinities from Levitus 1994, and
    !  integrating using hydrostatic balance.
    !
    !  References:
    !
    !     Levitus, S., R. Burgett, and T.P. Boyer, World Ocean Atlas
    !          1994, Volume 3: Salinity, NOAA Atlas NESDIS 3, US Dept. of
    !          Commerce, 1994.
    !
    !     Levitus, S. and T.P. Boyer, World Ocean Atlas 1994,
    !          Volume 4: Temperature, NOAA Atlas NESDIS 4, US Dept. of
    !          Commerce, 1994.
    !
    !     Dukowicz, J. K., 2000: Reduction of Pressure and Pressure
    !          Gradient Errors in Ocean Simulations, J. Phys. Oceanogr.,
    !          submitted.
    """
    # Taken from pressure() in POP's state_mod.F90
    # Note that this function returns value in units of atm rather than bars
    # [consistent with units of atmospheric pressure]
    return(0.059808*(np.exp(-0.025*depth)-1.) + 0.100766*depth + 2.28405e-7*depth*depth)

def xkw_to_u10sqr(xkw): # units of xkw should be cm/s
    # in marbl_diagnostics_mod.F90: diags(ind_diag%ECOSYS_XKW)%field_2d(:) = piston_velocity(:)
    # in marbl_mod.F90:             piston_velocity = xkw_coeff*u10_sqr(:)
    # in marbl_settings_mod.F90:    xkw_coeff =   6.97e-9_r8 ! in s/cm, from a = 0.251 cm/hr s^2/m^2 in Wannikhof 2014
    xkw_coeff = 6.97e-9 # s/cm
    # Note that this function returns value in units m^/s^2 rather than cm^2/s^2
    return((xkw/xkw_coeff)*0.0001)

def get_surface_value(data_in, column_var):
    # first just populate it with ice_frac (another scalar) to get all metadata
    new_da = data_in['ice_frac'].copy(deep=True)
    new_da.values = data_in[column_var].values[0]
    new_da.attrs['long_name'] = 'Sea Surface {}'.format(data_in[column_var].attrs['long_name'])
    new_da.attrs['units'] = data_in[column_var].attrs['units']
    try:
        new_da.encoding['scale_factor'] = data_in[column_var].encoding['scale_factor']
    except:
        pass
    return(new_da)