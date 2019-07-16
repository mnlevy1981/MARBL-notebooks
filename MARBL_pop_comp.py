#!/usr/bin/env python

def compare_POP_and_MARBL():
    import xarray as xr
    import numpy as np
    import physics # local file with helper functions
    import shared_forcing

    import matplotlib
    import matplotlib.pyplot as plt

    grid_pts, ncol = shared_forcing.set_grid_points()

    ds_pop = shared_forcing.read_pop_data_set()
    ds_marbl = shared_forcing.read_marbl_data_set()

    comp_list = []
    for var in ds_marbl:
        if var not in ds_pop:
            print("{} is in MARBL data set but not POP".format(var))
        else:
            comp_list.append(var)

    for var in comp_list:
        if 'z_t_150m' in ds_pop[var].coords:
            print('Skipping {} due to dimension mis-match'.format(var))
            continue
        if var.startswith('TEND'):
            print('Skipping {} because MARBL and POP compute values very differently'.format(var))
            continue
        print_header = True
        for n, pt in enumerate(grid_pts.keys()):
            pop_data = ds_pop[var].isel(time=0, nlat=grid_pts[pt]['lat'], nlon=grid_pts[pt]['lon']).data
            marbl_data = ds_marbl[var].isel(num_cols=n).data
            abs_err = np.nanmax(np.abs(pop_data - marbl_data))
            denom = np.nanmax(np.abs(pop_data))
            if denom > 0:
                rel_err = abs_err/denom
                print_err = rel_err > 1e-13
            else:
                rel_err = 'NaN (POP is all zeroes)'
                print_err = abs_err > 0
            if print_err:
                if print_header:
                    print('\n{}\n----'.format(var))
                    print_header = False
                print('Col {} ({})\n  * max diff: {}\n  * rel diff: {}'.format(n+1, pt, abs_err, rel_err))

if __name__ == '__main__':
    compare_POP_and_MARBL()