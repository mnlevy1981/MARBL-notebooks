"""
Python module to ensure Forcing Data and Comparison notebooks
Are getting data from the same grid points of the same cases
"""

def set_grid_points():
    grid_pts = {}
    # Pick a (lat, lon) for gx1v7 grid
    # We want forcing from columns that meet the following criteria:
    # (a) OMZ column (eq. Pacific)
    #    (230, 264) is in oxygen minimum zone in eastern equatorial Pacific (257.6 E, 11.5 N); has 50 levels
    grid_pts['OMZ'] = {'lat' : 230, 'lon' : 264}
    # (b) multiple PAR columns (Southern Ocean)
    #    (37, 157) has large QSW_BIN_06 (137.2 E, 59.5 S); has 56 levels
    grid_pts['5 PAR subcols'] = {'lat' : 37, 'lon' : 157}
    # (c) Arctic [darkness on Jan 1]
    #     (174, 365) has no short-wave heat flux in January  (217 E, 87.3 N); has 50 levels
    grid_pts['no sun'] = {'lat' : 365, 'lon' : 174}
    # (d) Subtropical Atlantic (high Fe)
    #     (20, 268) is off African coast (17 E, 25 N); has 46 levels
    grid_pts['high Fe'] = {'lat' : 268, 'lon' : 20}
    # grid_pts['high Fe (orig)'] = {'lat' : 294, 'lon' : 288} # Mouth of Chesapeake Bay
    # grid_pts['high Fe 2'] = {'lat' : 223, 'lon' : 21} # Off tropical African coast
    # grid_pts['high Fe 3'] = {'lat' : 231, 'lon' : 17} # Off tropical African coast
    # (e) Small KMT with biomass to trigger sediment burial
    #     (148, 158) is shallow (138.3 E, -10.3 N); has 5 levels
    grid_pts['shallow'] = {'lat' : 148, 'lon' : 158}
    return grid_pts, len(grid_pts.keys())

def read_pop_data_set(second_step=False):
    import os
    import xarray as xr

    # gx1v7 output; nstep averages
    filename = os.path.join(os.path.expanduser('~/'), 'MARBL_data', 'g.e22b2.G1850ECO.T62_g17.gen_single_col.nstep_out.nc')
    if not os.path.isfile(filename):
        casename='g.e22b2.G1850ECO.T62_g17.gen_single_col.nstep_out'
        rundir=os.path.join(os.path.sep, 'glade', 'scratch', 'mlevy', casename, 'run')
        if second_step:
            date='0001-01-01-14400'
        else:
            date='0001-01-01-07200'
        filename = os.path.join(rundir, '%s.pop.h.%s.nc' % (casename, date))

    if os.path.isfile(filename):
        print("Opening %s" % filename)
        # Workaround for https://github.com/pydata/xarray/issues/1576
        ds = xr.open_dataset(filename, decode_cf=False)
        del(ds['KMT'].attrs['_FillValue'])
        del(ds['KMT'].attrs['missing_value'])
        ds = xr.decode_cf(ds, decode_times=False, decode_coords=False)
        return ds

    raise FileNotFoundError("Can not find {}".format(filename))

def read_marbl_data_set():
    import os
    import xarray as xr

    # gx1v7 output; nstep averages
    filename = os.path.join(os.path.expanduser('~/'), 'codes', 'MARBL', 'tests', 'regression_tests', 'compute_cols', 'history_1inst.nc')
    if os.path.isfile(filename):
        print("Opening %s" % filename)
        # Workaround for https://github.com/pydata/xarray/issues/1576
        ds = xr.open_dataset(filename, decode_cf=True)
        return ds

    raise FileNotFoundError("Can not find {}".format(filename))

