#!/usr/bin/env python

"""
    Generate weights for linear regression based on POP history file and region mask

    1. Initially just mimic what is done in Loess step by step notebook
    2. Rather than taking the "n nearest points", look for all neighboring points within
       some pre-defined area (maybe elipse parameterized by x- and y-axis lengths)
    3. Use region mask: don't allow non-zero weights from Pacific for points in Atlantic
       (and vice versa)
"""

import numpy as np
import xarray as xr
import dask.bag

class LoessWeightGenClass(object): # pylint: disable=useless-object-inheritance,too-few-public-methods
    """
    A class to wrap function calls required to generate weights for LOESS regression
    """
    def __init__(self, hist_file, mask_file):
        self._read_files(hist_file, mask_file)
        self._gen_weights()
        # self.to_netcdf()

    #####################

    def _read_files(self, hist_file, mask_file):
        # Read netcdf file
        self._hist_ds = xr.open_dataset(hist_file, decode_times=False).isel(time=0)
        # Read binary file: >i4 => big-endian (>) and 4-byte integer (i4)
        self._region_mask = np.fromfile(mask_file, dtype=">i4")

    #####################

    def _gen_weights(self, ndims=3, num_nearest_pts=384): # pylint: disable=too-many-locals
        """
            Main subroutine to generate weights for linear regression step of LOESS

            Region masks come from POP log:

                                          +-----------------------+
                                          |  Marginal Seas Only   |
              Region        Region        |  Area        Volume   |
              Number        Name          | (km^2)       (km^3)   |
              ------ -------------------  -----------  -----------
                 1        Southern Ocean
                 2         Pacific Ocean
                 3          Indian Ocean
                -4          Persian Gulf  3.21703E+05  2.57362E+04
                -5               Red Sea  5.55613E+05  5.56528E+04
                 6        Atlantic Ocean
                 7     Mediterranean Sea
                 8          Labrador Sea
                 9               GIN Sea
                10          Arctic Ocean
                11            Hudson Bay
               -12            Baltic Sea  3.70458E+05  1.74489E+04
               -13             Black Sea  4.50883E+05  3.29115E+05
               -14           Caspian Sea  2.50748E+05  7.52244E+03
        """

        # Pull necessary data from files
        npts = self._hist_ds['KMT'].data.size
        ds_coords = dict()
        ds_coords['ndims'] = ndims
        ds_coords['npts'] = npts
        ds_coords['num_nearest_pts'] = num_nearest_pts

        land_mask = self._hist_ds['KMT'].data.reshape(npts) != 0
        atlantic_mask = (self._region_mask == 6).reshape(npts)
        pacific_mask = (self._region_mask == 2).reshape(npts)
        lat = self._hist_ds['TLAT'].data.reshape(npts)
        lon = self._hist_ds['TLONG'].data.reshape(npts)
        # Convert (lat, lon) -> (x, y, z)
        deg2rad = np.pi/180.
        lon_loc = lon * deg2rad
        lat_loc = lat * deg2rad
        x3d = np.cos(lat_loc)*np.cos(lon_loc)
        y3d = np.cos(lat_loc)*np.sin(lon_loc)
        z3d = np.sin(lat_loc)

        self.grid = xr.Dataset(coords=ds_coords)
        self.grid['ndims'] = xr.DataArray(np.arange(ndims)+1, dims='ndims')
        self.grid['npts'] = xr.DataArray(np.arange(npts)+1, dims='npts')
        self.grid['num_nearest_pts'] = xr.DataArray(np.arange(num_nearest_pts)+1,
                                                    dims='num_nearest_pts')
        self.grid['coord_matrix'] = xr.DataArray(np.array([x3d, y3d, z3d]), dims=['ndims', 'npts'])
        self.grid['included_pts'] = xr.DataArray(land_mask, dims=['npts'])
        self.grid['norm'] = xr.DataArray(np.empty((npts, num_nearest_pts), dtype=np.float64),
                                         dims=['npts', 'num_nearest_pts'])
        self.grid['norm_jind'] = xr.DataArray(np.empty((npts, num_nearest_pts), dtype=np.int),
                                              dims=['npts', 'num_nearest_pts'])

        # from scipy.sparse import lil_matrix
        # norms_mat = lil_matrix((npts, npts))
        import time
        loop1_start = time.time()
        for i in np.where(self.grid['included_pts'])[0]:
            self.grid['norm_jind'].data[i, :], self.grid['norm'].data[i, :] = _weight_gen_loop(i, self.grid.copy(), num_nearest_pts,
                               atlantic_mask, pacific_mask)
        loop1_end = time.time()
        print("Loop1 time: {}".format(loop1_end-loop1_start))

        #     for j, jj in enumerate(norm_jind):
        #         norms_mat[i, jj] = norms_loc[j]
        # self.grid['norm_csr'] = norms_mat.tocsr()

    #####################

    def to_netcdf(self, out_file):
        """
            Dump self['LOESS_weights'] to netcdf
        """
        if hasattr(self, 'grid'):
            print("Dump to netcdf file: {}".format(out_file))
            self.grid.to_netcdf(out_file)

class LoessWeightGenParallelClass(LoessWeightGenClass): # pylint: disable=useless-object-inheritance,too-few-public-methods
    """
    A class to wrap function calls required to generate weights for LOESS regression
    (Parallelized via dask)
    """
    def __init__(self, hist_file, mask_file, npart=50):
        self._read_files(hist_file, mask_file)
        self._gen_weights(npart=npart)
        # self.to_netcdf()

    def _gen_weights(self, ndims=3, num_nearest_pts=384, npart=50): # pylint: disable=too-many-locals
        """
            Main subroutine to generate weights for linear regression step of LOESS

            Region masks come from POP log:

                                          +-----------------------+
                                          |  Marginal Seas Only   |
              Region        Region        |  Area        Volume   |
              Number        Name          | (km^2)       (km^3)   |
              ------ -------------------  -----------  -----------
                 1        Southern Ocean
                 2         Pacific Ocean
                 3          Indian Ocean
                -4          Persian Gulf  3.21703E+05  2.57362E+04
                -5               Red Sea  5.55613E+05  5.56528E+04
                 6        Atlantic Ocean
                 7     Mediterranean Sea
                 8          Labrador Sea
                 9               GIN Sea
                10          Arctic Ocean
                11            Hudson Bay
               -12            Baltic Sea  3.70458E+05  1.74489E+04
               -13             Black Sea  4.50883E+05  3.29115E+05
               -14           Caspian Sea  2.50748E+05  7.52244E+03
        """

        # Pull necessary data from files
        npts = self._hist_ds['KMT'].data.size
        ds_coords = dict()
        ds_coords['ndims'] = ndims
        ds_coords['npts'] = npts
        ds_coords['num_nearest_pts'] = num_nearest_pts

        land_mask = self._hist_ds['KMT'].data.reshape(npts) != 0
        atlantic_mask = (self._region_mask == 6).reshape(npts)
        pacific_mask = (self._region_mask == 2).reshape(npts)
        lat = self._hist_ds['TLAT'].data.reshape(npts)
        lon = self._hist_ds['TLONG'].data.reshape(npts)
        # Convert (lat, lon) -> (x, y, z)
        deg2rad = np.pi/180.
        lon_loc = lon * deg2rad
        lat_loc = lat * deg2rad
        x3d = np.cos(lat_loc)*np.cos(lon_loc)
        y3d = np.cos(lat_loc)*np.sin(lon_loc)
        z3d = np.sin(lat_loc)

        self.grid = xr.Dataset(coords=ds_coords)
        self.grid['ndims'] = xr.DataArray(np.arange(ndims)+1, dims='ndims')
        self.grid['npts'] = xr.DataArray(np.arange(npts)+1, dims='npts')
        self.grid['num_nearest_pts'] = xr.DataArray(np.arange(num_nearest_pts)+1,
                                                    dims='num_nearest_pts')
        self.grid['coord_matrix'] = xr.DataArray(np.array([x3d, y3d, z3d]), dims=['ndims', 'npts'])
        self.grid['included_pts'] = xr.DataArray(land_mask, dims=['npts'])
        self.grid['norm'] = xr.DataArray(np.empty((npts, num_nearest_pts), dtype=np.float64),
                                         dims=['npts', 'num_nearest_pts'])
        self.grid['norm_jind'] = xr.DataArray(np.empty((npts, num_nearest_pts), dtype=np.int),
                                              dims=['npts', 'num_nearest_pts'])

        import time
        loop1_start = time.time()
        # FIXME: parallelize this loop
        ocn_pts = dask.bag.from_sequence(np.where(self.grid['included_pts'])[0].tolist(), npartitions=npart)
        dask_out = ocn_pts.map(_weight_gen_loop, self.grid.copy(), num_nearest_pts,
                               atlantic_mask, pacific_mask).compute()
        loop1_end = time.time()
        loop2_start = time.time()
        for n, i in enumerate(np.where(self.grid['included_pts'])[0]):
            self.grid['norm_jind'].data[i, :] = dask_out[n][0]
            self.grid['norm'].data[i, :] = dask_out[n][1]
        loop2_end = time.time()
        print("Loop1 time: {}".format(loop1_end-loop1_start))
        print("Loop2 time: {}".format(loop2_end-loop2_start))

#######################

def _weight_gen_loop(i, grid_obj, num_nearest_pts, atlantic_mask, pacific_mask):
        # Find num_nearest_pts smallest values
        # https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi

        # Using home-spun norm
        temp_array = np.linalg.norm(grid_obj['coord_matrix'].data[:,i].reshape(3,1) - grid_obj['coord_matrix'].data, axis=0)
        norms_loc = np.partition(temp_array, num_nearest_pts-1)[:num_nearest_pts]
        norm_jind = np.argpartition(temp_array, num_nearest_pts-1)[:num_nearest_pts]
        if atlantic_mask[i]:
            norms_loc = np.where(pacific_mask[norm_jind], 0, norms_loc)
        elif pacific_mask[i]:
            norms_loc = np.where(atlantic_mask[norm_jind], 0, norms_loc)
        # grid_obj['norm_jind'].data[i, :] = norm_jind
        # grid_obj['norm'].data[i, :] = norms_loc
        return norm_jind, norms_loc

#####################
# Can run as script #
#####################

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a netCDF file containing linear " +
                                     "regression weights for LOESS")

    # Command line argument to point to land mask file
    parser.add_argument('-f', '--history-file', action='store', dest='hist_file', required=True,
                        help='Location of POP history file (netCDF format) containing grid info')

    parser.add_argument('-m', '--region-mask-file', action='store', dest='mask_file',
                        required=True, help='Location of binary file containing POP regions')

    parser.add_argument('-o', '--output-file', action='store', dest='out_file', required=False,
                        default='weights.nc', help='Name of netCDF file to write weights in')

    return parser.parse_args()

##############################
# __name__ == __main__ block #
##############################

if __name__ == "__main__":
    args = _parse_args() # pylint: disable=invalid-name
    wgts = LoessWeightGenClass(args.hist_file, args.mask_file) # pylint: disable=invalid-name
    wgts.to_netcdf(args.out_file)
