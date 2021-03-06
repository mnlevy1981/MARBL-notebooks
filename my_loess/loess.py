""" A private base class that contains functions needed for LOESS (locally estimated
    scatterplot smoothing), and several public classes providing different interfaces.

    1. LinearLoess1D(x, num_nearest_points): used for 1D data sets
    2. LinearLoess1D(x, y, num_nearest_points): used for 2D data sets
    3. LinearLoessGlobal(lon, lat, num_nearest_points): used for global data sets
    4. LinearLoessFromGridFile(grid_file): read previously-generated grid file
    5. LinearLoessFromPOPHistoryFile(history_file, num_nearest_points): use TLONG and TLAT
         for lon and lat and also set up mask (KMT == 0)

    For now, this class only performs linear regression (bilinear for 2D); quadratic will
    be added at a later time

    Steps to take:

    1) Construct class by defining the grid to smooth data on
       - intent(in):  all coordinates, number of neighbors to include in regression, and an
                      optional flag to convert (lat, lon) grid to unit sphere (default: false)
       - intent(out): self.loess_grid, an xarray dataset containing grid information as well
                      as matrix containing distance to nearest neighbors (and their indices)
"""

import os
import numpy as np
import xarray as xr

class _LinearLoessBaseClass(object):
    def __init__(self, x=None, num_nearest_pts=None, mask=None, grid_file=None):
        """ * x is a matrix of coordinates (nrow = number of dimensions,
              ncol = number of points)
              -- vector x is converted to 1 x n matrix
            * num_nearest_pts is number of local points to use in smoothing
            * set convert_lat_lon to true if x = (lat, lon) and you want to
              smooth on unit sphere
            Alternatively, initialize grid by reading grid_file
        """
        if grid_file:
            if not os.path.isfile(grid_file):
                raise FileNotFound("Can not find '{}'".format(grid_file))
            self.grid = xr.open_dataset(grid_file)
            self.ndims = self.grid.sizes['ndims']
            self.npts = self.grid.sizes['npts']
            self.num_nearest_pts = self.grid.sizes['num_nearest_pts']
            return

        if x is None or num_nearest_pts is None:
            raise ValueError("Unless reading a grid file, x and num_nearest_pts are both required")

        if len(x.shape) != 2:
            raise ValueError("x must be 2D (ndims x npts)")

        self.ndims = x.shape[0]
        self.npts = x.shape[1]
        self.num_nearest_pts = num_nearest_pts

        self.grid = xr.Dataset(coords={'ndims' : self.ndims, 'npts' : self.npts, 'num_nearest_pts' : self.num_nearest_pts})
        self.grid['ndims'] = xr.DataArray(np.arange(self.ndims)+1, dims='ndims')
        self.grid['npts'] = xr.DataArray(np.arange(self.npts)+1, dims='npts')
        self.grid['num_nearest_pts'] = xr.DataArray(np.arange(self.num_nearest_pts)+1, dims='num_nearest_pts')
        self.grid['coord_matrix'] = xr.DataArray(x, dims=['ndims', 'npts'])
        if mask is None:
            self.grid['included_pts'] = xr.DataArray([True]*self.npts, dims=['npts'])
        else:
            self.grid['included_pts'] = xr.DataArray(mask, dims=['npts'])
        self.compute_distances(num_nearest_pts)

    def compute_distances(self, num_nearest_pts):
        """ For each point x[:,i], find the num_nearest_pts points x[:,j] closest
            to x[:,i]
            * store ||x[:,i] - x[:,j]|| in self.grid['norm'] (npts x num_nearest_pts
              matrix of type r8)
            * store the j indices corresponding to the num_nearest_pts dimension of the
              norm in self.grid['norm_jind'] (npts x num_nearest_pts matrix of type int)
        """
        self.grid['norm'] = xr.DataArray(np.empty((self.npts, self.num_nearest_pts), dtype=np.float64), dims=['npts', 'num_nearest_pts'])
        self.grid['norm_jind'] = xr.DataArray(np.empty((self.npts, self.num_nearest_pts), dtype=np.int), dims=['npts', 'num_nearest_pts'])
        for i in np.where(self.grid['included_pts'])[0]:
#        for i in range(0, self.npts):
            # Find num_nearest_pts smallest values
            # https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi
            temp_array = np.linalg.norm(self.grid['coord_matrix'][:,i] - self.grid['coord_matrix'], axis=0)
            self.grid['norm'].data[i,:] = np.partition(temp_array, num_nearest_pts-1)[:num_nearest_pts]
            self.grid['norm_jind'].data[i,:] = np.argpartition(temp_array, num_nearest_pts-1)[:num_nearest_pts]

    def poly_fit_at_i(self, i, data, t=0):
        """ Given an index i:
            1. Find the n x-coordinates closest to self.x[i]
            2. Fit a polynomial of degree d to those points
            3. Return polynomial coefficients as well as all (self.x, self.data) values used to determine polynomial
            4. Apply robustness t times
        """
        j_ind = [j for j in self.grid['norm_jind'].data[i,:] if not np.isnan(data[j])]
        xj = self.grid['coord_matrix'].data[:,j_ind]
        dataj = np.array(data[j_ind])

        h = np.max(self.grid['norm'].data[i,:])
        poly_coeffs = self._compute_poly_coeffs(xj, np.squeeze(dataj), _W(self.grid['coord_matrix'].data[:,[i]] - xj, h))
        for _ in range(t):
                data_est = evaluate_poly(poly_coeffs, xj)
                residuals = np.squeeze(np.asarray(np.abs(data_est - dataj)))
                s = np.median(residuals)
                poly_coeffs = self._compute_poly_coeffs(xj, np.squeeze(dataj), _B(residuals/(6*s))*_W(self.grid['coord_matrix'].data[:,[i]] - xj, h))

        return poly_coeffs, xj, dataj

    def poly_fit(self, data, t=0):
        data_est = np.where(self.grid['included_pts'], 0, np.nan)
        for i in np.where(self.grid['included_pts'])[0]:
#        for i in range(0, self.npts):
            poly_coeffs, _, _ = self.poly_fit_at_i(i, data, t)
            data_est[i] = evaluate_poly(poly_coeffs, self.grid['coord_matrix'].data[:,i])
        return data_est

    def _compute_poly_coeffs(self, x_mat, z_arr, wgts=None):
        """ x_mat is m by n matrix, z is array of length n
             return n+1 coefficients c for first order multivariate polynomial f(x[:])
             such that
             z ~= c[0] + c[1]*x[0] + c[2]*x[1] + ... + c[m]*x[m-1]
        """

        assert (type(x_mat) == np.ndarray) or type(x_mat) == np.matrix, "x_mat must be a numpy array or matrix, not {}".format(type(x_mat))
        assert type(z_arr) == np.ndarray, "z_arr must be a numpy array, not {}".format(type(z_arr))
        assert len(x_mat.shape) == 2, "x_mat must be m by n"
        assert x_mat.shape[1] == z_arr.size, "second dimension of {} matrix x_mat must equal size of z_arr ({})".format(x_mat.shape[1], z_arr.size)

        if wgts is None:
            print('No weights provided')
            sqrt_wgts = np.ones(z_arr.size)
        else:
            sqrt_wgts = np.sqrt(wgts)

        a = np.empty((x_mat.shape[1], x_mat.shape[0]+1))
        a[:, 0]  = sqrt_wgts
        a[:, 1:] = np.multiply(x_mat, sqrt_wgts).transpose()

        return np.linalg.lstsq(a, z_arr*sqrt_wgts, rcond=-1)[0]

def evaluate_poly(coeffs, x_coord):
    return np.squeeze(np.array(coeffs[0] + coeffs[1:].dot(x_coord)))

def _W(x, h):
    """
    Tricube weight function from Cleveland (1979)
    We should only be calling this for points where |x| <= h
    """
    return np.where(np.linalg.norm(x, axis=0) < h, (1 - np.linalg.norm(x/h, axis=0)**3)**3, 0)

def _B(x):
    return np.where(x < 1, (1 - x**2)**2, 0)

####################
# CLASS EXTENSIONS #
####################

class LinearLoess1D(_LinearLoessBaseClass):
    def __init__(self, x, num_nearest_pts, mask=None):
        super(LinearLoess1D, self).__init__(np.array([x]), num_nearest_pts, mask)

class LinearLoess2D(_LinearLoessBaseClass):
    def __init__(self, x, y, num_nearest_pts, mask=None):
        super(LinearLoess2D, self).__init__(np.array([x, y]), num_nearest_pts, mask)

class LinearLoessGlobal(_LinearLoessBaseClass):
    def __init__(self, lon, lat, num_nearest_pts, convert_lat_lon=False, is_degrees=True, mask=None):
        """ Given longitude and latitude, can """
        if convert_lat_lon:
            if is_degrees:
                deg2rad = np.pi/180.
                lon_loc = lon * deg2rad
                lat_loc = lat * deg2rad
            x3d = np.cos(lat_loc)*np.cos(lon_loc)
            y3d = np.cos(lat_loc)*np.sin(lon_loc)
            z3d = np.sin(lat_loc)
            xmat = np.array([x3d, y3d, z3d])
        else:
            xmat = np.array([lon, lat])
        super(LinearLoessGlobal, self).__init__(xmat, num_nearest_pts, mask)

class LinearLoessFromGridFile(_LinearLoessBaseClass):
    def __init__(self, grid_file):
        super(LinearLoessFromGridFile, self).__init__(grid_file=grid_file)

class LinearLoessFromPOPHistoryFile(LinearLoessGlobal):
    def __init__(self, history_file, num_nearest_points, convert_lat_lon=False):
        self.pop_ds = xr.open_dataset(history_file, decode_times=False).isel(time=0)
        npts = self.pop_ds['TLONG'].data.size
        lon = self.pop_ds['TLONG'].data.reshape(npts)
        lat = self.pop_ds['TLAT'].data.reshape(npts)
        mask = np.where(self.pop_ds['KMT'].data == 0, False, True).reshape(npts)
        super(LinearLoessFromPOPHistoryFile, self).__init__(lon, lat,
                                                            num_nearest_points,
                                                            convert_lat_lon=convert_lat_lon,
                                                            mask=mask)
