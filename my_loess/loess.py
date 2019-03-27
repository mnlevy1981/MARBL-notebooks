""" A class to contain functions needed for LOESS (locally estimated scatterplot smoothing)
"""

import numpy as np

class loess(object):
    def __init__(self, x, data, convert_lat_lon=False):
        """ * x is a matrix of coordinates (nrow = number of dimensions, ncol = number of points)
              -- vector x is converted to 1 x n matrix
            * data is data sampled at x (to be smoothed)
        """
        if convert_lat_lon:
            lon = np.squeeze(np.array(x[0,:]))
            lat = np.squeeze(np.array(x[1,:]))
            x3d = np.cos(lat)*np.cos(lon)
            y3d = np.cos(lat)*np.sin(lon)
            z3d = np.sin(lat)
            self.x = np.matrix([x3d, y3d, z3d])
        else: 
            self.x = np.asmatrix(x)
        self.data = np.asmatrix(data)
        assert x.shape[-1] == data.shape[-1], "x and fx must be same length"
        self.npts = data.shape[-1]

    def poly_fit_at_i(self, i, n, d, t=0):
        """ Given an index i:
            1. Find the n x-coordinates closest to self.x[i]
            2. Fit a polynomial of degree d to those points
            3. Return polynomial coefficients as well as all (self.x, self.data) values used to determine polynomial
            4. Apply robustness t times
        """
        # Find n smallest values
        # https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi
        temp_array = np.linalg.norm(self.x[:,i] - self.x, axis=0)
        hi = np.partition(temp_array, n-1)[:n]
        h = np.max(hi)
        j_ind = np.argpartition(temp_array, n-1)[:n]
        xj = self.x[:,j_ind]
        dataj = np.array(self.data[0,j_ind])

        if False:
            # Regression: yj ~= sum b_n * (xj**n)
            assert self.x.shape[0] == 1, "Currently expecting 1d coordinates"
            poly_data = np.poly1d(np.polyfit(np.squeeze(np.asarray(xj)), np.squeeze(np.asarray(dataj)), d, w=np.sqrt(_W(self.x[:,i] - xj, h))))
            for _ in range(t):
                data_est = poly_data(xj)
                residuals = np.squeeze(np.asarray(np.abs(data_est - dataj)))
                s = np.median(residuals)
                poly_data = np.poly1d(np.polyfit(np.squeeze(np.asarray(xj)), np.squeeze(np.asarray(dataj)), d, w=np.sqrt(_B(residuals/(6*s))*_W(self.x[:,i] - xj, h))))
        poly_coeffs = self._compute_poly_coeffs(np.squeeze(xj), np.squeeze(dataj), _W(self.x[:,i] - xj, h))
        for _ in range(t):
                data_est = evaluate_poly(poly_coeffs, xj)
                residuals = np.squeeze(np.asarray(np.abs(data_est - dataj)))
                s = np.median(residuals)
                poly_coeffs = self._compute_poly_coeffs(np.squeeze(xj), np.squeeze(dataj), _B(residuals/(6*s))*_W(self.x[:,i] - xj, h))

        return poly_coeffs, xj, dataj

    def poly_fit(self, n, d, t=0):
        data_est = np.zeros(self.npts)
        for i in range(0, self.npts):
            poly_coeffs, _, _ = self.poly_fit_at_i(i, n, d, t)
            data_est[i] = evaluate_poly(poly_coeffs, self.x[:,i])
        return data_est

    def _compute_poly_coeffs(self, x_mat, z_arr, wgts=None):
        """ x_mat is m by n matrix, z is array of length n
             return n+1 coefficients c for first order multivariate polynomial f(x[:])
             such that
             z ~= c[0] + c[1]*x[0] + c[2]*x[1] + ... + c[m]*x[m-1]
        """

        assert type(x_mat) == np.matrix, "x_mat must be a numpy matrix, not {}".format(type(x_mat))
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
