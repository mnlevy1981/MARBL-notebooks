""" A class to contain functions needed for LOESS (locally estimated scatterplot smoothing)
"""

import numpy as np

class loess(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert x.size == y.size, "x and y must be same size"
        self.npts = x.size

    def poly_fit_at_i(self, i, n, d, t=0):
        """ Given an index i:
            1. Find the n x-coordinates closest to self.x[i]
            2. Fit a polynomial of degree d to those points
            3. Return polynomial coefficients as well as all (self.x,self.y) values used to determine polynomial
            4. Apply robustness t times
        """
        # Find n smallest values
        # https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi
        hi = np.partition(np.abs(self.x[i] - self.x), n-1)[:n]
        h = np.max(hi)
        j_ind = np.argpartition(np.abs(self.x[i] - self.x), n-1)[:n]
        xj = self.x[j_ind]
        yj = self.y[j_ind]

        # Regression: yj ~= sum b_n * (xj**n)
        betas = np.polyfit(xj, yj, d, w=np.sqrt(_W(self.x[i] - xj, h)))[::-1]
        for _ in range(t):
            yhat = np.zeros(n)
            for p, beta in enumerate(betas):
                yhat += beta*(xj**p)
            residuals = np.abs(yhat - yj)
            s = np.median(residuals)
            betas = np.polyfit(xj, yj, d, w=np.sqrt(_B(residuals/(6*s))*_W(self.x[i] - xj, h)))[::-1]

        return betas, xj, yj

    def poly_fit(self, n, d, t=0):
        yhat = np.zeros(self.npts)
        for i in range(0, self.npts):
            betas, _, _ = self.poly_fit_at_i(i, n, d, t)
            for p, beta in enumerate(betas):
                yhat[i] += beta * (self.x[i]**p)
        return yhat

def _W(x, h):
    """
    Tricube weight function from Cleveland (1979)
    We should only be calling this for points where |x| <= h
    """
    return np.where(np.abs(x) < h, (1 - np.abs(x/h)**3)**3, 0)

def _B(x):
    return np.where(np.abs(x)<1, (1 - x**2)**2, 0)
