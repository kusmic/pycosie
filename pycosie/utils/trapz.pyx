# distutils: language=c++

import numpy as np
cimport numpy as np

def _binwise_trapz_sorted(np.ndarray x, np.ndarray y, np.ndarray bin_edges):
    # 
    # Trapezoidal integration over bins.

    # Integrate each row of `y(x)` over each bin defined by `bin_edges` using
    # trapezoidal integration. The values of `bin_edges` do not have to coincide
    # with values given in `x`, the rows of `y` are linearly interpolated
    # correspondingly.

    # Parameters
    # ----------
    # x : `numpy.ndarray` (N_x,)
    #     `x`-values corresponding to each column of `y`. Assumed to be sorted in
    #     ascending or descending order. Integrated values will be negative for
    #     descending order.
    # y : `numpy.ndarray` (N, N_x)
    #     N functions of `x` evaluated at each of its values.
    # bin_edges : `numpy.ndarray` (N_bins+1,)
    #     Edges of the bins over which to perform integration. Assumed to be
    #     sorted in same order as `x` and to span a range <= the range spanned by
    #     `x`.

    # Returns
    # -------
    # res : `numpy.ndarray` (N, N_bins)
    #     Integral over each bin of each row of `y`.
    # 
    cdef res = np.empty((y.shape[1], len(bin_edges)-1))

    cdef i1 = 0
    cdef i2 = 0
    cdef y1 = np.empty((y.shape[1]))
    cdef y2 = np.empty((y.shape[1]))
    cdef x1 = 0.0
    cdef x2 = 0.0
    for j in range(res.shape[1]):
        x1 = bin_edges[j]
        x2 = bin_edges[j+1]

        # ascending
        if x[0] < x[1]:
            # Find last element <x1 and last element <x2 in x.
            while x1 > x[i1+1]:
                i1 += 1
            i2 = i1
            while x2 > x[i2+1]:
                i2 += 1
        # descending
        elif x[0] > x[1]:
            # Find last element >x1 and last element >x2 in x.
            while x1 < x[i1+1]:
                i1 += 1
            i2 = i1
            while x2 < x[i2+1]:
                i2 += 1
        else:
            raise ValueError("Identical values in `x`!")

        # Find y1=y(x1) and y2=y(x2) by interpolation.
        y1 = (
            (x[i1+1]-x1)*y[:, i1] + (x1-x[i1])*y[:, i1+1]
        ) / (x[i1+1]-x[i1])
        y2 = (
            (x[i2+1]-x2)*y[:, i2] + (x2-x[i2])*y[:, i2+1]
        ) / (x[i2+1]-x[i2])

        if i1 == i2:
            # Have only area from x1 to x2.
            res[:, j] = (x2-x1)*(y1+y2)/2
        else:
            # Area from x1 to x(i1+1).
            res[:, j] = (x[i1+1]-x1)*(y1+y[:, i1+1])/2
            # Add area from x(i1+1) to x(i2-1).
            for i in range(i1+1, i2):
                res[:, j] += (x[i+1]-x[i])*(y[:, i]+y[:, i+1])/2
            # Add area from x(i2) to x2.
            res[:, j] += (x2-x[i2])*(y2+y[:, i2])/2

    return res