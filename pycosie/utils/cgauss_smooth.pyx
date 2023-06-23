# distutils: language=c++

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, erf


def recenter(np.ndarray coord, np.ndarray domainWidth, double Dx, double Dy, double Dz):
    cdef np.ndarray tempCoord = np.zeros( (coord.shape[0], coord.shape[1]), dtype=float )
    cdef list recenterArr = [0,0,0]
    if Dx > domainWidth[0]/2:
        recenterArr[0] = domainWidth[0]
    if Dy > domainWidth[1]/2:
        recenterArr[1] = domainWidth[1]
    if Dz > domainWidth[2]/2:
        recenterArr[2] = domainWidth[2]
    for gi in range(len(coord)):
        for qi in range(len(recenterArr)):
            check = coord[gi,qi] < recenterArr[qi]/2
            if check:
                tempCoord[gi,qi] = coord[gi,qi] + recenterArr[qi]
            else:
                tempCoord[gi,qi] = coord[gi,qi]
    return( np.copy(tempCoord) )

def gaussErf(double t, double a, double b, double sigma):
    cdef float C = 0.707107
    cdef double xa = (t-a)/(sqrt(2)*sigma)
    cdef double xb = (t-b)/(sqrt(2)*sigma)
    cdef double soln # will hold the definite integral quantity
    if ( np.abs(xa) < 5*sqrt(2) ) and ( np.abs(xb) < 5*sqrt(2) ): # grid edges not 5-sigma away from particle + smoothing length
        soln = 0.5 * ( erf(C*xa/(sqrt(2)*sigma)) - erf(C*xb/(sqrt(2)*sigma)) )
    else: # is 5-sigma away
        if t > a and t < b: # particle in that grid
            soln = 0.5 * ( erf(C*xa/(np.sqrt(2)*sigma)) - erf(C*xb/(np.sqrt(2)*sigma)) )
        else: # not in that grid, do not compute
            soln = 0.0
    return(soln)
#
def gaussLoop(int gL, np.ndarray testCoord, double testSL, double L):
    """# 
    Creates and returns a Gaussian kernel for a single particle, even accounts off-center from each cell.
    Args:
        gL (int): Side length of kernel in number of cells. Assumes cubic kernel
        testCoord (yt.Cosmology.arr): Coordinate of the single particle.
        testSL (yt.Cosmology.arr): Smoothing length of the single particle.
        L (float): [in ckpc/h] Side length of the zoomed in region. Assumes cube.
    Returns:
        gaussKernel (numpy.array): The Gaussian kernel
    """#
    cdef np.ndarray gaussKernel = np.zeros( (gL, gL, gL) )
    cdef Edges = np.linspace(0, L, gL+1)
    cdef double SPHtoGauss = 0.33479
    cdef double sigma = SPHtoGauss * testSL
    cdef np.ndarray dLimLo = testCoord - (5*sigma)
    cdef np.ndarray dLimHi = testCoord + (5*sigma)
    # Gotta find indices of edges s.t. edge < lower limit then edge > higher limit
    cdef np.ndarray ilo = np.argwhere(Edges < dLimLo[0])
    # gotta do this by my own code, find minimum and maximum indices i,j,k
    cdef int imax = gL
    cdef int imin = 0
    for __i in range(len(Edges)):
        if Edges[__i] < dLimLo[0]:
            if __i > imin:
                imin = __i
                continue
        if Edges[__i] > dLimHi[0]:
            if __i < imax:
                imax = __i
                break
    cdef int jmax = gL
    cdef int jmin = 0
    for __j in range(len(Edges)):
        if Edges[__j] < dLimLo[1]:
            if __j > jmin:
                jmin = __j
                continue
        if Edges[__j] > dLimHi[1]:
            if __j < jmax:
                jmax = __j
                break
    cdef int kmax = gL
    cdef int kmin = 0
    for __k in range(len(Edges)):
        if Edges[__k] < dLimLo[2]:
            if __k > kmin:
                kmin = __k
                continue
        if Edges[__k] > dLimHi[2]:
            if __k < kmax:
                kmax = __k
                break
    # Now to loop and find the gaussian kernel
    cdef double Kx
    cdef double Ky
    cdef double Kz
    for i in range(imin, imax):
        for j in range(jmin,jmax):
            for k in range(kmin,kmax):
                Kx = gaussErf(testCoord[0], Edges[i], Edges[i+1], sigma)
                Ky = gaussErf(testCoord[1], Edges[j], Edges[j+1], sigma)
                Kz = gaussErf(testCoord[2], Edges[k], Edges[k+1], sigma)
                gaussKernel[i,j,k] = Kx * Ky * Kz
    print(np.sum(gaussKernel))
    gaussKernel = gaussKernel / np.sum(gaussKernel)
    return(gaussKernel)