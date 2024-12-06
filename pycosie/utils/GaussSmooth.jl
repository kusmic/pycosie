# TO DO:
#   - Need ot make unit tests to make sure this works
#       * Stuff may not be coded properly for Julia....
#       * Have test particle in test box and test grid smoothing over
#         a test value to make sure it's working

using PyCall
using SpecialFunctions

# TODO: convert any NumPyArrays usage to convert

np = pyimport("numpy")

function recenter(coord, domainWidth, Dx::Float64, Dy::Float64, Dz::Float64)
    """
    I forgot what this is for. It's legacy code that I'm scared to delete.
    """
    # First converting to Julia types
    coordJl = convert(Array, coord)
    domWidthJl = convert(Array, domainWidth)

    recenterArr = [0,0,0]
    # REMEMBER julia is 1-indexing
    X = coord.shape[1]
    Y = coord.shape[2]
    tempCoord::Array = convert(Array, np.zeros((X,Y)))
    # recentering 
    if Dx > domWidthJl[1]/2
        recenterArr[1] = domWidthJl[1]
    end
    if Dy > domWidthJl[2]/2
        recenterArr[2] = domWidthJl[2]
    end
    if Dz > domWidthJl[3]
        recenterArr[3] = domWidthJl[3]
    end

    for gi = 1:length(coordJl)
        for qi in 1:length(recenterArr)
            check = coord[gi,qi] < recenterArr[qi]
            if check
                tempCoord[gi,qi] = coordJl[gi,qi] + recenterArr[qi]
            else
                tempCoord[gi,qi] = coordJl[gi,qi]
            end
        end
    end

    return tempCoord
end

function gaussErf(t::Float64, a::Float64, b::Float64, sigma::Float64)
    """
    Calculates the definite integral of Gaussian dist. from a to b assuming
    mean t and std. dev. sigma. Used to get area in grid cells in this case (itc)
    Args:
        t (float): mean of Gaussian, or itc coordinate of particle
        a (float): lower limit of integral itc lower edge of cell 
        b (float): upper limit of integral itc upper edge of cell 
        sigma (float): std. dev. itc smoothing length converted to 
            Gaussian kernel
    Returns:
        soln (float): solution of the integral

    """
    C::Float64 = 0.707107
    xa = (t-a)/(sqrt(2)*sigma) 
    xb = (t-b)/(sqrt(2)*sigma) 
    soln::Float64 = 0.0

    crit1::Float64 = 5.0 # 5 sigma
    if (abs(xa) < crit1) & (abs(xb) < crit1) # within 5 sigma
        A = SpecialFunctions.erf(C*xa)
        B = SpecialFunctions.erf(C*xb)
        soln = 0.5 * (A-B)
    else # somehow not caught in 5 sigma because
        if t>a & t<b  # inside cell, still do it
            A = SpecialFunctions.erf(C*xa)
            B = SpecialFunctions.erf(C*xb)
            soln = 0.5 * (A-B)
        else # outside 5 sigma, do not calc
            soln = 0.0
        end
    end

    return soln 
end

function gaussLoop(gL::Int64, testCoord, testSL::Float64, L::Float64)
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

    gaussKernel::Array = convert(Array, np.zeros((gL,gL,gL)) )
    Edges::Array = convert(Array, np.linspace(0,L,gL+1) )
    SPHtoGauss::Float64 = 0.33479
    sigma::Float64 = SPHtoGauss * testSL
    dLimLo::Array = convert(Array, testCoord )
    dLimLo = dLimLo - (5*sigma)*[1,1,1]
    dLimHi::Array = convert(Array, testCoord)
    dLimHi = dLimHi + (5*sigma)*[1,1,1]
    # Gotta find indices of edges s.t. edge < lower limit then edge > higher limit
    #argWhere = pycall(np.argwhere, Array, Edges < dLimLo[1])
    #ilo::Array = convert(Array, argWhere)
    # i index = x
    imax::Int64 = gL
    imin::Int64 = 1
    for __i = 1:length(Edges)
        if Edges[__i] < dLimLo[1]
            if __i > imin 
                imin = __i
                continue
            end
        end
        if Edges[__i] > dLimHi[1]
            if __i < imax 
                imax = __i
                continue
            end
        end
    end
    # j index = Y
    jmax::Int64 = gL
    jmin::Int64 = 1
    for __j = 1:length(Edges)
        if Edges[__j] < dLimLo[2]
            if __j > jmin 
                jmin = __j
                continue
            end
        end
        if Edges[__j] > dLimHi[2]
            if __j < jmax 
                jmax = __j
                continue
            end
        end
    end
    # k index = z 
    kmax::Int64 = gL
    kmin::Int64 = 1
    for __k = 1:length(Edges)
        if Edges[__k] < dLimLo[3]
            if __k > kmin 
                kmin = __k
                continue
            end
        end
        if Edges[__k] > dLimHi[3]
            if __k < kmax 
                kmax = __k
                continue
            end
        end
    end
    # calculating gaussian kernel values in ,loop
    Kx::Float64 = 0.0
    Ky::Float64 = 0.0
    Kz::Float64 = 0.0
    normTot::Float64 = 0.0
    for i = imin:imax 
        for j = jmin:jmax 
            for k = kmin:kmax 
                Kx = gaussErf(testCoord[1], Edges[i], Edges[i+1], sigma)
                Ky = gaussErf(testCoord[2], Edges[j], Edges[j+1], sigma)
                Kz = gaussErf(testCoord[3], Edges[k], Edges[k+1], sigma)
                gaussKernel[i,j,k] = gaussKernel[i,j,k] + ( Kx * Ky * Kz )
                normTot += Kx * Ky * Kz
            end
        end
    end

    gaussKernel = gaussKernel #/ normTot
    return gaussKernel

end
