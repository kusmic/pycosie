# Import only this code
# Do not run

import numpy as np

def power_spectrum(fname, vcol, species=None,  dvChunk=8000, tauEff = -1, skipNegPix = True):
    # Takes the QuasarCosie Table and calculates the power spectrum of the LAF
    # Returns the power spectrum and wavenumber if wavenumber is positive number

    try:
        if isinstance(species, str): # checks to see if it is a string
            scol = _species_table(species) # uses hard-coded look-up-table function
        elif isintance(species, int):
            scol = species # assumes it is an integer. Please input string or integer.
        else:
            raise ValueError("Please input a string or integer for 'species' parameter.")
    except ValueError:
        raise

    vMC,F=np.loadtxt(fname,usecols=(vcol,scol),unpack=True)

    # assume that the pixels all have the same velocity width
    dv_mean = np.absolute(vMC[1] - vMC[0])
    nPixPerChunk = np.int(np.round(dvChunk / dv_mean)) # number of pixels in each chunk

    nOrig = len(F) # amount of data in our one big "bin"
    nChunks = np.int( np.round( nOrig / nPixPerChunk ) ) # number of chunks ("bin" = "chunk")

    # reshape flux and velocity arrays into chunks 8000 km/s long
    #flux = F.reshape(nChunks,nPixPerChunk) # reshaped so each array bin has arrays of data
    #vbins = vMC.reshape(nChunks,nPixPerChunk) # velocities binned like fluxes
    vMC_bw = np.absolute(np.max(vMC) - np.min(vMC))

    k = 2.0 * np.pi * np.fft.fftfreq(nPixPerChunk, d=dv_mean) # wavenumber

    pk = np.zeros((nChunks,nPixPerChunk), dtype=np.float_)  # empty array to hold power spectrum. nbins = number of spectra

    # use the global average mean flux to normalize the flux (just above equation 11 in Lukic+2015)
    if(tauEff > 0):
        _T_rescale(F, nOrig, tauEff)
    mean_flux = _T_mean(F, skipNegPix, nOrig, 1.0)

    # These two used to divide transmission array, so any dvChunk can be used
    ni = 0
    nf = nPixPerChunk
    for i in np.arange(nChunks): # for each bin
        flux = np.array(F[ni:nf])
        delta = (flux/mean_flux) - 1 # normalized flux contrast
        xk =  np.fft.fft(delta, n=nPixPerChunk) # The Fourier Transform in Lukic+ 2015 eq. 12.
	# use norm=None for updated python installations
        pk[i] = np.real(xk * np.conj(xk)) / (nPixPerChunk**2) # the power spectrum by definition
        
        #Now using ni, nf to find the correct index
        ni += nPixPerChunk
        nf += nPixPerChunk
        if nf > F.size:
            nf = F.size

    
    mean_pk = np.mean(pk[:], axis=0) # mean power in each k
    Delta_f = mean_pk * k * dv_mean * dvChunk / np.pi # dimensionless power spectrum from Lukic+ 2015 eq. 12

    return(Delta_f[k > 0], k[k > 0])

def write_power_spectrum(k, Delta_f, fname):
    # Writes the k and power spectrum to a file. Input is k array, PS array,
    # and a string for the file name
    data=np.zeros(2*len(k))
    data2d=data.reshape(len(k),2)
    for i in np.arange(len(k)):
        data2d[i,0] = k[i]
        data2d[i,1] = Delta_f[i]
    ofile=open(fname,"w")
    np.savetxt(ofile,data2d,fmt='%.6e %6e',newline='\n')
    ofile.close()

def _T_mean(T, skipNegPix, N, A):
    iCounted = 0
    
    Tmean = 0
    for i in range(N): # for each index in chunk
        if( T[i] > 0 ): # positive transmission
            tauThis = -np.log(T[i])
            Tmean += np.exp(-tauThis * A)
            iCounted += 1
        elif( (T[i] < 0) and (not skipNegPix) ): # negative transmission and we are not skipping it
            tauThis = -np.log(-T[i])
            Tmean += -np.exp(-tauThis * A)
            iCounted += 1
    # end of for
    Tmean /= iCounted
    return(Tmean)

def _get_y(TTarget, T, N, A):
    y = TTarget - _T_mean(T, N, A)

def _T_rescale(T, N, tauTarget):
    print("Rescaling transmission flux...")
    # Newton-Raphson used to determine scaling factor A s.t. mean flux matches observed flux
    TTarget = np.exp(-tauTarget)
    NIt = 0
    A1 = 1.0
    A2 = 0.99

    y1 = _get_y(TTarget, T, N, A1)
    y2 = _get_y(TTarget, T, N, A2)
    # Newton-Raphson: 0 = y2 + dydA * dA
    while(np.abs(y2) > 1e-3):
        try:
            dydA = (y2 - y1) / (A2 - A1)
            dA = -y2 / dydA
            A1 = A2
            A2 += dA
            y1 = y2
            y2 = _get_y(TTarget, T, N, A2)
        
            NIt += 1
            if(NIt > 20):
                raise IterationError("Over-iterated in re-scaling transmissions, NIt > 20")
        except IterationError:
            raise
        print( "Iterations:{0} A:{1}".format(NIt, A2) )
    # end of while
        
    # Now rescaling T
    # If T is array, python passes by reference, so no need for returns
    for i in range(N):
        if( T[i] > 0 ):
            tauThis = -np.log(T[i])
            T[i] = np.exp(-tauThis * A2)
        elif( T[i] < 0 ):
            tauThis = -np.log(-T[i])
            T[i] = -exp(-tauThis * A2)
    #end of for
    #end of function

class IterationError(Exception):
    pass
        
def _species_table(name):
    # This is a referencing function hard-coded to pick out the specific
    # column number of a data file produced with the QuasarCosie code.
    # It will look to see if the name string contains specific characters in
    # a specific order
    try:
        if name in "HI 1216":
            return(21)
        elif name in "CII 1334.53":
            return(22)
        elif name in "CIV 1548.20":
            return(23)
        elif name in "OI 1302.17":
            return(24)
        elif name in "SiII 1260.42":
            return(25)
        elif name in "SiIV 1393.76":
            return(26)
        elif name in "MgII 2796.35":
            return(27)
        else:
            raise ValueError("Line not found. Please check spelling. If issue persists, please input desired column number for 'species' instead.")
    except ValueError:
        raise
