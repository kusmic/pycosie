import numpy as np
from hoki import load
#from hoki.spec import bin_luminosity
from scipy.interpolate import interp1d
import astropy.units as u 


# Need to add bin_luminosity so spectrum binned to Cloudy wavelengths
#  
class BPASSSpectrum():
    """Create synthetic stellar spectrum using BPASS
    - Currently only uses binary population files
    - currently only imf135_100 model
    """
    
    def __init__(self, Mstar, Zstar, tstar, bpass_path, bpass_version, wl_arr=None):
        """
        initialize it, create spectrum using BPASS and parameters.

        Args:
            Mstar (float, unit:Msun): star particle mass, in Msun
            Zstar (float): star particle metallicity
            tstar (float): age of star particle in yr
            bpass_path (str): path to BPASS directory, holding all .dat tables
            bpass_version (str): version of BPASS
            wave_arr (array[float], str): array of wavelengths for spectrum, in Angstroms.
                If None, then takes a default wavelengths in Cloudy. If "bpass" then uses
                BPASS spectra table wavelengths. Default is None.
        """
    
        self.mstar = Mstar
        self.zstar = Zstar
        self.tstar = tstar
        self.path = bpass_path
        self.version = bpass_version
        
        if type(wl_arr) == type(None):
            self.WL = np.load("__cloudywave.npy")[::-1]
            # BPASS increases wl in arr, Cloudy decreases
        else:
            self.WL = wl_arr
        
        if self.version == "2.2.1":
            spectraFile = "spectra-bin-imf135_100"
            starmassFile = "starmass-bin-imf135_100"
            # 2.2.1 manual
            # "The units of flux are Solar Luminosities per Angstrom, normalised for a cluster of 1e6
            # Msun formed in a single instantaneous burst."
            _unitFlux = u.L
        elif self.version == "2.1":
            spectraFile = "spectra-bin"
            starmassFile = "starmass-bin"
        else:
            raise RuntimeError("Sorry! That is not a supported BPASS version yet!")
        
        # Loading respective BPASS tables with hoki
        # doing upper and lower values, interpolating in between!
        bpass_Zval_arr = np.array([.04, .03, .02, .014, .01, .008, .006, .004, .003, .002, .001, 1e-4, 1e-5])
        bpass_Zstr_arr = ["040", "030", "020", "014", "010", "008", "006", "004", "003", "002", "001", "em4", "em5"]
    
        # Is star metallicity in BPASS range or out?
    
        outLo = self.zstar<1e-5
        outHi = self.zstar>.04
        #starmass and mass loss func
        if not outLo and not outHi:
            idxHigherZ = np.argwhere(bpass_Zval_arr>self.zstar).flatten()[0]
            idxLowerZ = np.argwhere(bpass_Zval_arr<self.zstar).flatten()[-1]
            starmassL = load.model_output(f"{self.path}/{starmassFile}.z{bpass_Zstr_arr[idxLowerZ]}.dat") #these 2 are pandas DataFrames
            starmassH = load.model_output(f"{self.path}/{starmassFile}.z{bpass_Zstr_arr[idxHigherZ]}.dat")

            x = np.array((bpass_Zval_arr[idxLowerZ],bpass_Zval_arr[idxHigherZ]))
            y = np.array((   [starmassL["log_age"].values, starmassL["stellar_mass"].values], 
                            [starmassH["log_age"].values,starmassH["stellar_mass"].values] ))
            starmass_interp = interp1d(x, y, axis=0)

            # Determining how much mass was lost and linearly fixing luminosity
            starmass = starmass_interp(self.zstar)
            mass_loss_func = interp1d(starmass[0,:], starmass[1,:])
        elif outLo:
            starmass = load.model_output(f"{self.path}/{starmassFile}.zem5.dat")
            mass_loss_func = interp1d(starmass["log_age"].values, starmass["stellar_mass"].values)
        elif outHi:
            starmass = load.model_output(f"{self.path}/{starmassFile}z040.dat")
            mass_loss_func = interp1d(starmass["log_age"].values, starmass["stellar_mass"].values)
            
        surviveMass = mass_loss_func(np.log10(self.tstar))
        # spectra
        # more complex because have to interpolate on metallicity and age
        spectra_logageVal_arr = np.arange(6.0, 11.1, 0.1) # these are default age range for BPASS v2.2.1
        spectra_logageStr_arr = np.array([f"{x:.1f}" for x in spectra_logageVal_arr], dtype=str) # converting to string for pandas headers
        logAge = np.log10(self.tstar)
        if not outLo and not outHi:
            #idxHigherZ = np.argwhere(bpass_Zval_arr>self.zstar).flatten()[0]
            #idxLowerZ = np.argwhere(bpass_Zval_arr<self.zstar).flatten()[-1]
            spectraL = load.model_output(f"{self.path}/{spectraFile}.z{bpass_Zstr_arr[idxLowerZ]}.dat")
            spectraH = load.model_output(f"{self.path}/{spectraFile}.z{bpass_Zstr_arr[idxHigherZ]}.dat")
            
            idxHigherT = np.argwhere(spectra_logageVal_arr>=logAge).flatten()[0]
            idxLowerT = np.argwhere(spectra_logageVal_arr<logAge).flatten()[-1]
            
            wlBPASS = spectraL.WL
            age = np.array((spectra_logageVal_arr[idxLowerT],spectra_logageVal_arr[idxHigherT]))
            
            LT = spectra_logageStr_arr[idxLowerT]
            yLT = np.array([spectraL[LT].to_numpy(), spectraH[LT].to_numpy()])
            specLT_interp = interp1d(x, yLT, axis=0)
            specLT = specLT_interp(self.zstar)
            
            LT = spectra_logageStr_arr[idxHigherT]
            yLT = np.array([spectraL[LT].to_numpy(), spectraH[LT].to_numpy()])
            specLT_interp = interp1d(x, yLT, axis=0)
            specHT = specLT_interp(self.zstar)
            
            y = np.array([specLT, specHT])
            spec_interp = interp1d(age,y, axis=0)
            
            self._spectrum = spec_interp(np.log10(self.tstar)) * u.Lsun / u.AA
            
        elif outLo:
            spectraL = load.model_output(f"{self.path}/{spectraFile}.zem5.dat")
            
            idxHigherT = np.argwhere(spectra_logageVal_arr>=self.tstar).flatten()[0]
            idxLowerT = np.argwhere(spectra_logageVal_arr<self.tstar).flatten()[-1]
            
            wlBPASS = spectraL.WL
            age = np.array((spectra_logageVal_arr[idxLowerT],spectra_logageVal_arr[idxHigherT]))
            
            LT = spectra_logageStr_arr[idxLowerT]
            HT = spectra_logageStr_arr[idxHigherT]
            
            y = np.array([spectraL[LT].to_numpy(), spectraL[HT].to_numpy()])
            spec_interp = interp1d(age,y, axis=0)
            
            self._spectrum = spec_interp(np.log10(self.tstar)) * u.Lsun / u.AA
            
        elif outHi:
            spectraL = load.model_output(f"{self.path}/{spectraFile}.z040.dat")
            
            idxHigherT = np.argwhere(spectra_logageVal_arr>=self.tstar).flatten()[0]
            idxLowerT = np.argwhere(spectra_logageVal_arr<self.tstar).flatten()[-1]
            
            wlBPASS = spectraL.WL
            age = np.array((spectra_logageVal_arr[idxLowerT],spectra_logageVal_arr[idxHigherT]))
            
            LT = spectra_logageStr_arr[idxLowerT]
            HT = spectra_logageStr_arr[idxHigherT]
            
            y = np.array([spectraL[LT].to_numpy(), spectraL[HT].to_numpy()])
            spec_interp = interp1d(age,y, axis=0)
            
            self._spectrum = spec_interp(np.log10(self.tstar)) * u.Lsun / u.AA
            
        if wl_arr == "bpass":
            self.WL = wlBPASS
        elif wl_arr != "bpass":
            dWLCloudy = np.gradient(self.WL)/2 # SHOULD BE CLOUDY OR CUSTOM WAVELENGTH ARR'
            _wlLower = self.WL - dWLCloudy
            wlEdges = np.append(_wlLower, self.WL[-1] + dWLCloudy[-1])
            print("DEBUG", type(self.WL), type(self._spectrum.to(u.Lsun/u.AA).value), type(wlEdges))
            wlSpecNew = bin_luminosity(wlBPASS, np.array([wlBPASS, self._spectrum.to(u.Lsun/u.AA).value]), bins=wlEdges)
            self._spectrum = wlSpecNew[1]
            
    def get_spectrum(self, units="esAc", dist_norm=10.0*u.pc):
        """_summary_

        Args:
            units (str, optional): Units to output spectrum to. We have:
                    - "LsolA": L_Sun/Ang
                    - "esAc": erg/s/cm2/Ang
                Defaults to "esAc".
            dist_norm (float, optional): Given distance observing spectrum. Default is 10 pc.
        """
        if units == "LsolA":
            return self._spectrum
        elif units == "esAc":
            norm = 4 * np.pi * dist_norm**2
            spec_ = self._spectrum/norm
            return(spec_.to(u.erg / u.s / u.cm**2 / u.AA))
        
# I'm sorry for copying this from hoki but I need to know what's not working
# I keep passing numpy arrays, but numba is complaining it's a pandas.Series SOMEWHERE
def bin_luminosity(wl, spectra, bins=10):
    """
    Bin spectra conserving luminosity.

    Given spectra sampled at certain wavelengths/frequencies will compute their
    values in given wavelength/frequency bins. These values are bin averages
    computed using trapezoidal integration, which ensures that the luminosity
    per bin is conserved. Of course, only downsampling really makes sense here,
    i.e. the input spectra should be well sampled compared to the desired
    output bins.

    Effectively converts input spectra to step functions of
    wavelength/frequency. Note, in particular, that this means that only
    rectangle rule integration can sensibly be performed on the output
    spectra. Other integration methods are not meaningful.

    Parameters
    ----------
    wl : `numpy.ndarray` (N_wl,)
        Wavelengths or frequencies at which spectra are known.
    spectra : `numpy.ndarray` (N, N_wl)
        The spectra to bin given as L_lambda [Energy/Time/Wavelength] or L_nu
        [Energy/Time/Frequency] in accordance with `wl`.
    bins : int or `numpy.ndarray` (N_edges,), optional
        Either an integer giving the number `N_bins` of desired equal-width
        bins or an array of bin edges, required to lie within the range given
        by `wl`. In the latter case, `N_bins=N_edges-1`.

    Returns
    -------
    wl_new : `numpy.ndarray` (N_bins,)
        The wavelength/frequency values to which spectra were binned,
        i.e. centre bin values.
    spectra_new : `numpy.ndarray` (N, N_bins)
        The binned spectra.

    Notes
    -----
    For the actual integration, `wl` has to be sorted in ascending or
    descending order. If this is not the case, `wl` and `spectra` will be
    sorted/re-ordered. `bins` will always be sorted in the same order as `wl`
    as it is assumed to generally be relatively small.

    Although the language used here refers to spectra, the primary intended
    application area, the code can naturally be used to bin any function with
    given samples, conserving its integral bin-wise.
    """
    for arr, ndim in zip([wl, spectra, bins], [[1], [2], [0, 1]]):
        if np.ndim(arr) not in ndim:
            raise ValueError("Wrong dimensionality of input arrays.")
    if spectra.shape[1] != len(wl):
        raise ValueError("Shapes of `wl` and `spectra` are incompatible.")

    diff = np.diff(wl)
    if np.all(diff > 0):
        asc = True
    elif np.all(diff < 0):
        asc = False
    else:
        if np.any(diff == 0):
            raise ValueError("Identical values provided in `wl`.")
        ids = np.argsort(wl)
        wl = wl[ids]
        spectra = spectra[:, ids]
        asc = True

    if isinstance(bins, numbers.Integral):
        bins = np.linspace(wl[0], wl[-1], num=bins+1)
    else:
        if asc:
            bins = np.sort(bins)
        else:
            bins = bins[np.argsort(-1*bins)]
    if not (np.amax(bins) <= np.amax(wl) and np.amin(bins) >= np.amin(wl)):
        raise ValueError("Bin edges outside of valid range!")

    wl_new = (bins[1:] + bins[:-1])/2
    spectra_new = _binwise_trapz_sorted(wl, spectra, bins) \
        / np.diff(bins)

    return wl_new, spectra_new


@jit(nopython=True, nogil=True, cache=True)
def _binwise_trapz_sorted(x, y, bin_edges):
    """
    Trapezoidal integration over bins.

    Integrate each row of `y(x)` over each bin defined by `bin_edges` using
    trapezoidal integration. The values of `bin_edges` do not have to coincide
    with values given in `x`, the rows of `y` are linearly interpolated
    correspondingly.

    Parameters
    ----------
    x : `numpy.ndarray` (N_x,)
        `x`-values corresponding to each column of `y`. Assumed to be sorted in
        ascending or descending order. Integrated values will be negative for
        descending order.
    y : `numpy.ndarray` (N, N_x)
        N functions of `x` evaluated at each of its values.
    bin_edges : `numpy.ndarray` (N_bins+1,)
        Edges of the bins over which to perform integration. Assumed to be
        sorted in same order as `x` and to span a range <= the range spanned by
        `x`.

    Returns
    -------
    res : `numpy.ndarray` (N, N_bins)
        Integral over each bin of each row of `y`.
    """
    res = np.empty((y.shape[0], len(bin_edges)-1))

    i1 = 0
    i2 = 0
    y1 = np.empty((y.shape[0]))
    y2 = np.empty((y.shape[0]))
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
