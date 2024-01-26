import numpy as np
from hoki import load
from hoki.spec import bin_luminosity
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
            self.WL = np.load("__cloudywave.npy")
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
            yLT = np.array([spectraL[LT], spectraH[LT]])
            specLT_interp = interp1d(x, yLT, axis=0)
            specLT = specLT_interp(self.zstar)
            
            LT = spectra_logageStr_arr[idxHigherT]
            yLT = np.array([spectraL[LT], spectraH[LT]])
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
            
            y = np.array([spectraL[LT], spectraL[HT]])
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
            
            y = np.array([spectraL[LT], spectraL[HT]])
            spec_interp = interp1d(age,y, axis=0)
            
            self._spectrum = spec_interp(np.log10(self.tstar)) * u.Lsun / u.AA
            
        if wl_arr == "bpass":
            self.WL = wlBPASS
        elif wl_arr != "bpass":
            dWLCloudy = np.gradient(self.WL)/2 # SHOULD BE CLOUDY OR CUSTOM WAVELENGTH ARR'
            _wlLower = self.WL - dWLCloudy
            wlEdges = np.concatenate(_wlLower, np.array([self.WL[-1] + dWLCloudy[-1]], dtype=float))
            wlSpecNew = bin_luminosity(wlBPASS, self._spectrum, bins=wlEdges)
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
