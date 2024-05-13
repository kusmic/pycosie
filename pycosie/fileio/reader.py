import numpy as np
from astropy.io import ascii
import astropy.units as u

class VPM():
    #"%4d %6.4lg %12.1f %6.3lg %6.4lg %6.3lg -1 %6.3lg %6.3lg\n",nLines, N/1.e+13, v_mean, b, sigma_N/1.e+13, sigma_b, ew, sigma_ew
    def __init__(self, fname):
        with open(fname) as f:
            content = f.read().split("\n")
            x = content[1].split(" ")
            self.line = x[0]
        ids, colDens, vm, absWidth, colDensUnc, absWidthUnc, min1, EW, EWUnc= np.loadtxt(fname, skiprows=2, unpack=True)
        self.line_id = ids
        self.column_density = colDens * 1e13 / u.cm**2
        self.v_mean = vm * u.km / u.s
        self.v_width = absWidth * u.km / u.s
        self.column_density_unc = colDensUnc * 1e13 / u.cm**2
        self.v_width_unc = absWidthUnc * u.km / u.s
        self.EW = EW * u.Angstrom
        self.EW_unc = EWUnc * u.Angstrom

class VPM_Merged():
    
    def __init__(self, fname):
        with open(fname) as f:
            content = f.read().split("\n")
            x = content[1].split(" ")
            self.line = x[0]
        ids, colDens, vm, absWidth, EW, EWUnc= np.loadtxt(fname, skiprows=2, unpack=True)
        self.line_id = ids
        self.column_density = colDens * 1e13 / u.cm**2
        self.central_velocity = vm * u.km / u.s
        self.v_width = absWidth * u.km / u.s
        self.EW = EW * u.Angstrom
        self.EW_unc = EWUnc * u.Angstrom

class CLN():

    def __init__(self, fname):
        self.line = fname[:-8]
        lam, v, F, dF= np.loadtxt(fname, skiprows=2, unpack=True)
        self.wavelength = lam * u.Angstrom
        self.velocity = v * u.km / u.s
        self.norm_flux = F
        self.norm_flux_unc = dF

class Loser():

    def __init__(self, fname):
        l_id, UVmagDust, UVmag = np.loadtxt(fname, unpack=True, usecols=(0,25,41))
        self.ids = l_id.astype(int)
        self.UV_mag = UVmag
        self.UV_mag_dust = UVmagDust

