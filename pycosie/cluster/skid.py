import numpy as np
import yt
from glob import glob
from unyt import unyt_array
import astropy.units as u
import astropy.constants as c
from yt.utilities.cosmology import Cosmology

class SkidCatalog():
        """
        Object to hold the mass and position of galaxy groups found in SKID
        """
        
        def __init__(self, statname, snapname):
                """
                For this catalog defaults, masses are in Msun and position is ckpc/h.
                
                statname: str
                    The name of the .stat file outputted from SKID.
                snapname: str
                    The name of the snapshot used for SKID analysis. Use 0th file
                    if multifile output.
                """
                ds = yt.load(snapname)
                H0 = ds.hubble_constant * 100. * u.km / u.Mpc / u.s
                co = Cosmology(omega_lambda=ds.omega_lambda,
                               omega_matter=ds.omega_matter,
	                       omega_radiation=0,
	                       omega_curvature=0,
	                       hubble_constant=ds.hubble_constant)
                
                rho_c = 3* H0**2 / (8 * np.pi * c.G)
                rho_c = rho_c.to(u.g / u.cm**3)
                L = ds.domain_width.to("kpccm/h")[0]
                volSim = L.to("cm")**3
                massUnit = co.arr(rho_c.value * volSim.value, "g").to("Msun")
                
                ID, totMass, gasMass, stlMass, x, y, z = np.loadtxt(statname, usecols=(0,2,3,4,12,13,14),
                                                                    unpack=True)
                pos = np.asarray([x, y, z]).T + 0.5
                self.total_mass = totMass * massUnit
                self.gas_mass = gasMass * massUnit
                self.stellar_mass = stlMass * massUnit
                self.pos = pos * L
                self.ids = ID


	

