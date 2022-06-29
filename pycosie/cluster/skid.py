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
        
        def __init__(self, statname, snapname=None, unit_base=None, bounding_box=None, cosmo=None, L=None):
                """
                For this catalog defaults, masses are in Msun and position is ckpc/h.
                
                statname: str
                    The name of the .stat file outputted from SKID.
                snapname: str
                    The name of the snapshot used for SKID analysis. Use 0th file
                    if multifile output.
                unit_base: dict
                    Units for yt as it loads snapshot, same as the `unit_base` there.
                bounding_box: array-like
                    Box bounds for yt to load snapshot, same as `bounding_box` there.
                cosmo: yt.Cosmology
                    The Cosmology object defining the parameters of the simulation the 
                    catalog came from. This alongside L can be used instead of snapname.
                L: float
                    IN ckpc/h, the box length of the square simulation. Can be used 
                    alongside cosmo instead of snapname
                """
                # Check that only using snapname OR cosmo and L EXCLUSIVELY
                # Then creating units for each case
                if snapname==None:
                        if cosmo==None or L==None:
                                raise ValueError("Undefined parameters. Please either provide snapname (path to snapshot or yt Dataset) OR cosmo and L")
                        else:
                                H0 = cosmo.hubble_constant * 100. * u.km / u.Mpc / u.s
                                rho_c = 3 * H0**2 / (8 * np.pi * c.G)
                                rho_c = rho_c.to(u.g/u.cm**3)
                                L = cosmo.arr(L, "kpccm/h")
                                volSim = L.to("cmcm")**3
                                massUnit = cosmo.arr(rho_c.value * volSim.value, "g").to("Msun")
                                
                if snapname!=None:
                        if cosmo!=None or L!=None:
                                raise RuntimeError("Ambiguous definition. Please define snapname OR cosmo and L, but not all three.")
                        else:
                                if type(snapname)==str:
                                        ds = yt.load(snapname, unit_base=unit_base, bounding_box=bounding_box)
                                else:
                                        ds = snapname
                                H0 = ds.hubble_constant * 100. * u.km / u.Mpc / u.s
                                co = Cosmology(omega_lambda=ds.omega_lambda,
                                               omega_matter=ds.omega_matter,
	                                       omega_radiation=0,
	                                       omega_curvature=0,
	                                       hubble_constant=ds.hubble_constant)
                
                                rho_c = 3* H0**2 / (8 * np.pi * c.G)
                                rho_c = rho_c.to(u.g / u.cm**3)
                                L = ds.domain_width.to("kpccm/h")[0]
                                volSim = L.to("cmcm")**3
                                massUnit = co.arr(rho_c.value * volSim.value, "g").to("Msun")
                
                ID, npart, totMass, gasMass, stlMass, vcmax, hvc, ovc, rmvc, rhm, outr, dv, x, y, z, vx, vy, vz, xb, yb, zb = np.loadtxt(statname, unpack=True)
                pos = np.asarray([x, y, z]).T + 0.5
                self.total_mass = totMass * massUnit
                self.gas_mass = gasMass * massUnit
                self.stellar_mass = stlMass * massUnit
                self.pos = pos * L
                self.ids = ID.astype(int)


	

