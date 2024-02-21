import numpy as np
import astropy.units as u 

# Working in h5py WAY faster, need some units handling.

# These are defaults for Gadget that seem consistent per version:

hDefault = 0.6774
defaultMass = 1.0e10 * u.Msun / hDefault 
defaultLength = 1.0 * u.kpc / hDefault
defaultVelocity = 1.o * u.km / u.sec

# These are special, in that they are set by Kristian

defaultStellarFormTime = 1.0 * u.yr
defaultSFR = 1.0 * u.Msun/u.yr

class TDUnits:
    
    def __init__(self, unit_dict=None, h=hDefault) -> None:
        if type(unit_dict) == None:
            self.mass = defaultMass
            self.length = defaultLength
            self.velocity = defaultVelocity
        else:
            self.mass = unit_dict["mass"]
            self.length = unit_dict["length"]
            self.velocity = unit_dict["velocity"]

        self.h = h