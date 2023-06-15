import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

from cgauss_smooth import recenter, gaussErf, gaussLoop

import numpy as np
from pycosie.cluster.skid import SkidCatalog
import yt
import h5py as h5
import astropy.units as u
import astropy.constants as c
from yt.utilities.cosmology import Cosmology
from scipy.spatial.distance import pdist, squareform

class GalaxyGrid():
    
    #__recenter = __recenter
    #__gridGaussLoop = __gridGaussLoop
    #__gaussIntgErf = __gaussIntgErf
    
    def __init__(self, id, dsSphere, ds, gridLength, metals=None):
        self.id = id
        self.metalDensityGrids = dict()
        # TD metallicities enumerated "Metallicity_0X"
        # 0:C, 1:O, 2:Si, 3:Fe, 4:N, 5:Ne, 6:Mg, 7:S, 8:Ca, 9:Ti, in that order
        if metals == None:
            __metalArr = ["C","O","Si","Fe","N","Ne","Mg","S","Ca","Ti"]
        else:
            __metalArr = metals
        
        for mi in range(len(__metalArr)):
            self.metalDensityGrids[__metalArr[mi]] = np.zeros((gridLength, gridLength, gridLength))
        
        __gPartCoord = sp["PartType0","Coordinates"].to("kpccm/h").value # ckpc/h
        __sPartCoord = sp["PartType4","Coordinates"].to("kpccm/h").value
        
        xMin = min(__gPartCoord[:,0]) # getting min and max of each cartesian axis
        xMax = max(__gPartCoord[:,0])
        yMin = min(__gPartCoord[:,1])
        yMax = max(__gPartCoord[:,1])
        zMin = min(__gPartCoord[:,2])
        zMax = max(__gPartCoord[:,2])
        
        # Need to recenter coordinates around galaxy and not split around periodic boundaries
    
        Dx = abs(xMax - xMin)
        Dy = abs(yMax - yMin) # Finding distance span
        Dz = abs(zMax - zMin)
        
        __domainWidth = ds.domain_width.to("kpccm").value
        __gPartCoord = recenter(__gPartCoord, __domainWidth, Dx, Dy, Dz)
        __sPartCoord = recenter(__sPartCoord, __domainWidth, Dx, Dy, Dz)

        xMin = min(__gPartCoord[:,0]) # calculate new transformed coordinates
        xMax = max(__gPartCoord[:,0])
        yMin = min(__gPartCoord[:,1])
        yMax = max(__gPartCoord[:,1])
        zMin = min(__gPartCoord[:,2])
        zMax = max(__gPartCoord[:,2])
        Dx = abs(xMax - xMin)
        Dy = abs(yMax - yMin) # Finding distance span
        Dz = abs(zMax - zMin)
        
        L = max([Dx,Dy,Dz])
        
        self.zoomLength = ds.cosmology.arr(L, "kpccm/h")
        
        __gPartSL = sp["PartType0","SmoothingLength"].to("kpccm/h").value #ckpc/h
        __gPartMass =  sp["PartType0","Masses"].to("Msun").value #Msol
        __gPartZ = sp["PartType0","metallicity"].value # unitless
        __gPartZarr = []
        for i in range(10):
            __gPartZarr.append( sp["PartType0", f"Metallicity_{i:02}"].value ) # unitless
            
        # So indexing of __gPartZarr is (index of species in metal_arr, index of gas particle)
        
        __gPartTemperature = sp["PartType0","Temperature"].to("K").value # K
        
        dVcell = (L/gridLength)**3
        
        self.densityGrid = np.zeros((gridLength, gridLength, gridLength))
        for i in range(len(__gPartCoord)):
            __gaussGrid = gaussLoop(gridLength, __gPartCoord[i], __gPartSL[i], L)
            __denGrid = __gPartMass[i]* __gaussGrid / dVcell
            for mi in range(len(__metalArr)):
                self.metalDensityGrids[__metalArr[mi]] = self.metalDensityGrids[__metalArr[mi]] + (__denGrid * __gPartZarr[mi, i])
            self.densityGrid = self.densityGrid + __denGrid
            
        
#
#
#
        
class GalaxyGridDataset():
    
    def __init__(self, ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac):
        
        __skidMstarArr = skidcat.stellar_mass.to["Msun"]
        __skidIDArr = skidcat.ids
        
        self.galaxyGridsList = []
        for i in range(len(__skidIDArr)):
            rvir_i = __get_rvir(__skidMstarArr[i], snapname, fstar, deltac)
            r_s = rvir_frac * rvir_i.to("kpccm/h")
            center = skidcat.pos[idx_max]
            sp = ds.sphere(center, r_s)
            galGrid = GalaxyGrid(__skidIDArr[i], sp, ds, )
    
    def __get_rvir(Mstar, snapname, fstar, deltac):
    
        f = h5.File(snapname, "r")
        Ob = f["Header"].attrs["OmegaBaryon"]
        Om = f["Header"].attrs["Omega0"]
        Ol = f["Header"].attrs["OmegaLambda"]
        z = f["Header"].attrs["Redshift"]
        h = f["Header"].attrs["HubbleParam"]
    
        Ms = Mstar.value * u.Msun
        rhoc = ((3 * (h*100 * u.km / u.s / u.Mpc)**2) / (8 * np.pi * c.G)).decompose() * (Om*(1+z)**3 + Ol)
        Mvir = (Ms/fstar) * (Om/Ob)
        Rvir_p = np.power( ((3*Mvir)/(4*np.pi*rhoc*(1+deltac))), 1/3).to(u.kpc)
        Rvir = ds.arr(Rvir_p, "kpc")
    
        return(Rvir)
    
    def save(self, filedirname):
        print("Doesn't save yet")
        
#
#
#

def make_galaxy_grids(snapname, statname, nproc=1, fstar=0.1, deltac = 200.0, rvir_frac = 0.15):
    
    yt.set_log_level(0)
    ds = yt.load(snapname)
    skidcat = SkidCatalog(statname, ds)
    
    galGridDs = GalaxyGridDataset(ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac)
    
    return galGridDs