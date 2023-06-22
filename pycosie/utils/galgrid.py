import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

from cgauss_smooth import recenter, gaussLoop

import numpy as np
from pycosie.cluster.skid import SkidCatalog
import yt
import h5py as h5
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import interp1d


class GalaxyGrid():
    
    #__recenter = __recenter
    #__gridGaussLoop = __gridGaussLoop
    #__gaussIntgErf = __gaussIntgErf
    
    def __init__(self, id, sp, ds, gridLength, metals=None, star_SL_func=None):
        
        if star_SL_func == None:
            star_SL_func = interp1d([91450.403, 188759.328], [0.100, 0.300], kind="linear", fill_value="extrapolate") # stellar mass in Msun to smoothing length in ckpc/h

        self.id = id
        self.gasMetalDensityGrids = dict()
        # TD metallicities enumerated "Metallicity_0X"
        # 0:C, 1:O, 2:Si, 3:Fe, 4:N, 5:Ne, 6:Mg, 7:S, 8:Ca, 9:Ti, in that order
        if metals == None:
            __metalArr = ["C","O","Si","Fe","N","Ne","Mg","S","Ca","Ti"]
        else:
            __metalArr = metals
        
        for mi in range(len(__metalArr)):
            self.gasMetalDensityGrids[__metalArr[mi]] = np.zeros((gridLength, gridLength, gridLength))
        
        __gPartCoord = sp["PartType0","Coordinates"].to("kpccm/h").value # ckpc/h
        __sPartCoord = sp["PartType4","Coordinates"].to("kpccm/h").value
        
        xMin = np.min(__gPartCoord[:,0]) # getting min and max of each cartesian axis
        xMax = np.max(__gPartCoord[:,0])
        yMin = np.min(__gPartCoord[:,1])
        yMax = np.max(__gPartCoord[:,1])
        zMin = np.min(__gPartCoord[:,2])
        zMax = np.max(__gPartCoord[:,2])
        
        # Need to recenter coordinates around galaxy and not split around periodic boundaries
    
        Dx = abs(xMax - xMin)
        Dy = abs(yMax - yMin) # Finding distance span
        Dz = abs(zMax - zMin)
        
        __domainWidth = ds.domain_width.to("kpccm").value
        
        #print("before", __gPartCoord.shape )
        __gPartCoord = recenter(__gPartCoord, __domainWidth, Dx, Dy, Dz)
        __sPartCoord = recenter(__sPartCoord, __domainWidth, Dx, Dy, Dz)
        #print("after", __gPartCoord.shape )
        
        xMin = np.min(__gPartCoord[:,0]) # calculate new transformed coordinates
        xMax = np.max(__gPartCoord[:,0])
        yMin = np.min(__gPartCoord[:,1])
        yMax = np.max(__gPartCoord[:,1])
        zMin = np.min(__gPartCoord[:,2])
        zMax = np.max(__gPartCoord[:,2])
        Dx = np.abs(xMax - xMin)
        Dy = np.abs(yMax - yMin) # Finding distance span
        Dz = np.abs(zMax - zMin)
        
        L = max([Dx,Dy,Dz])
        
        self.zoomLength = ds.cosmology.arr(L, "kpccm/h")
        
        __gPartSL = sp["PartType0","SmoothingLength"].to("kpccm/h").value #ckpc/h
        __gPartMass =  sp["PartType0","Masses"].to("Msun").value #Msol
        
        __gPartZarr = []
        for i in range(10):
            __gPartZarr.append( sp["PartType0", f"Metallicity_{i:02}"].value ) # unitless
            
        # So indexing of __gPartZarr is (index of species in metal_arr, index of gas particle)
        
        __gPartTemperature = sp["PartType0","Temperature"].to("K").value # K
        
        dVcell = (L/gridLength)**3
        
        self.gasDensityGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        self.gasTemperatureGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        
        for s in __metalArr:
            self.gasMetalDensityGrids[s] = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        
        for i in range(len(__gPartCoord)):
            __gaussGrid = gaussLoop(gridLength, __gPartCoord[i], __gPartSL[i], L)
            __mT = __gPartMass[i]* __gaussGrid  * __gPartTemperature[i]
            __denGrid = __gPartMass[i]* __gaussGrid / dVcell
            for mi in range(len(__metalArr)):
                self.gasMetalDensityGrids[__metalArr[mi]] = self.gasMetalDensityGrids[__metalArr[mi]] + (__denGrid * __gPartZarr[mi][i])
            self.gasDensityGrid = self.gasDensityGrid + __denGrid
            self.gasTemperatureGrid = self.gasTemperatureGrid + __mT
            
        self.gasTemperatureGrid = self.gasTemperatureGrid / (self.gasDensityGrid * dVcell)
            
        __sPartMass = sp["PartType4","Masses"].to("Msun").value
        __sPartZ = sp["PartType4","metallicity"].value
        __sPartNStar = sp["PartType4","NstarsSpawn"].value
        __sPartSFT = sp["PartType4","StellarFormationTime"].value
        
        
        __sPartZarr = []
        for i in range(10):
            __sPartZarr.append( sp["PartType0", f"Metallicity_{i:02}"].value )
        
        self.starMassGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        self.starNSpawnGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        self.starSFTGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        self.starMetallicityGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        
        self.starMetalMassGrids = dict()
        for s in __metalArr:
            self.starMetalMassGrids[s] = np.zeros((gridLength, gridLength, gridLength), dtype=float)
        
        for i in range(len(__sPartCoord)):
            starSL = star_SL_func(__sPartMass[i])
            __gaussGrid = gaussLoop(gridLength, __sPartCoord[i], starSL, L)
            self.starMassGrid = self.starMassGrid + __sPartMass[i] * __gaussGrid
            self.starNSpawnGrid = self.starNSpawnGrid + __sPartNStar[i] * __gaussGrid
            self.starSFTGrid = self.starSFTGrid + __sPartSFT[i] * __gaussGrid
            self.starMetallicityGrid = self.starMetallicityGrid + __sPartZ * __gaussGrid * __sPartMass[i]
            for mi in range(len(__metalArr)):
                self.starMetalMassGrids[__metalArr[mi]] = self.starMetalMassGrids[__metalArr[mi]] + (__sPartMass[i] * __sPartZarr[mi][i] * __gaussGrid) 
                
        self.starMetallicityGrid = self.starMetallicityGrid / self.starMassGrid

#
#
#
        
class GalaxyGridDataset():
    
    def __init__(self, ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals=None, star_SL_func=None):
        
        __skidIDArr = skidcat.ids
        __skidMstarArr = skidcat.stellar_mass
        self.galaxyGridsList = []
        self.galaxyID = []
        
        for i in range(len(__skidIDArr)):
            rvir_i = self.__get_rvir( __skidMstarArr[i], snapname, ds, fstar, deltac) 
            r_s = rvir_frac * rvir_i.to("kpccm/h")
            center = skidcat.pos[i]
            sp = ds.sphere(center, r_s)
            galGrid = GalaxyGrid(__skidIDArr[i], sp, ds, grid_length, metals, star_SL_func) #self, id, dsSphere, ds, gridLength, metals=None, star_SL_func=None
            self.galaxyGridsList.append(galGrid)
            self.galaxyID.append(__skidIDArr[i])
    
    def __get_rvir(self, Mstar, snapname, ds, fstar, deltac):
        
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
        print("Doesn't save yet to npy")
        
    def load(self, filedirname):
        print("Doesn't load yet from npy")
#
#
#

def make_galaxy_grids(snapname, statname, grid_length=64, nproc=1, fstar=0.1, deltac = 200.0, rvir_frac = 0.15, metals=None, star_SL_func=None):
    
    yt.set_log_level(0)
    ds = yt.load(snapname)
    skidcat = SkidCatalog(statname, ds)
    
    galGridDs = GalaxyGridDataset(ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals, star_SL_func)
    # self, ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals=None, star_SL_func=None
    
    return galGridDs