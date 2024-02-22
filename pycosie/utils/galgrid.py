import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

from cgauss_smooth import recenter, gaussLoop, gaussErf

import numpy as np
from pycosie.cluster.skid import SkidCatalog
import yt
import h5py as h5
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import interp1d
import pickle
import multiprocessing as mp
import time
import sys
from time import sleep

def print_prog(counter, totGalNum):
    print(f"GalaxyGridDataset complete: {int(counter.value)}/{totGalNum}", end='\r', flush=True)
    
class GalaxyGrid():
    
    #__recenter = __recenter
    #__gridGaussLoop = __gridGaussLoop
    #__gaussIntgErf = __gaussIntgErf
    
    """
        Attributes:
        - gasMetalMetallicityGrids: dict(array[float]): Stores mass-weighted metallicities of cell. Hardcoded keys are 
        ["C","O","Si","Fe","N","Ne","Mg","S","Ca","Ti"]
        - zoomLength: float: side length of square grid in ckpc/h
        - gasDensityGrid: array[float]: Holds density of grid in each cell in units Msun/(ckpc/h)^3
        - gasTemperatureGrid: array[float]: Holds temperature of grid in each cell in units K
        - starParticle: dict(array[int, array]): Dictionary holding star information. Key "id" has integer star particle
        ID from simulation. Key "pos" has position of star particle relative to origin of grid in ckpc/h
        - starCount: int: Number of stars in galaxy. Can be used as flag to check if stars in galaxy.
        
        
        """
    
    def __init__(self, id, sp, ds, gridLength, metals=None, star_SL_func=None):
        
        while True:
            if star_SL_func == None:
                star_SL_func = interp1d([91450.403, 188759.328], [0.100, 0.300], kind="linear", fill_value="extrapolate") # stellar mass in Msun to smoothing length in ckpc/h

            self.id = id
            self.gasMetalMetallicityGrids = dict()
            # TD metallicities enumerated "Metallicity_0X"
            # 0:C, 1:O, 2:Si, 3:Fe, 4:N, 5:Ne, 6:Mg, 7:S, 8:Ca, 9:Ti, in that order
            if metals == None:
                __metalArr = ["C","O","Si","Fe","N","Ne","Mg","S","Ca","Ti"]
            else:
                __metalArr = metals

            for mi in range(len(__metalArr)):
                self.gasMetalMetallicityGrids[__metalArr[mi]] = np.zeros((gridLength, gridLength, gridLength))

            __gPartCoord = sp["PartType0","Coordinates"].to("kpccm/h").value # ckpc/h
            __sPartCoord = sp["PartType4","Coordinates"].to("kpccm/h").value
            __sPartID = sp["PartType4","ParticleIDs"].value
            __sPartZ = sp["PartType4","metallicity"].value
            __sPartM = sp["PartType4","Masses"].to("Msun").value
            __sPartT = ds.current_time.to("yr").value-sp["PartType4","StellarFormationTime"].value

            #print(len(__sPartCoord))
            self.starCount = len(__sPartCoord)
            

            #return None

            if len(__gPartCoord) < 1 and len(__sPartCoord) < 1: # no gas or no stars, do not consider
                print(f"No stars and no gas in galaxy {self.id}! Creating None data for all...\n")
                self.gasMetalMetallicityGrids = None
                self.zoomLength = None
                self.gasDensityGrid = None
                self.gasTemperatureGrid = None
                self.starParticle = None
                break
            
            elif len(__gPartCoord) < 1:
                print(f"No gas in galaxy {self.id}! Creating None data for only gas...\n")
                self.gasMetalMetallicityGrids = None
                self.zoomLength = None
                self.gasDensityGrid = None
                self.gasTemperatureGrid = None
                self.starParticle = {"id":[], "pos":[]}
                for si in range(len(__sPartCoord)):
                    self.starParticle["id"].append(int(__sPartID[si]))
                    self.starParticle["pos"].append(__sPartCoord[si])
                self.starParticle["pos"] = np.array(self.starParticle["pos"])
                break


            try:
                xMin = np.min( np.concatenate((__gPartCoord[:,0], __sPartCoord[:,0])) ) # getting min and max of each cartesian axis
                xMax = np.max( np.concatenate((__gPartCoord[:,0], __sPartCoord[:,0])) )
                yMin = np.min( np.concatenate((__gPartCoord[:,1], __sPartCoord[:,1])) )
                yMax = np.max( np.concatenate((__gPartCoord[:,1], __sPartCoord[:,1])) )
                zMin = np.min( np.concatenate((__gPartCoord[:,2], __sPartCoord[:,2])) )
                zMax = np.max( np.concatenate((__gPartCoord[:,2], __sPartCoord[:,2])) )

            except ValueError:
                print("gPx, ",__gPartCoord[:,0])
                print("gPy, ",__gPartCoord[:,1])
                print("gPz, ",__gPartCoord[:,2])
                print("FUCKING FUCKY-WUCKY!!!!")
                sys.exit()

                return
            # Need to recenter coordinates around galaxy and not split around periodic boundaries

            Dx = abs(xMax - xMin)
            Dy = abs(yMax - yMin) # Finding distance span
            Dz = abs(zMax - zMin)

            __domainWidth = ds.domain_width.to("kpccm/h").value

            __gPartCoord = recenter(__gPartCoord, __domainWidth, Dx, Dy, Dz)
            __sPartCoord = recenter(__sPartCoord, __domainWidth, Dx, Dy, Dz)

            xMin = np.min( np.concatenate((__gPartCoord[:,0], __sPartCoord[:,0])) ) # calculate new transformed coordinates
            xMax = np.max( np.concatenate((__gPartCoord[:,0], __sPartCoord[:,0])) )
            yMin = np.min( np.concatenate((__gPartCoord[:,1], __sPartCoord[:,1])) )
            yMax = np.max( np.concatenate((__gPartCoord[:,1], __sPartCoord[:,1])) )
            zMin = np.min( np.concatenate((__gPartCoord[:,2], __sPartCoord[:,2])) )
            zMax = np.max( np.concatenate((__gPartCoord[:,2], __sPartCoord[:,2])) )
            Dx = np.abs(xMax - xMin)
            Dy = np.abs(yMax - yMin) # Finding distance span
            Dz = np.abs(zMax - zMin)

            L = max([Dx,Dy,Dz])

            

            # putting zeropoint at mins

            for i in range(len(__gPartCoord)):
                __gPartCoord[i] = __gPartCoord[i] - np.array([xMin, yMin, zMin])
            for i in range(len(__sPartCoord)):
                __sPartCoord[i] = __sPartCoord[i] - np.array([xMin, yMin, zMin])
                
            self.originPoint = np.array([xMin, yMin, zMin])
            self.zoomLength = ds.cosmology.arr(L, "kpccm/h")
            self.starParticle = {"id":[], "pos":[],"mass":[],"z":[],"age":[]}
            for si in range(len(__sPartCoord)):
                self.starParticle["id"].append(int(__sPartID[si]))
                self.starParticle["pos"].append(__sPartCoord[si])
                self.starParticle["mass"].append(__sPartM[si])
                self.starParticle["z"].append(__sPartZ[si])
                self.starParticle["age"].append(__sPartT[si])
            self.starParticle["id"] = np.array(self.starParticle["id"], dtype=int)
            self.starParticle["pos"] = np.array(self.starParticle["pos"])
            self.starParticle["mass"] = np.array(self.starParticle["mass"])
            self.starParticle["z"] = np.array(self.starParticle["z"])
            self.starParticle["age"] = np.array(self.starParticle["age"])

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


            for i in range(len(__gPartCoord)):
                __gaussGrid = gaussLoop(gridLength, __gPartCoord[i], __gPartSL[i], L)
                __mT = __gPartMass[i]* __gaussGrid  * __gPartTemperature[i]
                __massGrid = __gPartMass[i]* __gaussGrid
                __denGrid = __gPartMass[i]* __gaussGrid / dVcell
                for mi in range(len(__metalArr)):
                    self.gasMetalMetallicityGrids[__metalArr[mi]] = self.gasMetalMetallicityGrids[__metalArr[mi]] + (__massGrid * __gPartZarr[mi][i])
                self.gasDensityGrid = self.gasDensityGrid + __denGrid
                self.gasTemperatureGrid = self.gasTemperatureGrid + __mT
                
            for mi in range(len(__metalArr)):
                self.gasMetalMetallicityGrids[__metalArr[mi]] = self.gasMetalMetallicityGrids[__metalArr[mi]] / (self.gasDensityGrid * dVcell)

            self.gasTemperatureGrid = self.gasTemperatureGrid / (self.gasDensityGrid * dVcell)

            #if len(__sPartCoord) < 1: # no stars, do not consider
            #    print(f"No stars in galaxy {self.id}! Creating None data...\n")
            #    self.starMassGrid = None
            #    self.starSFTGrid = None
            #    self.starMetallicityGrid = None
            #    break
#
            #__sPartMass = sp["PartType4","Masses"].to("Msun").value
            #__sPartZ = sp["PartType4","metallicity"].value
            #__sPartNStar = sp["PartType4","NstarsSpawn"].value
            #__sPartSFT = sp["PartType4","StellarFormationTime"].value # in scale factor
            #__sTMax = sp["PartType4","TemperatureMax"].value
#
#
            #__sPartZarr = []
            #for i in range(10):
            #    __sPartZarr.append( sp["PartType0", f"Metallicity_{i:02}"].value )
#
            #self.starMassGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
            #self.starSFTGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
            #self.starMetallicityGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
            #self.starTemperatureMaxGrid = np.zeros((gridLength, gridLength, gridLength), dtype=float)
#
            #self.starMetalMassGrids = dict()
            #for s in __metalArr:
            #    self.starMetalMassGrids[s] = np.zeros((gridLength, gridLength, gridLength), dtype=float)
#
            #for i in range(len(__sPartCoord)):
            #    starSL = star_SL_func(__sPartMass[i])
            #    __gaussGrid = gaussLoop(gridLength, __sPartCoord[i], starSL, L)
            #    self.starMassGrid = self.starMassGrid + __sPartMass[i] * __gaussGrid
            #    self.starNSpawnGrid = self.starNSpawnGrid + __sPartNStar[i] * __gaussGrid
            #    self.starMetallicityGrid = self.starMetallicityGrid + (__sPartZ[i] * __gaussGrid * __sPartMass[i])
            #    self.starTemperatureMaxGrid = self.starTemperatureMaxGrid + (__sTMax[i] * __gaussGrid * __sPartMass[i])
            #    # hope I don't need different metal fractions, I don't know what they are it's 10x2 array WHAT IS THE 2?!    
#
            #self.starMetallicityGrid = self.starMetallicityGrid / self.starMassGrid
            #self.starTemperatureMaxGrid = self.starTemperatureMaxGrid / self.starMassGrid

            break

#
#
#
        
class GalaxyGridDataset():
    """
    Arrtibutes:
    - galaxyID: list[int]: Holds integer galaxy ID from SKID catalog
    - galaxyGridsList: list[GalaxyGrid]: Holds GalaxyGrid object for galaxy
    """
    
    def __init__(self, ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals=None, star_SL_func=None, filedirname=None):

        __skidIDArr = skidcat.ids
        __skidMstarArr = skidcat.stellar_mass
        self.galaxyGridsList = []
        self.galaxyID = []
        
        totGalNum = len(__skidIDArr)
        
        if type(filedirname) == type(None):
            filedirname = "SOMETHING"
        if nproc == 1:
            for i in range(totGalNum): 
                rvir_i = self.__get_rvir( __skidMstarArr[i], snapname, ds, fstar, deltac) 
                r_s = rvir_frac * rvir_i.to("kpccm/h")
                center = skidcat.pos[i]
                sp = ds.sphere(center, r_s)
                galGrid = GalaxyGrid(__skidIDArr[i], sp, ds, grid_length, metals, star_SL_func) #self, id, dsSphere, ds, gridLength, metals=None, star_SL_func=None
                self.galaxyGridsList.append(galGrid)
                self.galaxyID.append(__skidIDArr[i])
                print(f"GalaxyGridDataset complete: {int(counter.value)}/{totGalNum}", end='\r', flush=True)
            
        elif nproc > 1:
            def ggproc(idL, gridL, skidIDArr, skidMstarArr, ds, grid_length, metals, star_SL_func, counter):
                for i in range(len(skidIDArr)):
                    rvir_i = self.__get_rvir( skidMstarArr[i], snapname, ds, fstar, deltac) 
                    r_s = rvir_frac * rvir_i.to("kpccm/h")
                    center = skidcat.pos[i]
                    sp = ds.sphere(center, r_s)
                    idL.append(skidIDArr[i])
                    temp = GalaxyGrid(skidIDArr[i], sp, ds, grid_length, metals, star_SL_func)
                    gridL.append(temp)
                    counter.value += 1
                    print(f"GalaxyGridDataset complete: {int(counter.value)}/{totGalNum}", end='\r', flush=True)
            
            idxArr = np.linspace(0,totGalNum, nproc+1, dtype=int)
            manager = mp.Manager()
            proc_counter = manager.Value('i',0)
            
            grid_list = []
            id_list = []
            processes = []
            for i in range(nproc):
                id_list.append(manager.list())
                grid_list.append(manager.list())
                arg_tup = (id_list[i], grid_list[i], __skidIDArr[idxArr[i]:idxArr[i+1]], __skidMstarArr[idxArr[i]:idxArr[i+1]], ds, grid_length, 
                           metals, star_SL_func, proc_counter)
                processes.append( mp.Process(target=ggproc, args=arg_tup) )
                processes[i].start()
                
            for p in processes:
                while True:
                    if p.is_alive():
                        print_prog(proc_counter, totGalNum)
                        sleep(1)
                        continue
                    else:
                        print_prog(proc_counter, totGalNum)
                        sleep(1)
                        break
                    
            for i in range(len(id_list)):
                temp_id = list(id_list[i])
                temp_grid = list(grid_list[i])
                
            for j in range(len(temp_id)):
                self.galaxyID.append(temp_id[j])
                self.galaxyGridsList.append(temp_grid[i])
            
                
    
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
        Rvir_p = np.power( ((3*Mvir)/(4*np.pi*rhoc*(deltac))), 1/3).to(u.kpc)
        Rvir = ds.arr(Rvir_p, "kpc")
    
        return(Rvir)
        
        
#
#
#

def make_galaxy_grids(snapname, statname, grid_length=64, nproc=1, fstar=0.1, deltac = 200.0, rvir_frac = 0.15, metals=None, star_SL_func=None):
    """_summary_

    Args:
        snapname (str): Path and name of simulation snapshot
        statname (str): Path and name of SKID stat file for snapshot
        grid_length (int, optional): Length of grid cell in cells. Defaults to 64.
        nproc (int, optional): Number of processors to use. Defaults to 1.
        fstar (float, optional): Fraction of galaxy stellar mass to halo mass. Defaults to 0.1.
        deltac (float, optional): Overdensity. Defaults to 200.0.
        rvir_frac (float, optional): Fraction of virial radius to use for making grid. Defaults to 0.15.
        metals (array_like[str], optional): Holds string information of shorthand metal names, e.g. "C", "O". If none, then
        uses a predefined array. Defaults to None.
        star_SL_func (interp1d, optional): Scipy interp1d function used for smoothing length of star particles. If none,
        uses a predefined function. Defaults to None.

    Returns:
        GalaxyGridDataset: Dataset of grids for galaxies in snapshot
    """
    yt.set_log_level(0)
    print("Loading snapshots...")
    ds = yt.load(snapname)
    print("Loading SKID stat...")
    skidcat = SkidCatalog(statname, ds)
    print("Creating galaxy grid dataset...")
    galGridDs = GalaxyGridDataset(ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals, star_SL_func)
    # self, ds, skidcat, snapname, nproc, fstar, deltac, rvir_frac, grid_length, metals=None, star_SL_func=None
    print("Done.")
    
    return galGridDs

def save(ggds, filedirname):
        
    with open(filedirname, "wb") as fdn:
        pickle.dump(ggds, fdn, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data at {filedirname}")
    
    
def load(filedirname):
    
    with open(filedirname, "rb") as fdn:
        ggds = pickle.load(fdn)
    print(f"Loaded data from {filedirname}")
    return ggds