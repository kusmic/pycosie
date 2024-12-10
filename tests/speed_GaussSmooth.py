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
#import time #used for debugging
import sys
from time import sleep
from datetime import datetime
from julia import Main

Main.include("../pycosie/utils/GaussSmooth.jl")
class GalaxyGridCython():    
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
        
class GalaxyGridJulia():    
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
    
    def __init__(self, id, sp, ds, gridLength, metals=None, star_SL_func=None, testidx=100):
        
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

        __gPartCoord = Main.recenter(__gPartCoord, __domainWidth, Dx, Dy, Dz)
        __sPartCoord = Main.recenter(__sPartCoord, __domainWidth, Dx, Dy, Dz)

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
            __gaussGrid = Main.gaussLoop(gridLength, __gPartCoord[i], __gPartSL[i], L)
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
    
class GalaxyGridDSCython():
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
            for i in range(totGalNum,100): 
                rvir_i = self.__get_rvir( __skidMstarArr[i], snapname, ds, fstar, deltac) 
                r_s = rvir_frac * rvir_i.to("kpccm/h")
                center = skidcat.pos[i]
                sp = ds.sphere(center, r_s)
                galGrid = self._testGalaxyGrid(__skidIDArr[i], sp, ds, grid_length, metals, star_SL_func) #self, id, dsSphere, ds, gridLength, metals=None, star_SL_func=None
                self.galaxyGridsList.append(galGrid)
                self.galaxyID.append(__skidIDArr[i])
                print(f"GalaxyGridDataset complete: {int(i)}/{totGalNum}", end='\r', flush=True)
            
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
    
    def __testGalaxyGrid(self, id, sp, ds, gridLength, metals=None, star_SL_func=None):
        return GalaxyGridCython(id, sp, ds, gridLength, metals=None, star_SL_func=None)
    
class GalaxyGridDSJulia(GalaxyGridDSCython):
        
    def __testGalaxyGrid(self, id, sp, ds, gridLength, metals=None, star_SL_func=None):
        return GalaxyGridJulia(id, sp, ds, gridLength, metals=None, star_SL_func=None)
    
if __name__ == "__main__": 
    # all defined classes are copy-paste from galgrid.py, changing
    # running the gridding whether C or Julia
    
    # setting test dataset
    # using TD pl16.5n704 snapshot 003 for initial tests
    # please rename the below variable if using different path
    snapname = "/data1/kfinlator/runs/joker/pl16p5n704/snapdir_003/snap_pl16p5n704_003.0.hdf5"
    statname = "/data1/samirk/research/pl16p5n704/Gal/gal_003.stat"
    print("Loading test set...")
    ds = yt.load(snapname)
    skidcat = SkidCatalog(statname, ds)
    
    print("Beginning Cython run.")
    ti = datetime.now()
    test = GalaxyGridDSCython(ds, skidcat, snapname, 1, 0.00316, 200.0, 0.15, 16, None, None)
    tf = datetime.now()
    finalTime = ti - tf
    print("Time for Cython/C++ (sec): ", finalTime.total_seconds())
    
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print("Beginning Julia run.")
    ti = datetime.now()
    test = GalaxyGridDSJulia(ds, skidcat, snapname, 1, 0.00316, 200.0, 0.15, 16, None, None)
    tf = datetime.now()
    finalTime = ti - tf
    print("Time for Cython/C++ (sec): ", finalTime.total_seconds())
    