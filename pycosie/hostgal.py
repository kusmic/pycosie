import numpy as np
import h5py as h5
import glob
import yt
import caesar
from astropy.table import Table
import os
import multiprocessing as mp
import sys
import time
from datetime import datetime

from scipy.ndimage import convolve

# Okay, need simulation cosmology and size

# All absorption lines calculated in quasarcosie.
ion_lines = ("OI1302", "CIV1548", "CII1335", "MgII2796", "SiII1260", "SiIV1394")
N_LLP = 1625
UINT_MAX = 4294967295

xEdges = np.linspace(0.,1.,N_LLP)
yEdges = np.linspace(0.,1.,N_LLP)
zEdges = np.linspace(0.,1.,N_LLP)

#print(len(xEdges))

def __save_hdf5__(table, fname):
    # takes astropy Table object, saves into HDF5 file, each column is dataset
    # names=("SpeciesLine","AbsorberID", "GasParticleID", "GalaxyID")
    if os.path.isfile(fname):
        os.remove(fname)

    with h5.File(fname, "w") as f:
        print("Creating table...")
        refdata = f.create_group("ReferenceData")
        print("Saving species...")
        speclines = refdata.create_dataset("SpeciesLines", data=table["SpeciesLine"])
        print("Saving absorber IDs...")
        absorberID = refdata.create_dataset("AbsorberID", data=table["AbsorberID"])
        print("Saving gas IDs...")
        gasID = refdata.create_dataset("GasParticleID", data=table["GasParticleID"])
        print("Saving galaxy IDs...")
        galID = refdata.create_dataset("GalaxyIDs", data=table["GalaxyIDs"])
        print("Creating lookup...")
        specieslist = f.create_group("SpeciesLookup")
        for ioni, ion in enumerate(ion_lines):
            specieslist.attrs[ion] = ioni
    print(f"\n\nSaved output to {fname}.")

def __print_complete(counter, N):
    # Takes Manager.Value object starting at 0, integer total of particles N
    # Prints out completed particles
    percent = f"{counter.value} / {N}"
    percentage = f"..... {percent} launched particles processed"
    print(percentage, end="\r")

def __part__(gasIDArr, gasCoordArr, gasLLPArr, vpmDict, galPosArr, galIDArr, colSpecies, gasIDOut, vpmIDOut, galIDOut, __debugMode__, N, z, Hz, h, DXDZ, DYDZ, r_search, lbox, counter, maxCountGal, gal_buffer):
    # The searching

    for gi in range(gasIDArr.size):
        counter.value += 1
        # rint(f"..... {percent:.3f}% launched particles processed", end="\r")
        pos = gasCoordArr[gi] # This is in box units
        # percent = (gi+1)
        for ioni, ion in enumerate(ion_lines):
            sysID = vpmDict[ion]["ID"]
            sysVel = vpmDict[ion]["v"]
            for vi, v in enumerate(sysVel):
                xStart, yStart, zStart = 0., 0., 0.
                dSys = (v / Hz) * (1+z) * h # comoving h-1 Mpc
                dSys /= lbox.to("Mpccm/h").value[2] # in box units
                dz = dSys / np.sqrt(DXDZ**2 + DYDZ**2 + 1) # In box
                dx = dz * DXDZ
                dy = dz * DYDZ
                xSys = xStart + dx 
                xSys -= np.floor(xSys)
                ySys = yStart + dy
                ySys -= np.floor(ySys)
                zSys = zStart + dz
                zSys -= np.floor(zSys)

                # ADD check if x, y, z already larger than r_search
                r_s = r_search / lbox.to("kpccm/h").value[0]
                if xSys > r_s or ySys > r_s or zSys > r_s:
                    continue
                
                rSys = np.array([xSys, ySys, zSys])
                rDiff = np.sqrt( np.sum( (pos-rSys)**2 ) ) # box unit - box unit
                    
                if rDiff > r_s:
                    continue
                else:
                    # print("GOT!")
                    LLP = gasLLPArr[gi]
                    # UINT_MAX check up here
                    if int(LLP) == UINT_MAX:
                        continue
                    # LLP grid indices calculated
                    igas = int( LLP/(N_LLP**2) )
                    jgas = int( (LLP - igas*N_LLP**2)/N_LLP )
                    kgas = int( LLP - (igas*N_LLP+jgas)*N_LLP )
                    # if debug mode ran
                    if __debugMode__:
                        crit1 = (igas >= 1625) or (jgas >= 1625) or (kgas >= 1625)
                        crit2 = (igas < 0) or (jgas < 0) or (kgas < 0)
                        if crit1 or crit2:
                            print(LLP, pos)
                            partID = gasIDs[gi]
                            f.write(f"{partID} {LLP} {pos[0]} {pos[1]} {pos[2]} {igas} {jgas} {kgas}\n")
                        continue
                    
                    galOfLLP = np.ones(int(maxCountGal)) * -99
                    ind_ret = 0

                    # GEtting index of LLP grid
                    # Getting index ranges considering index buffer
                    
                    xmin_ind = igas - gal_buffer
                    xmax_ind = igas + gal_buffer +1

                    ymin_ind = jgas - gal_buffer
                    ymax_ind = jgas + gal_buffer +1

                    zmin_ind = kgas - gal_buffer
                    zmax_ind = kgas + gal_buffer +1
                    for gali, galp in enumerate(galPosArr): # should be in ckpc/h
                        
                        # Now checking for galaxies that are in the grid cells
                        galp /= lbox.to(kpccm/h).value[0] #CHECKME if outputs still bad can have bad units here

                        if xmin_ind < 0: # negative index
                            # Check either near end [min, 1] or periodic wrap to beginning [0,max]
                            inX = (galp[0] >= xEdges[xmin_ind] and galp[0] < 1.) or (galp[0] >= 0. and galp[0] < xEdges[xmax_ind])
                        else:
                            inX = galp[0] >= xEdges[xmin_ind] and galp[0] < xEdges[xmax_ind]
                            
                        if ymin_ind < 0: # negative index
                            # Check either near end [min, 1] or periodic wrap to beginning [0,max]
                            inY = galp[1] >= yEdges[ymin_ind] and galp[1] < 1. or (galp[1] >= 0. and galp[1] < yEdges[ymax_ind])
                        else:
                            inY = galp[1] >= yEdges[ymin_ind] and galp[1] < yEdges[ymax_ind]

                        if zmin_ind < 0: # negative index
                            # Check either near end [min, 1] or periodic wrap to beginning [0,max]
                            inZ = galp[2] >= zEdges[zmin_ind] and galp[2] < 1. or (galp[2] >= 0. and galp[2] < zEdges[zmax_ind])
                        else:
                            inZ = galp[2] >= zEdges[zmin_ind] and galp[2] < zEdges[zmax_ind]
                        
                        if inX and inY and inZ:
                            # if galaxy in grid cell, hold as potential host
                            galOfLLP[ind_ret] = galID[gali]
                            ind_ret += 1
                        # Now extracting all data out
                    # print(galOfLLP)
                    colSpecies.append(ioni)
                    gasIDOut.append(gasIDArr[gi])
                    vpmIDOut.append(sysID[vi])
                    galIDOut.append(galOfLLP)
                    # print(ion, gasIDs[gi], sysID[vi], galOfLLP)


def do_hostgals(vpmpath, simpath, caesarpath, r_search, bbox=None, unit_base=None, n_i=0, n_f=None, merged=True, N_LLP=N_LLP, multifile=True, write=True, __debugMode__ = False, gal_bfr=1, nproc=1):

    if __debugMode__:
        f = open("particle_LLP_debug.txt", "w")
        f.write("ParticleID LLP x y z i j k\n")
        f.close()
        f = open("particle_LLP_debug.txt", "a")

    """
    It should take the absorption features of all species outputted and relate
    them back to a point in the simulation box. This will only work with VPM
    and HDF5 files.
    
    I:
    - vpmpath: string of the path to the directory of vpm files
    - simpath: string of the path to the directory of simulation snapshots
    - caesarpath: string of path to directory of caesar ouptuts
    - photspecpath: string of path to pyloser outputs
    - r_search: IN COMOVING KPC/h the searching radius for absorption system, 
      looking for gas particles
    - bbox: 3x2 matrix array showing starting and ending value of simulation
      box, in units of length provided in "units"
    - unit_base: dictionary holding unit information, based on yt-project input
      for yt_load
    - n_i: the integer of the first snapshot to start on, i.e. an input of 
      4 should correlate to identifier '004', input of 23 should correlate to
      identifier '023', etc. Default: 0
    - n_f: the integer of the final snapshot. If it is 'None' then it finds the
      maximum amount by counting all the outputs in the qcdata directory, not
      from the autovp directory. The method is just counting the amount of
      files total in that directory N, as that should be the number of snapshots
      that one is working with. Finally, it calculates the true snapshot number
      n_f = N + n_i. Default: None
    - merged: boolean to determine if it should only work on the VPM.MERGED
      output instead of the regular VPM output. Cannot do both, or really, you
      should not do both. Default: True
    - N_LLP: number of grids of one side used in last launch calculations. This
      means for your grid over the box, you will work with N_LLP**3 cells.
    - multifile: bool to show whether snapshot files outputted in multifile
      mode, e.g. snap_...0XY.0.hdf5, snap_....1.hdf5, etc. Default is True.
    - write: bool to designate if you want to save the reference output as
      a FITS table. Default is True.
    - __debugMode__: DEVELOPER TOOL. Used to debug issues on gas finding. 
      Outputs an ASCII table of particles found with bag flags.

    O: Reference table
    """

    print(f"Run on {datetime.now()}\n")
    DXDZ = 0.35171 # differential movement w.r.t. z for x and y, used to find
    DYDZ = 0.0113  # the casting angle. Hard-coded in arcosie programs.
    # In box length units

    # Standardizing path strings
    if '/' not in vpmpath[-1]:
        vpmpath = vpmpath + "/"
    if '/' not in simpath[-1]:
        simpath = simpath + "/"
    if '/' not in caesarpath[-1]:
        caesarpath = caesarpath + "/"

    for snapi in range(n_i,n_f+1): # for each snapshot and associated data

        # Getting all files in each directory
        
        if merged:
            globStr = vpmpath + f"*_{snapi:03}.vpm.merged"
            vpmFiles = glob.glob(globStr)
        else:
            globStr = vpmpath + f"*_{snapi:03}.vpm"
            vpmFiles = glob.glob(globStr)

        if multifile:
            globStr = simpath + f"snapdir_{snapi:03}/snap_*.0.hdf5"
        else:
            globStr = simpath + "snap_*.hdf5"
        simFiles = glob.glob(globStr)[0]

        globStr = caesarpath + f"caesar_{snapi:03}.hdf5"
        caesarFiles = glob.glob(globStr)[0]
            
        # Getting snapshot's cosmology and size
        if bbox == None or unit_base == None:
            snapFile = yt.load(simFiles) # h5.File(simFiles[i], "r")
        else:
            snapFile = yt.load(simFiles, unit_base=unit_base, bounding_box=bbox)
        haloFile = caesar.load(caesarFiles)

        # Defining the cosmology and simulation size
        # DOING IT IN COMOVING
        lbox = snapFile.domain_width.to("kpccm/h")# ["Header"].attrs["BoxSize"]
        h = snapFile.hubble_constant #["Header"].attrs["HubbleParam"]
        OmegaM = snapFile.omega_matter # ["Header"].attrs["Omega0"]
        OmegaL = snapFile.omega_lambda # ["Header"].attrs["OmegaLambda"]
        z = snapFile.current_redshift # ["Header"].attrs["Redshift"]

        H0 = h * 100
        Hz = H0 * np.sqrt((OmegaM * np.power(1+z,3)) + OmegaL)
        # print(h, H0, Hz)

        # Getting gas information
        gasData = snapFile.all_data()
        # Getting galaxy IDs in caesar
        galID = np.asarray([i.GroupID for i in haloFile.galaxies])
        # And positions, in box length units
        galPos = np.asarray([i.pos.to("kpccm/h").value for i in haloFile.galaxies])
        # Now reading each VPM(.MERGED) file

        # This will hold all data of IDs for each sim snapshot
        colSpecies = []
        colGasID = []
        colAbsorberID = []
        colGalaxyID = []
        print(f"Working on snap {snapi:03}...")
        
        gasCoords_pre = gasData["PartType0", "Coordinates"].to("Mpccm/h").value / lbox.to("Mpccm/h").value[0]
        gasIDs_pre = gasData["PartType0","ParticleIDs"].value
        
        # Getting only particles that have launched
        LLP_arr = np.asarray(gasData["PartType0","LastLaunchPos"].value, dtype=int)
        hasLaunched = LLP_arr != UINT_MAX
        gasCoords = gasCoords_pre[hasLaunched]
        gasIDs = gasIDs_pre[hasLaunched]
        LLPs = LLP_arr[hasLaunched]

        vpmOut = dict()

        for sys in vpmFiles:
            speciesName = "."
            for ionName in ion_lines:
                if ionName in sys:
                    speciesName = ionName
                    break
            sysID, sysVel = np.loadtxt(sys, skiprows=2, usecols=(0,2), unpack=True)
            vpmOut[speciesName] = { "v":sysVel, "ID":sysID }
        
        colSpecArr = []
        colAbsIDArr = []
        colGasIDArr = []

        # sETTING BUFFER
        gal_buffer = gal_bfr
        left_edge = snapFile.domain_left_edge.value
        right_edge = snapFile.domain_right_edge.value
        g_range = [[l, r] for l,r in zip(left_edge,right_edge)] #[[0.,6000.],[0.,6000.],[0.,6000.]]

        # Getting max number of galaxies in index + neighbors given by buffer
        Hist_gal_temp, gal_edges = np.histogramdd(galPos, bins=N_LLP, range=g_range)
        buffer_width = 2*gal_buffer + 1
        sum_filter = np.ones((buffer_width,buffer_width,buffer_width), dtype=int)
        Hist_gal = convolve(Hist_gal_temp, sum_filter, mode="wrap")
                    
        maxCountGal = np.max(Hist_gal)
        del Hist_gal
        del Hist_gal_temp

        colGalIDArr = []

        if nproc >= 1:
            manager = mp.Manager()
            counter = manager.Value('i',0)

            N = gasIDs.size
            dn = int(N / nproc)
            ni = [0]
            for i in range(nproc):
                if i < nproc - 1:
                    temp = ni[i] + dn
                    ni.append(temp)
                else:
                    ni.append(N)

            procsArr = []
            argList = []
            for i in range(nproc):
                colSpecArr.append(manager.list())
                colAbsIDArr.append(manager.list())
                colGasIDArr.append(manager.list())
                colGalIDArr.append(manager.list())
                # colGalIDArr[i] = [-99 for __ in range(maxCountGal)]

                argTup = (gasIDs[ni[i]:ni[i+1]], gasCoords[ni[i]:ni[i+1]], LLPs[ni[i]:ni[i+1]], vpmOut, galPos, galID, colSpecArr[i], colGasIDArr[i], colAbsIDArr[i], colGalIDArr[i], __debugMode__, N, z, Hz, h, DXDZ, DYDZ, r_search, lbox, counter, maxCountGal, gal_buffer)
                #argument to pass into __part__

                # staring multiprocessing
                procsArr.append(mp.Process(target=__part__, args=argTup))
                procsArr[i].start()

            for i in range(len(procsArr)):
                #procsArr[i].join()
                while True:
                    if procsArr[i].is_alive():
                        #global counter
                        __print_complete(counter,N)
                        time.sleep(0.5)
                        continue
                    else:
                        #global counter
                        __print_complete(counter,N)
                        break
        else:
            print("ISSUE: Need nproc >= 1!")
            exit(5)

        colSpecies = []
        colAbsorberID = []
        colGasID = []
        colGalaxyID = []
        for ni in range(nproc):
            if ni == 0:
                colSpecies = np.asarray(list(colSpecArr[ni]), dtype=int)
                colAbsorberID = np.asarray(list(colAbsIDArr[ni]), dtype=int)
                colGasID = np.asarray(list(colGasIDArr[ni]), dtype=int)
                colGalaxyID = np.asarray(list(colGalIDArr[ni]), dtype=int)
                continue
            else:
                colSpecies = np.concatenate([colSpecies,list(colSpecArr[ni])])
                colAbsorberID = np.concatenate([colAbsorberID, list(colAbsIDArr[ni])])
                colGasID = np.concatenate([colGasID, list(colGasIDArr[ni])])

                if not colGalaxyID.size!=0 and list(colGasIDArr[ni]):
                    #print("1 has not, 2 has, CONDITION 1")
                    colGalaxyID = np.asarray(list(colGalIDArr[ni]), dtype=int)
                    continue
                elif colGalaxyID.size!=0 and not list(colGasIDArr[ni]):
                    #print("1 has, 2 has not, CONDITION 2")
                    continue
                elif not colGalaxyID.size!=0 and not list(colGasIDArr[ni]):
                    #print("1 has not, 2 has not, CONDITION 3")
                    continue
                else:
                    #print("1 has, 2 has, CONDITION 4")
                    colGalaxyID = np.concatenate([colGalaxyID, list(colGalIDArr[ni])])

        if __debugMode__:
            f.close()
            return(100)
        refTable = Table([colSpecies, colAbsorberID, colGasID, colGalaxyID], names=("SpeciesLine","AbsorberID", "GasParticleID", "GalaxyIDs"), dtype=[int, int, int, int])
        
        if write:
            fname = f"refTab_r{r_search}_{snapi:03}.hdf5"
            __save_hdf5__(refTable, fname)
            #refTable.write(fname, format="hdf5", path="ReferenceData", compression=True)
    print("\nDone.")
    return(refTable)

def main():
    if len(sys.argv) < 8:
        print("python hostgal.py n_i n_f vpmpath simpath caesarpath r nproc [...]")
        return(1)

    debug = False
    if "-d" in sys.argv or "--debug" in sys.argv:
        debug = True

    multifile = False
    if "-mf" in sys.argv or "--multifile" in sys.argv:
        multifile=True

    bbox = [[0,6000.],[0,6000.],[0,6000.]]
    u = {
    "length": (1.0, "kpc/h"),
    "velocity": (1.0, "km/s"),
    "mass": (1e10, "Msun/h"),
    }

    n_i = int(sys.argv[1])
    n_f = int(sys.argv[2])

    vpmpath = sys.argv[3]
    simpath = sys.argv[4]
    caesarpath = sys.argv[5]
    r = float(sys.argv[6])
    nproc = int(sys.argv[7])

    do_hostgals(vpmpath,simpath,caesarpath,r,n_i=n_i,n_f=n_f,multifile=multifile, nproc=nproc)
    return(0)
    
if __name__ == "__main__":
    main()
