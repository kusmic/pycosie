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
N_LLP = 1625 # Hard-coded grid space for LLP grid
UINT_MAX = 4294967295 # flag for unlaunched

xEdges = np.linspace(0.,1.,N_LLP)
yEdges = np.linspace(0.,1.,N_LLP)
zEdges = np.linspace(0.,1.,N_LLP)
# Grid edges for LLP grid

#print(len(xEdges))

def __save_hdf5__(table, fname):
    """Save to HDF5

    This function will save the inputted table into an HDF5 file holding both 
    the "ReferenceData," which holds all the object IDs linked together from 
    the host galaxy search, and "SpeciesLookup," which is a cheat-sheet on which
    integer value corresponds to which species line.

    Parameters
    ----------
    table: astropy.table.Table
        Table object holding all the reference data.
    
    fname: string
        String holding the file path name to save to.

    Returns
    -------
    None.
    """

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
    """Print Complete

    This function prints the progress of number of particles processed.

    Parameters
    ----------
    counter: multiprocessing.Manager.value(int)
        Shared counter value among processes that upticks for each processed particle.
    N: int
        Total number of particles (launched usually).
    
    Returns
    -------
    None.
    """

    # Takes Manager.Value object starting at 0, integer total of particles N
    # Prints out completed particles
    percent = f"{counter.value} / {N}"
    percentage = f"..... {percent} launched particles processed"
    print(percentage, end="\r")

def __part__(gasIDArr, gasCoordArr, gasLLPArr, vpmDict, galPosArr, galIDArr, 
             colSpecies, gasIDOut, vpmIDOut, galIDOut, __debugMode__, N, z, 
             Hz, h, DXDZ, DYDZ, r_search, lbox, counter, maxCountGal, 
             gal_buffer, f=None):
    """PART

    This non-usable, iteratable function is what is passed to the multiprocessing.Process
    list to start the splitted work.

    This does the whole bulk of the host galaxy finding: getting absorber 
    position, checking if gas particles in search radius, getting galaxy 
    position per LLP grid.

    This does not return anything technically, since it uses 
    multiprocessing.Manager.list quantities to save outputs by reference.

    Parameters
    ----------
    gasIDArr: numpy.ndarray
        Holds the (sliced) section of gas IDs.
    gasCoordArr: numpy.ndarray
        Holds the (sliced) section of gas positions.
    gasLLPArr: numpy.ndarray
        Holds the (sliced) section of gas LLP values.
    vpmDict: dict
        Holds the values for the velocity and ID of the absorber system, 
        implemented to reduce reading the vpm file numerous times.
    galPosArr: numpy.ndarray
        Holds the (sliced) section of galaxy positions.
    galIDArr: numpy.ndarray
        Holds the (sliced) section of galaxy IDs.
    colSpecies: multiprocessing.Manager.list
        Referenced list to save species IDs.
    gasIDOut: multiprocessing.Manager.list
        Referenced list to save gas IDs.
    vpmIDOut: multiprocessing.Manager.list
        Referenced list to save absorber system IDs.
    galIDOut: multiprocessing.Manager.list
        Referenced list to save galaxy IDs.
    __debugMode__: bool
        Whether to turn on debug mode (output file of ALL LLPs of gas searched).
    N: int
        Number gas particles in total.
    z: float
        Redshift.
    Hz: float
        Hubble constant at redshift z.
    h: float
        Hubble parameter, = H_0/100
    DXDZ: float
        x-axis gradient w.r.t. z-axis for the casted ray.
    DYDZ: float
        y-axis gradient w.r.t. z-axis for the casted ray.
    r_search: float
        Search radius to see if gas particle in absorber system range.
    lbox: yt.unyt_array
        Holds the dimensions and measurements of the simulation box.
    counter: multiprocessing.Manager.value(int)
        Shared counter value among processes that upticks for each processed 
        particle.
    maxCountGal: int
        Maximum number of galaxies in a grid cell for LLP grid.
    gal_buffer: int
        Index buffer to search around LLP grid.
    f: File, default=None
        File stream for debug mode. Defaults to None if not debug mode.

    Returns
    -------
    None.
    """
    # The searching

    for gi in range(gasIDArr.size):
        counter.value += 1
        # rint(f"..... {percent:.3f}% launched particles processed", end="\r")
        pos = gasCoordArr[gi] # This is in box units
        # percent = (gi+1)
        for ioni, ion in enumerate(ion_lines):
            sysID = vpmDict[ion]["ID"]
            sysVel = vpmDict[ion]["v"] # getting absorber ID and velocity
            for vi, v in enumerate(sysVel):
                xStart, yStart, zStart = 0., 0., 0.
                dSys = (v / Hz) * (1+z) * h # comoving h-1 Mpc
                dSys /= lbox.to("Mpccm/h").value[2] # in box units
                dz = dSys / np.sqrt(DXDZ**2 + DYDZ**2 + 1) # In box
                dx = dz * DXDZ
                dy = dz * DYDZ # getting (dx, dy, dz)
                xSys = xStart + dx 
                xSys -= np.floor(xSys)
                ySys = yStart + dy
                ySys -= np.floor(ySys)
                zSys = zStart + dz
                zSys -= np.floor(zSys) # converting them all to box space

                # ADD check if x, y, z already larger than r_search
                r_s = r_search / lbox.to("kpccm/h").value[0]
                #if xSys > r_s or ySys > r_s or zSys > r_s:
                #    continue
                
                rSys = np.array([xSys, ySys, zSys])
                rDiff = np.sqrt( np.sum( (pos-rSys)**2 ) ) # box unit - box unit
                    
                if rDiff > r_s:
                    continue
                else:
                    # print("GOT!")
                    LLP = gasLLPArr[gi] # LLP value of gas
                    # UINT_MAX check up here, continue if flag
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
                            partID = gasIDArr[gi]
                            f.write(f"{partID} {LLP} {pos[0]} {pos[1]} {pos[2]} {igas} {jgas} {kgas}\n")
                        continue
                    # creating array that will return with Manager.list
                    # based on moax counts on galaxies in LLP grid + buffer
                    galOfLLP = np.ones(int(maxCountGal)) * -99 
                    ind_ret = 0
                    # indexing value for this returned grid

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
                        galp /= lbox.to("kpccm/h").value[0] #CHECKME if outputs still bad can have bad units here

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
                            galOfLLP[ind_ret] = galIDArr[gali]
                            ind_ret += 1
                        # Now extracting all data out
                    # print(galOfLLP)
                    colSpecies.append(ioni)
                    gasIDOut.append(gasIDArr[gi])
                    vpmIDOut.append(sysID[vi])
                    galIDOut.append(galOfLLP)
                    # print(ion, gasIDs[gi], sysID[vi], galOfLLP)


def do_hostgals(vpmpath, simpath, caesarpath, r_search, bbox=None, unit_base=None, n_i=0, n_f=None, merged=True, N_LLP=N_LLP, multifile=True, write=True, __debugMode__ = False, gal_bfr=1, nproc=1):
    """Do Hostgals

    This is the user-interfacing method to run the host galaxy searching.

    Defining the necessary paths and search radius, one can find run the
    process and get the reference table

    Parameters
    ----------
    vpmpath: string
        Path to the directory holding all the VPM and/or VPM.MERGED files.
    simpath: string
        Path to the directory holding the simulation snapshots. This can be
        the files themselves (e.g. snap_..._XYZ.hdf5) or to the directories
        holding the multi-file output (e.g. snapdir_XYZ).
    caesarpath: string
        Path the the files outputed by caesar. This should hold your halo and 
        galaxy catalogs.
    r_search: float
        (IN ckpc/h) the searching radius for the particles around a given
        absorber.
    bbox: array-like
        Holds the upper and lower limit of the bounds of the simulation
        box in (x,y,z)
    unit_base: dict
        This holds the code units for length, velocity, and mass of your
        outputs, fed into yt. It's exactly the same structure as that of yt's
    n_i: int, default=0
       Initial snapshot value to start on.
    n_f: int, default=None
        Final snapshot to analyze. If None, then calculates the number by first
        getting the number of snapshots in the 'simpath' directory.
    merged: bool, default=True
        Tell whether to use the vpm.merged files rather than the vpm files.
    N_LLP: int, default=1625
        Grid width of the Last Launch Position (LLP) grid. This assumes the 
        grid is cubic.
    multifile: bool, default=True
        Tell whether the snapshot files are outputted in multifile mode.
    write: bool, default=True
        Tell whether to write the reference table into an HDF5 file.
    __debugMode__: bool, default=False
        Tell whether to enter debug mode. This is a developer's tool and not
        necessary for a user.
    gal_bfr: int, default=1
        Buffer zone searching around the LLP grid. As in, this looks an extra
        number of indices around a chosen index for a galaxy. It is a square
        search.
    nproc: int, default=1
        Number of processor cores to use for the analysis.

    Returns
    -------
    refTable: astropy.table.Table
        Reference table holding all gas particles associated with an absorber
        and linked to a galaxy. This will output when NOT run in debug mode.
    100:
        This output signals that the method was run in debug mode.

    """
    if __debugMode__: # Running debug mode
        f = open("particle_LLP_debug.txt", "w") # create debug output file
        f.write("ParticleID LLP x y z i j k\n") # column names
        f.close()
        f = open("particle_LLP_debug.txt", "a") # now in append mode
    else:
        f = None

    print(f"Run on {datetime.now()}\n") # tell time of run
    DXDZ = 0.35171 # differential movement w.r.t. z for x and y, used to find
    DYDZ = 0.0113  # the casting angle. Hard-coded in arcosie programs.
    # In box length units

    # Standardizing path strings, to make sure read proper
    if '/' not in vpmpath[-1]:
        vpmpath = vpmpath + "/"
    if '/' not in simpath[-1]:
        simpath = simpath + "/"
    if '/' not in caesarpath[-1]:
        caesarpath = caesarpath + "/"

    for snapi in range(n_i,n_f+1): # for each snapshot and associated data

        # Getting all files in each directory
        
        if merged: # if using vpm.merged files
            globStr = vpmpath + f"*_{snapi:03}.vpm.merged"
            vpmFiles = glob.glob(globStr)
        else:
            globStr = vpmpath + f"*_{snapi:03}.vpm"
            vpmFiles = glob.glob(globStr)

        if multifile: # if outputted in multifile mode
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
        lbox = snapFile.domain_width.to("kpccm/h")# box size in ckpc/h
        h = snapFile.hubble_constant # Hubble parameter
        OmegaM = snapFile.omega_matter # fraction of matter now
        OmegaL = snapFile.omega_lambda # fraction of dark energy now
        z = snapFile.current_redshift # redshift of snapshot

        H0 = h * 100 # Hubble constant at now
        Hz = H0 * np.sqrt((OmegaM * np.power(1+z,3)) + OmegaL)
        # Hubble constant at that redshift

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
        # Telling which snapshot being worked on
        print(f"Working on snap {snapi:03}...")
        
        # now culling out particles to only analyze those that launched
        gasCoords_pre = gasData["PartType0", "Coordinates"].to("Mpccm/h").value / lbox.to("Mpccm/h").value[0]
        gasIDs_pre = gasData["PartType0","ParticleIDs"].value
        
        # Getting only particles that have launched
        LLP_arr = np.asarray(gasData["PartType0","LastLaunchPos"].value, dtype=int)
        hasLaunched = LLP_arr != UINT_MAX
        gasCoords = gasCoords_pre[hasLaunched]
        gasIDs = gasIDs_pre[hasLaunched]
        LLPs = LLP_arr[hasLaunched]
        # final arrays to analyze of gas ID, position, and LLPs after culling

        vpmOut = dict()
        # Will read in all species, create single referenced dictionary
        for sys_abs in vpmFiles:
            speciesName = "."
            for ionName in ion_lines:
                if ionName in sys_abs:
                    speciesName = ionName
                    break
            sysID, sysVel = np.loadtxt(sys, skiprows=2, usecols=(0,2), unpack=True)
            vpmOut[speciesName] = { "v":sysVel, "ID":sysID }
        
        
        colSpecArr = []
        colAbsIDArr = []
        colGasIDArr = []

        # Setting galaxy buffer
        gal_buffer = gal_bfr
        left_edge = snapFile.domain_left_edge.value
        right_edge = snapFile.domain_right_edge.value
        g_range = [[l, r] for l,r in zip(left_edge,right_edge)]

        # Getting max number of galaxies in index + neighbors given by buffer
        Hist_gal_temp, gal_edges = np.histogramdd(galPos, bins=N_LLP, range=g_range)
        buffer_width = 2*gal_buffer + 1
        # creating square filter
        sum_filter = np.ones((buffer_width,buffer_width,buffer_width), dtype=int)
        # convolve on square filter == summing around indices
        Hist_gal = convolve(Hist_gal_temp, sum_filter, mode="wrap")
                    
        maxCountGal = np.max(Hist_gal)
        del Hist_gal
        del Hist_gal_temp

        colGalIDArr = []

        # now running, and doing multiprocessing
        if nproc >= 1: # check to make sure nproc is reasonable number
            # creating Manager to "talk" between all Processes and
            # collect their outputted reference data
            manager = mp.Manager()
            counter = manager.Value('i',0)

            # Splitting into sub-arrays for multiprocessing
            N = gasIDs.size
            dn = int(N / nproc) # split into almost equal groups per core
            ni = [0] # start at 0 index
            for i in range(nproc):
                if i < nproc - 1:
                    temp = ni[i] + dn # get index of ith core's end slice
                    ni.append(temp)
                else:
                    ni.append(N) # add last index
            # This creates the indices to slice the main array

            procsArr = []
            # Now starting multiprocessing
            for i in range(nproc):
                colSpecArr.append(manager.list())
                colAbsIDArr.append(manager.list())
                colGasIDArr.append(manager.list())
                colGalIDArr.append(manager.list())
                # creating manager lists to append to in processes

                # creating ugly argument tuple to pass into Process
                argTup = (gasIDs[ni[i]:ni[i+1]], gasCoords[ni[i]:ni[i+1]], 
                          LLPs[ni[i]:ni[i+1]], vpmOut, galPos, galID, 
                          colSpecArr[i], colGasIDArr[i], colAbsIDArr[i], 
                          colGalIDArr[i], __debugMode__, N, z, Hz, h, DXDZ, 
                          DYDZ, r_search, lbox, counter, maxCountGal, 
                          gal_buffer, f)
                #argument to pass into __part__

                # staring multiprocessing
                procsArr.append(mp.Process(target=__part__, args=argTup))
                procsArr[i].start()

            # Now hang out and make sure all Processes are finished
            # also output completion progress
            for i in range(len(procsArr)):
                while True:
                    if procsArr[i].is_alive(): # if this Process is working
                        #global counter
                        __print_complete(counter,N)
                        time.sleep(0.5)
                        continue # stay with it
                    else: # if it's finished
                        #global counter
                        __print_complete(counter,N)
                        break # go to the next
        else: # if non-reasonable number given to nproc
            raise RuntimeError("ISSUE: Need nproc >= 1!")
            exit(5)

        # Final collecting arrays to put into reference table
        colSpecies = []
        colAbsorberID = []
        colGasID = []
        colGalaxyID = []
        for ni in range(nproc): # for each processor core Process
            if ni == 0: # This is the first one, just copy arrays
                colSpecies = np.asarray(list(colSpecArr[ni]), dtype=int)
                colAbsorberID = np.asarray(list(colAbsIDArr[ni]), dtype=int)
                colGasID = np.asarray(list(colGasIDArr[ni]), dtype=int)
                colGalaxyID = np.asarray(list(colGalIDArr[ni]), dtype=int)
                continue
            else: # Need to concatenate otherwise, unless it's galaxy IDs...
                colSpecies = np.concatenate([colSpecies,list(colSpecArr[ni])])
                colAbsorberID = np.concatenate([colAbsorberID, list(colAbsIDArr[ni])])
                colGasID = np.concatenate([colGasID, list(colGasIDArr[ni])])

                if not colGalaxyID.size!=0 and list(colGasIDArr[ni]):
                    # collector empty but Process array has values
                    colGalaxyID = np.asarray(list(colGalIDArr[ni]), dtype=int)
                    continue
                elif colGalaxyID.size!=0 and not list(colGasIDArr[ni]):
                    # Collector has value but Process array have values
                    continue
                elif not colGalaxyID.size!=0 and not list(colGasIDArr[ni]):
                    # Neither has values
                    continue
                else:
                    # Both have values
                    colGalaxyID = np.concatenate([colGalaxyID, list(colGalIDArr[ni])])
                # This needed to be made because there was an error when 
                # concatenating array of arrays and a null array. This does
                # not appear in array of scalars and null arrays for some
                # reason.

        if __debugMode__: # finishing debug mode
            f.close()
            return(100)
            
        # Creating reference table
        refTable = Table([colSpecies, colAbsorberID, colGasID, colGalaxyID], names=("SpeciesLine","AbsorberID", "GasParticleID", "GalaxyIDs"), dtype=[int, int, int, int])
        
        if write: # if you want to write it out
            fname = f"refTab_r{r_search}_{snapi:03}.hdf5"
            __save_hdf5__(refTable, fname)
            #refTable.write(fname, format="hdf5", path="ReferenceData", compression=True)
    print(f"\nDone on {datetime.now()}\n.")
    return(refTable) # all done!

#def main():
#    if len(sys.argv) < 8:
#        print("python hostgal.py n_i n_f vpmpath simpath caesarpath r nproc [...]")
#        return(1)
#
#    debug = False
#    if "-d" in sys.argv or "--debug" in sys.argv:
#        debug = True

#    multifile = False
#    if "-mf" in sys.argv or "--multifile" in sys.argv:
#        multifile=True

#    bbox = [[0,6000.],[0,6000.],[0,6000.]]
#    u = {
#    "length": (1.0, "kpc/h"),
#    "velocity": (1.0, "km/s"),
#    "mass": (1e10, "Msun/h"),
#    }

#    n_i = int(sys.argv[1])
#    n_f = int(sys.argv[2])

#    vpmpath = sys.argv[3]
#    simpath = sys.argv[4]
#    caesarpath = sys.argv[5]
#    r = float(sys.argv[6])
#    nproc = int(sys.argv[7])
#
#    do_hostgals(vpmpath,simpath,caesarpath,r,n_i=n_i,n_f=n_f,multifile=multifile, nproc=nproc)
#    return(0)
    
#if __name__ == "__main__":
#    main()
