import numpy as np
import caesar
import yt
from glob import glob
from astropy.io import ascii
from astropy import table as tab
from unyt import unyt_array
import re
from yt.utilities.cosmology import Cosmology

class RockstarCatalog():

    def __init__(self, filebase=None, filename=None, units=None):
        if filebase != None:
            fname = glob(filebase)[0]
        elif filename != None:
            fname = filename
        with open(fname,"r") as f:
            s = []
            contents = f.read().split("\n")
            for x in contents:
                if "Om" in x and "Ol" in x and "h" in x:
                    chars = set(".;0123456789\n")
                    line = np.array([i for i in re.findall(r'\w+\D',x) if not any((c in chars) for c in i)], dtype=str)
                    s = np.array(re.findall(r"[-+]?(?:\d*\.\d+|\d+)",x), dtype=str)
            i_Om = np.where(line=="Om")
            i_Ol = np.where(line=="Ol")
            i_h = np.where(line=="h")
            self.Om = float(s[i_Om])
            self.Ol = float(s[i_Ol])
            self.Or = 1 - self.Om - self.Ol
            self.h = float(s[i_h])
            self.cosmo = Cosmology(
                hubble_constant=self.h,
                omega_matter=self.Om,
                omega_lambda=self.Ol,
                omega_curvature=0.0,
                omega_radiation=self.Or,
            )


        Tab, self.units = self.read_rockstar(filebase=filebase, filename=filename, units=units)
        print(Tab)
        self.ids = Tab["id"].value
        self.mvir = self.cosmo.arr(Tab["mvir"].value, self.units["mass"])
        self.rvir = self.cosmo.arr(Tab["rvir"].value, self.units["halo_dist_rad"])
        pos = np.array([Tab["x"].value,Tab["y"].value,Tab["z"].value]).T
        self.pos = self.cosmo.arr(pos, self.units["pos"])
        vel = np.array([Tab["vx"].value,Tab["vy"].value,Tab["vz"].value]).T
        self.vel = self.cosmo.arr(pos, self.units["pos"])
        

    def read_rockstar(self, filename=None, filebase=None, units=None):
    # id num_p mvir mbound_vir rvir vmax rvmax vrms x y z vx vy vz Jx Jy Jz E Spin PosUncertainty VelUncertainty bulk_vx bulk_vy bulk_vz BulkVelUnc n_core m200b m200c m500c m2500c Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) Rs Rs_Klypin T/|U| M_pe_Behroozi M_pe_Diemer idx i_so i_ph num_cp mmetric
        if units == None:
            units = {
                "pos": "Mpccm/h",
                "vel": "km/s",
                "mass": "Msun/h",
                "halo_dist_rad": "kpccm/h",
                "ang_mom": "(Msun/h)*(Mpc/h)*(km/s)",
                "spin": "dimensionless",
                "tot_eng": "(Msun/h)*(km/s)**2"
            }

        if filebase != None:
            fnames = glob(filebase)
            t = []
            for fname in fnames:
                t.append(ascii.read(fname, format="commented_header"))
            main_table = tab.vstack(t)
        elif filename != None:
            fname = filename
            main_table = ascii.read(fname, format="commented_header")
        return main_table, units
