import numpy as np
from rockstar import RockstarCatalog
import yt
from yt.data_objects.level_sets.api import *    

class GalFinder():

    def __init__(self, yt_ds, catalog="rockstar", rock_cat=None):
        self.yt_ds = yt_ds
        if catalog=="rockstar" and rock_cat!=None:
            self.halo_cat = rock_cat
        else:
            raise RuntimeError("Expected ROCKSTAR catalog, but missing argument definition.")
                
    def __run_clump(self):
        N = self.halo_cat.mvir.size
        for i in range(N):
            c = self.halo_cat.pos[i]
            r = (self.halo_cat.rvir, self.halo_cat.units["halo_dist_rad"])
            self.MASTER_CLUMP = Clump(c,r)
            
