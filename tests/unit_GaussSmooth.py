# This is to test the capabilities of GaussSmooth
# Make the tests their own functions/methods
from julia import Main
import numpy as np

# Throw in some known answers to test if will truly be an off-center 
# gaussian smoothing kernel.

# test 1: one particle in center of a grid, in center of cell. 
# It is basis to make sure GaussSmooth runs and provides a 
# uniform kernel as control
def test1():
    return None


if __name__ == "__main__": 
    # run the code and can switch out which functions you are using
    test1()