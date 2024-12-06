# This is to test the capabilities of GaussSmooth
# Make the tests their own functions/methods
from julia import Main
import numpy as np
import matplotlib.pyplot as plt


# NOTE: need to run with 'python-jl' command
# TODO: find a precompiling solution on Julia modules to do 
# with setup installation

# test 1: one particle in center of the grid, in center of cell. 
# It is basis to make sure GaussSmooth runs and provides a 
# uniform kernel as control
def test1(retval=False, plot=True): # retval: return the test gaussian grid for comparisons
    
    testL = 10.0 # test length
    testGrid = np.zeros([21,21,21]) # gotta be odd to have center be cell
    testCoord = np.ones(3) * (testL/2) # 5 will be center 
    sphToGauss = 0.33479
    testSL = 1.0 / sphToGauss # so sigma = 1
    # if 5 sigma limit true on code, values should barely
    # reach middle cells in center of face
    testGL = testGrid.shape[0]
    
    Main.include("../pycosie/utils/GaussSmooth.jl")
    testGauss = Main.gaussLoop(testGL, testCoord, testSL, testL)
    
    if plot==True:
        plt.imshow(testGauss[10,:,:]) # central slice in i(x) index
        plt.title("Test1")
        plt.colorbar()
        plt.show()
    if retval==False:
        return None
    else:
        return testGauss
    
# test 1: one particle in center of the grid, near bottom of cell. 
# It will test if it handles an off center value
def test2(retval=False, plot=True): # retval: return the test gaussian grid for comparisons
    testL = 10.0 # test length
    testGrid = np.zeros([21,21,21]) # gotta be odd to have center be cell
    dL = testL/testGrid.shape[0]
    testCoord = np.array([testL/2, testL/2 + 0.45*dL, testL/2]) # 5 will be center 
    sphToGauss = 0.33479
    testSL = 1.0 / sphToGauss # so sigma = 1
    # if 5 sigma limit true on code, values should barely
    # reach middle cells in center of face
    testGL = testGrid.shape[0]
    
    Main.include("../pycosie/utils/GaussSmooth.jl")
    testGauss = Main.gaussLoop(testGL, testCoord, testSL, testL)
    
    if plot==True:
        plt.imshow(testGauss[10,:,:]) # central slice in i(x) index
        plt.title("Test2")
        plt.colorbar()
        plt.show()
    if retval==False:
        return None
    else:
        return testGauss
# test 3: particle near grid edge
def test3(retval=False, plot=True):
    
    testL = 10.0 # test length
    testGrid = np.zeros([21,21,21]) # gotta be odd to have center be cell
    dL = testL/testGrid.shape[0]
    testCoord = np.array([testL/2, testL/2 + 8.1*dL, testL/2 + 8.4*dL]) # 5 will be center 
    sphToGauss = 0.33479
    testSL = 1.0 / sphToGauss # so sigma = 1
    # if 5 sigma limit true on code, values should barely
    # reach middle cells in center of face
    testGL = testGrid.shape[0]
    
    Main.include("../pycosie/utils/GaussSmooth.jl")
    testGauss = Main.gaussLoop(testGL, testCoord, testSL, testL)
    
    if plot==True:
        plt.imshow(testGauss[10,:,:]) # central slice in i(x) index
        plt.title("Test3")
        plt.colorbar()
        plt.show()
    if retval==False:
        return None
    else:
        return testGauss


if __name__ == "__main__": 
    # run the code and can switch out which functions you are using
    g1 = test1(retval=True, plot=True)
    g2 = test2(retval=True, plot=True)
    #g3 = test3(retval=True, plot=True)
    result = g1 - g2
    plt.imshow(result[10,:,:])
    plt.title("Residual")
    plt.colorbar()
    plt.show()