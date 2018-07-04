# This will load the parameter files to be worked with
# In future iterations this will be more interactive
from parameterFile import parameterFile


def loadParameFiles(mainFileName, bckgrndFileName, geomFileName, fluxFileName):

    # TODO: Create emptyGeometry and emptyFlux class methods
    # Created two parameterFile objects with all parameter states
    # initialized to False
    main = parameterFile.mainFile(mainFileName)
    bckgrnd = parameterFile.backgroundFile(bckgrndFileName)

    return main, bckgrnd


if __name__ == '__main__':
    loadParameFiles("mainTest", "bckgrndTest", "goemTest", "literalShit")
