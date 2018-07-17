# This will load the parameter files to be worked with
# In future iterations this will be more interactive
from file import parameterFile

import auxillary as aux

import os


def loadParamFiles(mainFileName, bckgrndFileName, geomFileName, fluxFileName):

    # TODO: Create emptyGeometry and emptyFlux class methods
    # Created two parameterFile objects with all parameter states
    # initialized to False
    mainPath = "param"
    bckgrndPath = "nete"
    geomPath = bckgrndPath
    fluxPath = bckgrndPath

    main = loadParamFile(mainFileName, mainPath, "main")
    bckgrnd = loadParamFile(bckgrndFileName, bckgrndPath, "background")

    return main, bckgrnd


def loadParamFile(fileName, filePath, fileType):
    paramFilePath = os.path.join(filePath, fileName)

    paramFile = None
    if os.path.isfile(paramFilePath):
        if fileType is "main":
            paramFile = parameterFile.mainFile(paramFilePath)
        elif fileType is "background":
            paramFile = parameterFile.backgroundFile(paramFilePath)
        elif fileType is "geometry":
            pass
        elif fileType is "flux":
            pass

    else:
        file = aux.colorFile(fileName)
        path = aux.colorDir(filePath)

        print("Could not find {} in {}!\n".format(file, path))

        aux.print_dirContents(filePath)

        newFileName = input("Please enter a new file or exit [enter]: ")

        if newFileName is not '':
            paramFile = loadParamFile(newFileName, filePath, fileType)

    return paramFile

if __name__ == '__main__':
    loadParamFiles("op12a_171122022_FeLBO3", "bckgrndTest", "goemTest", "literalShit")
