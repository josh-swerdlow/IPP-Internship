########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: This will load the parameter files to be worked with.

Details: Calls the local private function _loadParamFile
    to load

To Do:

"""

__author__ = 'Joshua Swerdow'

import os

import auxillary as aux

from file import ParameterFile

def loadParamFiles(mainFileName, bckgrndFileName, geomFileName, fluxFileName):

    # TODO: Create emptyGeometry and emptyFlux class methods
    # Created two ParameterFile objects with all parameter states
    # initialized to False
    mainPath = "param"
    bckgrndPath = "nete"
    geomPath = bckgrndPath
    fluxPath = bckgrndPath

    main = _loadParamFile(mainFileName, mainPath, "main")
    bckgrnd = _loadParamFile(bckgrndFileName, bckgrndPath, "background")

    return main, bckgrnd


def _loadParamFile(fileName, filePath, fileType):
    paramFilePath = os.path.join(filePath, fileName)

    paramFile = None
    if os.path.isfile(paramFilePath):
        if fileType is "main":
            paramFile = ParameterFile.mainFile(paramFilePath)
        elif fileType is "background":
            paramFile = ParameterFile.backgroundFile(paramFilePath)
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
