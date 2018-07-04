# This will load the parameter files to be worked with
# In future iterations this will be more interactive
from parameterFile import parameterFile
import os
import termcolor


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
        file = colorFile(fileName)
        path = colorDir(filePath)
        print("Could not find {} in {}!\n".format(file, path))

        print_dirContents(filePath)

        newFileName = input("Please enter a new file or exit [enter]: ")
        if newFileName is not '':
            paramFile = loadParamFile(newFileName, filePath, fileType)

    return paramFile


def print_dirContents(path):

    files = os.listdir(path)

    for file in files:
        f = os.path.join(path, file)

        if os.path.isfile(f):
            print("  {}  ".format(colorFile(file)), end='')
        elif os.path.isdir(f) is True:
            print("  {}  ".format(colorDir(file)), end='')
    print("\n")


def colorFile(file):
    return termcolor.colored(file, 'blue')


def colorDir(dir):
    return termcolor.colored(dir, 'green')


if __name__ == '__main__':
    loadParamFiles("op12a_171122022_FeLBO3", "bckgrndTest", "goemTest", "literalShit")
